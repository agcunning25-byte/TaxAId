import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import LlamaCpp
import re
import warnings
warnings.filterwarnings("ignore")

# ======================
# Paths
# ======================

DB_PATH = "./chroma_tax_db"
COLLECTION_NAME = "irs_tax_publications"
MODEL_PATH = "./models/Phi-3-mini-4k-instruct-q4.gguf"

# ======================
# Load Embeddings + Vector DB
# ======================

print("Loading vector database...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory=DB_PATH,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

print("Vector DB loaded. Documents:", vectordb._collection.count())

# ======================
# Load LLM
# ======================

print("Loading Phi-3 Mini model...")

llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.1,   # more deterministic
    max_tokens=180,
    n_ctx=2048,
    n_threads=12,
    n_batch=512,
    verbose=False,
)

print("Model loaded.")

# ======================
# Prompt Template (General + Grounded)
# ======================

template = """
You are TaxAId, an AI assistant that answers questions strictly using retrieved IRS publications.

Use ONLY the provided context.
If the answer is not in the context, say:
"I cannot find this information in the provided IRS publications."

Provide one concise answer (under 120 words).
Do not include citations inside the answer. 
Instead include exactly one citation line AFTER the answer in this format:

Source: <publication filename>

Do not repeat the answer.
Do not restate the question.
Do not provide multiple answers.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate.from_template(template)

# ======================
# Answer generation with similarity search and metadata routing
# ======================

def generate_answer_stream(question):
    question = question.replace("’", "'")
    question_lower = question.lower()

    print("Starting similarity search...")

    # -------------------
    # Routing logic
    # -------------------
    if "depreciation" in question_lower:
        print("Routing to: depreciation")
        docs = vectordb.similarity_search(
            question,
            k=2,
            filter={"topic": "depreciation"}
        )

    elif any(re.search(rf"\b{word}\b", question_lower) for word in[
        "mileage", "car", "vehicle", "commuting", "commute"
    ]):
        print("Routing to: travel_vehicle")
        docs = vectordb.similarity_search(
            question,
            k=2,
            filter={"topic": "travel_vehicle"}
        )

    elif any(word in question_lower for word in [
        "medical", "dental", "doctor", "hospital", "treatment"
    ]):
        print("Routing to: medical_expenses")
        docs = vectordb.similarity_search(
            question,
            k=2,
            filter={"topic": "medical_expenses"}
        )

    elif any(word in question_lower for word in [
        "child care", "dependent care", "credit for child"
    ]):
        print("Routing to: dependent_care_credit")
        docs = vectordb.similarity_search(
            question,
            k=2,
            filter={"topic": "dependent_care_credit"}
        )
    
    elif any(word in question_lower for word in [
        "dependent", "qualifying child", "standard deduction", "filing status"
    ]):
        print("Routing to: filing_dependents")
        docs = vectordb.similarity_search(
            question,
            k=2,
            filter={"topic": "filing_dependents"}
        )

    elif any(word in question_lower for word in [
        "hsa", "health savings", "medical savings", "fsa"
    ]):
        print("Routing to: health_accounts")
        docs = vectordb.similarity_search(
            question,
            k=2,
            filter={"topic": "health_accounts"}
        )

    elif any(re.search(rf"\b{word}\b", question_lower) for word in [
        "owe", "installment", "levy", "lien", "collection",
        "payment plan", "cannot pay", "can't pay", "cant pay"
    ]):
        print("Routing to: irs_collections")
        docs = vectordb.similarity_search(
            question,
            k=2,
            filter={"topic": "irs_collections"}
        )

    elif any(word in question_lower for word in [
        "taxable income", "nontaxable", "income include", "1099", "w2", "gross income"
    ]):
        print("Routing to: income_rules")
        docs = vectordb.similarity_search(
            question,
            k=2,
            filter={"topic": "income_rules"}
        )

    elif "business expense" in question_lower:
        print("Routing to: business_expenses")
        docs = vectordb.similarity_search(
            question,
            k=2,
            filter={"topic": "business_expenses"}
        )

    else:
        print("Routing to: full semantic search")
        docs = vectordb.similarity_search(
            question,
            k=2
        )

    print("Similarity search complete.")

    # -------------------
    # Build context
    # -------------------
    context = "\n\n".join([doc.page_content for doc in docs])

    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    print("Starting LLM stream...")

    # Generate full response
    response = llm.invoke(formatted_prompt)
    response = response.strip()

    # Remove Phi-3 chat artifact
    response = response.replace("<|AI_response|>", "")

    # Remove internal source citation hallucinations
    response = re.sub(r"<source:.*?>", "", response, flags=re.IGNORECASE)

    # Remove obvious prompt leakage artifacts
    response = re.sub(r"(===|Solution>|Instruction>|- \[response\]:)", "", response)

    # Split into paragraphs
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]

    # Keep ONLY the first paragraph
    clean_answer = paragraphs[0] if paragraphs else response

    # Remove any accidental Source lines in answer
    clean_answer = re.sub(r"Source:\s*\S+", "", clean_answer).strip()

    # Collapse whitespace
    clean_answer = " ".join(clean_answer.split())

    # Attach citation ourselves
    if not docs:
        yield "I cannot find this information in the provided IRS publications."
        return

    primary_source = docs[0].metadata.get("source")

    final_response = f"{clean_answer}\n\nSource: {primary_source}"

    yield final_response


# ====================
# Gradio UI
# ====================

import gradio as gr

with gr.Blocks() as demo:

    gr.Image("logo.png", width=400)

    gr.Markdown(
        """
        # TaxAId
        ### Grounded Answers from Official IRS Publications

        Ask a question about IRS tax rules.
        Responses are generated using retrieved IRS documents.

        ---
        """
    )

    chatbot = gr.Chatbot(height=250, render_markdown=True)

    user_input = gr.Textbox(
        placeholder="Ask a tax question (e.g. Can I deduct mileage for my car?)",
        label="Your Question"
    )

    submit_btn = gr.Button("Ask TaxAId")

    def respond(message, chat_history):
        chat_history = chat_history or []

        # Add user message
        chat_history.append({
            "role": "user",
            "content": message
        })

        # Add assistant placeholder
        chat_history.append({
            "role": "assistant",
            "content": "🧠 TaxAId is reviewing IRS publications..."
        })

        # Show immediately
        yield chat_history, ""

        # Generate answer
        response = ""
        for chunk in generate_answer_stream(message):
            response += chunk

        # Replace ONLY assistant placeholder
        chat_history[-1] = {
            "role": "assistant",
            "content": response
        }

        yield chat_history, ""

    submit_btn.click(
        respond,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input],
        show_progress="hidden",
    )

    gr.Markdown(
        """

        ---
        ⚠️ **Disclaimer:** TaxAId provides informational guidance based on IRS publications - but AI is known to hallucinate.
        It should not be used as a substitute for professional tax advice.
        """
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

