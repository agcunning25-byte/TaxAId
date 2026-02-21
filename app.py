import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import LlamaCpp
import warnings
warnings.filterwarnings("ignore")

# ======================
# Paths
# ======================

DB_PATH = "./chroma_tax_db"
COLLECTION_NAME = "irs_tax_publications"
MODEL_PATH = "./models\Phi-3-mini-4k-instruct-q4.gguf"

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
    temperature=0.2,
    max_tokens=160,
    n_ctx=1536,
    n_threads=12,
    n_batch=512,
    stop=["\nQuestion:", "\nInstructions:"],
    verbose=False,
    streaming=True
)

print("Model loaded.")

# ======================
# Prompt Template (General + Grounded)
# ======================

template = """
You are an AI assistant that answers questions using official IRS publications.

Use ONLY the provided context from IRS documents.
Do NOT use prior knowledge.
If the information is not in the context, say:
"I cannot find this information in the provided IRS publications."

Context:
{context}

Question:
{question}

Instructions:
1. Identify the relevant IRS rule from the context.
2. Provide a clear explanation in plain language.
3. Clearly state the conclusion.
4. Cite the IRS publication filename(s) clearly.
5. Keep the entire answer under 150 words.
6. Be brief, concise, adn avoid repetition.

Provide your answer below and stop after completing the answer.

Answer:
"""

prompt = PromptTemplate.from_template(template)

# ======================
# Answer generation with similarity search and metadata routing
# ======================

def generate_answer_stream(question):
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

    elif (
        "mileage" in question_lower
        or "car" in question_lower
        or "vehicle" in question_lower
        or "commuting" in question_lower
        or "commute" in question_lower
    ):
        print("Routing to: travel_vehicle")
        docs = vectordb.similarity_search(
            question,
            k=2,
            filter={"topic": "travel_vehicle"}
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

    for chunk in llm.stream(formatted_prompt):
        yield chunk


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

    chatbot = gr.Chatbot(height=250)

    user_input = gr.Textbox(
        placeholder="Ask a tax question (e.g. Can I deduct mileage for my car?)",
        label="Your Question"
    )

    submit_btn = gr.Button("Ask TaxAId")

    def respond(message, chat_history):
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": message})
        chat_history.append({
            "role": "assistant",
            "content": "🧠 TaxAId is reviewing IRS publications..."
        })

        yield chat_history, ""

        response = ""

        for chunk in generate_answer_stream(message):
            response += chunk
            chat_history[-1]["content"] = response
            yield chat_history, ""

        

    submit_btn.click(
        respond,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input],
        show_progress="minimal",
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

