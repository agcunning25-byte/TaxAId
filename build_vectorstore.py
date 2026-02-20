import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOC_PATH = "./documents"
DB_PATH = "./chroma_tax_db"
COLLECTION_NAME = "irs_tax_publications"

# ===============
# Publication Metadata Map
# ===============

publication_metadata = {
    "pub463.pdf": {
        "topic": "travel_vehicle",
        "entity": "individual_and_business"
    },
    "pub946.pdf": {
        "topic": "depreciation",
        "entity": "business"
    },
    "pub334.pdf": {
        "topic": "small_business",
        "entity": "business"
    },
    "pub17.pdf": {
        "topic": "individual_income_tax",
        "entity": "individual"
    },
    "pub535.pdf": {
        "topic": "business_expenses",
        "entity": "business"
    }
}

print("Loading IRS documents...")

all_docs = []

for file in os.listdir(DOC_PATH):
    if file.endswith(".pdf"):
        print(f"Loading {file}")
        loader = PyPDFLoader(os.path.join(DOC_PATH, file))
        docs = loader.load()
        
        for doc in docs:
            doc.metadata["source"] = file
            # Attaching structured metadata
            extra_meta = publication_metadata.get(file, {})
            for key, value in extra_meta.items():
                doc.metadata[key] = value

        all_docs.extend(docs)

print(f"Total pages loaded: {len(all_docs)}")

print("Splitting documents into chunks....")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 600,
    chunk_overlap = 100
)

split_docs = splitter.split_documents(all_docs)

print(f"Total chunks created: {len(split_docs)}")

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

print("Building vector database...")

vectordb = Chroma.from_documents(
    documents = split_docs,
    embedding = embeddings,
    persist_directory = DB_PATH,
    collection_name = COLLECTION_NAME
)

print("Vector database built successfully!")
print("Stored documents:", vectordb._collection.count())