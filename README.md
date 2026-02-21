# 🧠 TaxAId

**TaxAId** is a metadata-aware Retrieval-Augmented Generation (RAG) system that answers U.S. tax questions using official IRS publications.

It retrieves relevant sections from IRS PDFs, routes queries using structured metadata filters, and generates controlled, citation-anchored responses using a locally hosted open-weight LLM.

---

## 🚀 Overview

TaxAId was built to demonstrate production-grade RAG architecture with:

- Metadata-aware retrieval routing
- Controlled citation formatting
- Output sanitization to prevent hallucinated structure
- Deterministic model configuration
- Lightweight local inference using GGUF models
- Clean Gradio-based UI

The system answers tax questions strictly using retrieved IRS documentation and appends a single controlled citation line.

---

## 🏗 Architecture

### 1️⃣ Document Processing

- IRS publications are loaded via `PyPDFLoader`
- Documents are chunked using `RecursiveCharacterTextSplitter`
- Structured metadata is attached per publication:
  - `topic`
  - `entity`
  - `source`
- Embeddings generated via `sentence-transformers/all-MiniLM-L6-v2`
- Stored in a persistent Chroma vector database

---

### 2️⃣ Metadata-Aware Retrieval Routing

Instead of relying purely on semantic similarity, TaxAId:

- Applies rule-based routing using regex boundary matching
- Routes queries to filtered Chroma subsets via metadata
- Falls back to full semantic search if no match is found

Example routing filters:

- `travel_vehicle`
- `medical_expenses`
- `dependent_care_credit`
- `health_accounts`
- `irs_collections`
- `income_rules`
- `business_expenses`
- `filing_dependents`

This reduces irrelevant retrieval and improves answer grounding.

---

### 3️⃣ Controlled LLM Generation

- Model: **Phi-3 Mini (GGUF)**
- Inference via `llama-cpp-python`
- Deterministic configuration (`temperature=0.1`)
- Token limit constrained
- Prompt explicitly enforces:
  - One concise answer
  - No repetition
  - Single citation line
  - No hallucinated citations

Output cleaning layer removes:
- Prompt leakage
- Duplicate citations
- Model artifacts (e.g. `<|AI_response|>`)

---

### 4️⃣ UI Layer

Built with Gradio:

- Chat-style interface
- “Thinking” placeholder state
- Controlled markdown rendering
- Clean UX without duplicate outputs

---

## 📦 Tech Stack

- Python
- LangChain (modular components)
- ChromaDB
- sentence-transformers
- llama-cpp-python
- GGUF quantized LLM
- Gradio

---

## 🖥 Running Locally

### 1️⃣ Clone Repo

```bash
git clone https://github.com/agcunning25-byte/TaxAId.git
cd TaxAId
```
---
### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```
---
### 3️⃣ Install Requirements
```bash
pip install -r requirements.txt
```
---
### 4️⃣ Download Model
```bash
python download_model.py
```
Or manually download:
- `Phi-3-mini-4k-instruct-q4.gguf`
---
### 5️⃣ Add IRS Publications
Place required IRS PDFs inside:
```bash
documents/
```
Then build the vector database:
```bash
python build_vectorstore.py
```
---
### 6️⃣ Run the Application
```bash
python app.py
```
---
## 🎯 Design Goals
This project demonstrates:
- Practical RAG engineering
- Retrieval routing beyond naive semantic search
- Output guardrails and sanitization
- Lightweight local LLM inference
- Production-aware repository hygiene
---
## 📌 Future Enhancements
- Token-level streaming
- Confidence scoring
- Retrieval preview panel
- Query rewriting layer
- HuggingFace Spaces deployment
- Retrieval evaluation metrics
---
## ⚠️ Disclaimer
TaxAId provides information guidance based strictly on retrieved IRS publications.
It is not a substitute for professional tax advice.
---
## 👤 Author
Built by Adam Cunningham | 
Machine Learning Engineer in progress