# Learn RAG: Chat with Your PDF 📚

A production-grade, beginner-friendly **Retrieval-Augmented Generation (RAG)** application built with Python, LangChain, Qdrant Cloud, Hugging Face embeddings, and OpenRouter.

---

## 🌟 Architecture & Workflow

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             THE RAG WORKFLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

[PHASE A: DOCUMENT INGESTION & CLOUD INDEXING]
 1. Upload PDF ──► 2. PyPDFLoader ──► 3. RecursiveTextSplitter ──► 4. Embeddings ──► 5. Qdrant Cloud
   (Streamlit)      (Extract Pages     (1,000 chars size,           (all-MiniLM-L6      (learning_rag
                     + 1-idx Page)      400 chars overlap)           384 dimensions)     Collection)

[PHASE B: RETRIEVAL & GROUNDED ANSWER GENERATION]
 6. User Query ──► 7. Vectorize Query ──► 8. Cosine Similarity ──► 9. OpenRouter LLM ──► 10. Answer +
   ("What is...")   (all-MiniLM-L6)          (Scoped to doc_id)     (recursive/flint)        Page Citations
```

---

## 🚀 Key Features

- **Page-Accurate Citations**: Tracks 1-indexed human-friendly page numbers directly from the PDF.
- **Strict Anti-Hallucination Grounding**: Configured with strict system prompts refusing to answer questions not present in the document.
- **Isolated Document Namespacing**: Uses deterministic SHA-256 `doc_id` hashing to guarantee queries never leak between documents.
- **Zero Local Docker Dependency**: Employs Qdrant Cloud managed vector database.
- **Interactive Streamlit Web UI**: Includes step-by-step pipeline status indicators, source citation badges, and chunk inspection diagnostics.

---

## 📂 Project Structure

```text
learn-rag/
├── app.py                      # Streamlit UI & interactive chat workflow
├── rag/                        # Core modular RAG package
│   ├── __init__.py             # Public package exports
│   ├── config.py               # Environment configuration and validation
│   ├── ingestion.py            # PDF loading, SHA-256 hashing, chunking, Qdrant upload
│   ├── retrieval.py            # Embedding cache, scoped similarity search, citations
│   └── generation.py           # Grounded prompt engineering and OpenRouter LLM call
├── tests/                      # Automated test suite
│   ├── __init__.py             # Test package initializer
│   ├── test_rag_pipeline.py    # Unit tests for chunking, citations, and grounding
│   └── validate_cloud_setup.py # End-to-end live cloud infrastructure test
├── requirements.txt            # Python package dependencies
├── .env.example                # Template for environment configuration
├── .gitignore                  # Git ignore rules for secrets and virtualenvs
├── README.md                   # Project summary and workshop guide
└── implementation.md           # Step-by-step implementation guide
```

---

## 🛠️ Setup & Installation

### 1. Create and Activate Virtual Environment

```powershell
# Using Python / uv
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```env
QDRANT_URL=https://your-cluster-id.us-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=learning_rag
OPENROUTER_API_KEY=sk-or-v1-your_openrouter_api_key
OPENROUTER_MODEL=recursive/flint-v1.0:free
```

---

## 🧪 Testing & Validation

Run the automated offline unit tests:
```powershell
pytest tests/ -v
```

Validate live cloud connectivity (once credentials are set in `.env`):
```powershell
python tests/validate_cloud_setup.py
```

---

## 💻 Running the Application

Launch the Streamlit interface:
```powershell
streamlit run app.py
```

The app will be available in your browser at `http://localhost:8501`.
