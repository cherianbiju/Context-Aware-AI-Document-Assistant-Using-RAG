# Context-Aware AI Document Assistant Using RAG

An end-to-end, production-ready Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangChain**, **Google Gemini / Groq**, **FAISS Vector Store**, and **LangSmith** for LLM evaluation.

---

## 📌 Features

- **Multi-Document Ingestion:** Supports `.pdf`, `.docx`, and `.txt` files with recursive character splitting and metadata extraction.
- **Session-Isolated Vector Stores:** Dynamic session-based FAISS vector indexing preventing cross-session contamination.
- **Maximal Marginal Relevance (MMR):** Reduces redundancy and optimizes information diversity during document retrieval.
- **Context-Aware Conversational Chain:** Re-phrases follow-up queries using prior chat history before context lookup to ensure accurate multi-turn dialog.
- **Pluggable LLM Backends:** Switch seamlessly between Google Gemini (`gemini-3.8-flash`) and Groq (`openai/gpt-oss-20b`).
- **LLMOps & Evaluation:** Automated LLM-as-a-Judge evaluations and LangSmith observability pipeline to measure retrieval correctness.
- **Clean Web Interface:** Modern, responsive chat UI included (`Jinja2`, `HTML5`, `CSS3`).

---

## 🏗️ Architecture

```
User Document (.pdf, .docx, .txt)
        │
        ▼
Document Splitter (RecursiveCharacterTextSplitter)
        │
        ▼
Embedding Generator (Google Gemini Embeddings)
        │
        ▼
FAISS Vector Store (Sessionized Index)
        │
        ├── User Query + Chat History
        ▼
Contextualize Query Chain (Query Re-writing)
        │
        ▼
MMR Retriever (k=5, fetch_k=20, lambda=0.5)
        │
        ▼
Context QA Chain (Gemini / Groq LLM)
        │
        ▼
Synthesized Answer
```

---

## 📁 Project Structure

```
.
├── multi_doc_chat/               # Core application package
│   ├── config/                   # Model and retriever YAML configs
│   ├── exception/                # Custom exception handling
│   ├── logger/                   # Structured logging (structlog)
│   ├── model/                    # Pydantic schemas and enums
│   ├── prompts/                  # Prompt registry for contextualization & QA
│   ├── src/
│   │   ├── document_chat/        # Conversational RAG chain & FAISS retrieval
│   │   └── document_ingestion/   # Document processing & FAISS index management
│   └── utils/                    # Config, file I/O, and model loaders
├── templates/                    # Web UI (Jinja2 HTML)
├── static/                       # Web UI styling (CSS)
├── guide/                        # Architecture & local guide documentation
├── notebook/                     # Jupyter notebooks for experiments & evaluation
├── tests/                        # Unit and integration test suite
├── .env.example                  # Environment template
├── Dockerfile                    # Containerization definition
├── main.py                       # FastAPI entrypoint
├── pyproject.toml                # Project metadata and dependencies
├── requirements.txt              # Production dependencies
└── run_evaluations.py            # LangSmith evaluation pipeline
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.12+
- Google Gemini API Key and/or Groq API Key

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/Context-Aware-AI-Document-Assistant-Using-RAG.git
cd Context-Aware-AI-Document-Assistant-Using-RAG
```

### 3. Environment Setup
Copy the example environment file and set your API keys:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GOOGLE_API_KEY=your_actual_google_api_key
GROQ_API_KEY=your_actual_groq_api_key
LLM_PROVIDER=google
```

### 4. Install Dependencies
Using `pip`:
```bash
pip install -r requirements.txt
```
Or using [`uv`](https://astral.sh/uv):
```bash
uv sync
```

### 5. Run the Application
```bash
uvicorn main:app --reload --port 8000
```
Open your browser at `http://localhost:8000/`.

---

## 🐳 Running with Docker

Build and run using Docker:
```bash
# Build image
docker build -t multi-doc-chat .

# Run container
docker run -p 8080:8080 --env-file .env multi-doc-chat
```
Access the application at `http://localhost:8080/`.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web chat interface |
| `GET` | `/health` | Health check endpoint |
| `POST` | `/upload` | Multipart file upload; creates session and builds FAISS index |
| `POST` | `/chat` | Takes `{ session_id, message }` and returns `{ answer }` |

---

## 🧪 Testing & Evaluation

### Run Unit & Integration Tests
```bash
pytest -v
```

### Run LangSmith Evaluations
Set your `LANGSMITH_API_KEY` in `.env`, then run:
```bash
# Default evaluation
python run_evaluations.py

# Run all evaluators
python run_evaluations.py --evaluator all

# Custom parameters
python run_evaluations.py --evaluator correctness --chunk-size 500 --k 5
```

Interactive exploration is also available in `notebook/Evaluations.ipynb`.

---

## 📚 Technical Documentation

- **MMR Implementation Details:** [`guide/MMR_IMPLEMENTATION.md`](guide/MMR_IMPLEMENTATION.md)
- **Local Jenkins Guide:** [`guide/JENKINS_LOCAL_GUIDE.md`](guide/JENKINS_LOCAL_GUIDE.md)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
