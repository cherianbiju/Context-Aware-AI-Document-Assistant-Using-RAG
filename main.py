
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage

from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from multi_doc_chat.exception.custom_exception import DocumentPortalException


# ----------------------------
# FastAPI initialization
# ----------------------------
app = FastAPI(title="MultiDocChat", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"

app.mount(
    "/static",
    StaticFiles(directory=str(static_dir)),
    name="static",
)

templates = Jinja2Templates(directory=str(templates_dir))


# ----------------------------
# In-memory chat history
# ----------------------------
SESSIONS: Dict[str, List[dict]] = {}


# ----------------------------
# FastAPI file adapter
# ----------------------------
class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile to the interface expected by ChatIngestor."""

    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()


# ----------------------------
# Request and response models
# ----------------------------
class UploadResponse(BaseModel):
    session_id: str
    indexed: bool
    message: str | None = None


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files uploaded.",
        )

    try:
        wrapped_files = [FastAPIFileAdapter(f) for f in files]

        ingestor = ChatIngestor(use_session_dirs=True)
        session_id = ingestor.session_id

        # Save documents, split them, create embeddings, and build FAISS.
        ingestor.built_retriver(
            uploaded_files=wrapped_files,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5,
        )

        # Initialize history only after ingestion succeeds.
        SESSIONS[session_id] = []

        print(
            f"Upload successful. Session: {session_id}",
            flush=True,
        )

        return UploadResponse(
            session_id=session_id,
            indexed=True,
            message="Indexing complete with MMR",
        )

    except DocumentPortalException as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {e}",
        ) from e

    finally:
        for f in files:
            await f.close()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id
    message = req.message.strip()

    if not session_id or session_id not in SESSIONS:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired session_id. Please upload documents again.",
        )

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    try:
        print("1. Chat request received", flush=True)

        # Create ONE RAG object.
        rag = ConversationalRAG(session_id=session_id)
        print("2. RAG initialized", flush=True)

        # Load the FAISS retriever into that same object.
        index_path = str(BASE_DIR / "faiss_index" / session_id)

        rag.load_retriever_from_faiss(
            index_path=index_path,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5,
        )
        print("3. FAISS retriever loaded", flush=True)

        # Convert existing conversation history into LangChain messages.
        simple = SESSIONS[session_id]

        lc_history = []

        for m in simple:
            role = m.get("role")
            content = m.get("content", "")

            if role == "user":
                lc_history.append(HumanMessage(content=content))

            elif role == "assistant":
                lc_history.append(AIMessage(content=content))

        print("4. Calling RAG chain; waiting for answer...", flush=True)

        # Invoke the SAME object whose retriever was loaded.
        answer = rag.invoke(
            message,
            chat_history=lc_history,
        )

        print("5. RAG response received", flush=True)

        # Save history only after a successful response.
        simple.append({
            "role": "user",
            "content": message,
        })

        simple.append({
            "role": "assistant",
            "content": answer,
        })

        return ChatResponse(answer=answer)

    except DocumentPortalException as e:
        print(f"RAG error: {e}", flush=True)

        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e

    except Exception as e:
        print(f"Chat error: {e}", flush=True)

        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {e}",
        ) from e


# ----------------------------
# Run application
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
