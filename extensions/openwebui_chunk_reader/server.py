import argparse
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from extensions.openwebui_chunk_reader.ui import ui_html

DEFAULT_PROMPT = (
    "Summarize this document chunk in 3 concise bullet points. "
    "Preserve key names, claims, and chronology when present."
)
DEFAULT_OPEN_WEBUI_URL = "http://localhost:3000"


@dataclass
class ChunkedDocument:
    document_id: str
    name: str
    source: str
    content: str
    chunk_size: int
    chunks: list[str] = field(default_factory=list)


class SummaryRequest(BaseModel):
    chunk_index: int = Field(..., ge=0)
    base_url: str = Field(DEFAULT_OPEN_WEBUI_URL, min_length=1)
    model: str = Field(..., min_length=1)
    api_key: str | None = None
    system_prompt: str = Field(DEFAULT_PROMPT, min_length=1)


app = FastAPI(title="Open WebUI Chunk Reader")
DOCUMENTS: dict[str, ChunkedDocument] = {}


def chunk_text(content: str, chunk_size: int) -> list[str]:
    text = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError("document is empty")

    chunks: list[str] = []
    start = 0
    total = len(text)
    while start < total:
        end = min(total, start + chunk_size)
        if end < total:
            boundary = max(
                text.rfind("\n\n", start, end),
                text.rfind(". ", start, end),
                text.rfind("! ", start, end),
                text.rfind("? ", start, end),
                text.rfind(" ", start, end),
            )
            if boundary > start + max(40, chunk_size // 3):
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def to_chat_completions_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if cleaned.endswith("/api/chat/completions"):
        return cleaned
    if cleaned.endswith("/api"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/api/chat/completions"


def summarize_chunk(
    *,
    chunk: str,
    base_url: str,
    model: str,
    api_key: str | None,
    system_prompt: str,
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.post(
            to_chat_completions_url(base_url),
            json=payload,
            headers=headers,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"failed to reach Open WebUI endpoint: {exc}",
        ) from exc

    if not response.ok:
        detail = response.text.strip() or response.reason
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Open WebUI request failed: {detail}",
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Open WebUI response did not include a chat completion message",
        ) from exc


def store_document(*, name: str, source: str, content: str, chunk_size: int) -> ChunkedDocument:
    try:
        chunks = chunk_text(content, chunk_size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document_id = str(uuid.uuid4())
    document = ChunkedDocument(
        document_id=document_id,
        name=name,
        source=source,
        content=content,
        chunk_size=chunk_size,
        chunks=chunks,
    )
    DOCUMENTS[document_id] = document
    return document


def get_document(document_id: str) -> ChunkedDocument:
    document = DOCUMENTS.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return document


@app.get("/health")
def health():
    return {"status": "ok", "documents_loaded": len(DOCUMENTS)}


@app.get("/")
def root():
    return HTMLResponse(ui_html())


@app.get("/ui")
def ui():
    return HTMLResponse(ui_html())


@app.get("/api/config")
def config():
    return {
        "default_open_webui_url": DEFAULT_OPEN_WEBUI_URL,
        "default_prompt": DEFAULT_PROMPT,
    }


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(..., ge=100, le=10000),
):
    name = file.filename or "document.txt"
    if Path(name).suffix.lower() not in {".txt", ".md", ".text"}:
        raise HTTPException(status_code=400, detail="only .txt, .md, and .text files are supported")
    content = (await file.read()).decode("utf-8", errors="ignore")
    document = store_document(
        name=name,
        source="upload",
        content=content,
        chunk_size=chunk_size,
    )
    return {
        "document_id": document.document_id,
        "name": document.name,
        "source": document.source,
        "chunk_size": document.chunk_size,
        "chunk_count": len(document.chunks),
    }


@app.post("/api/documents/path")
def load_document_from_path(
    path: str = Form(...),
    chunk_size: int = Form(..., ge=100, le=10000),
):
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    if file_path.suffix.lower() not in {".txt", ".md", ".text"}:
        raise HTTPException(status_code=400, detail="only .txt, .md, and .text files are supported")
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = file_path.read_text(encoding="utf-8", errors="ignore")

    document = store_document(
        name=file_path.name,
        source=str(file_path),
        content=content,
        chunk_size=chunk_size,
    )
    return {
        "document_id": document.document_id,
        "name": document.name,
        "source": document.source,
        "chunk_size": document.chunk_size,
        "chunk_count": len(document.chunks),
    }


@app.get("/api/documents/{document_id}")
def document_details(document_id: str):
    document = get_document(document_id)
    return {
        "document_id": document.document_id,
        "name": document.name,
        "source": document.source,
        "chunk_size": document.chunk_size,
        "chunk_count": len(document.chunks),
    }


@app.get("/api/documents/{document_id}/chunks/{chunk_index}")
def document_chunk(document_id: str, chunk_index: int):
    document = get_document(document_id)
    if chunk_index < 0 or chunk_index >= len(document.chunks):
        raise HTTPException(status_code=404, detail="chunk not found")
    return {
        "document_id": document.document_id,
        "chunk_index": chunk_index,
        "chunk_count": len(document.chunks),
        "chunk": document.chunks[chunk_index],
    }


@app.post("/api/documents/{document_id}/summaries")
def document_summary(document_id: str, req: SummaryRequest):
    document = get_document(document_id)
    if req.chunk_index >= len(document.chunks):
        raise HTTPException(status_code=404, detail="chunk not found")
    summary = summarize_chunk(
        chunk=document.chunks[req.chunk_index],
        base_url=req.base_url,
        model=req.model,
        api_key=req.api_key,
        system_prompt=req.system_prompt,
    )
    return {
        "document_id": document.document_id,
        "chunk_index": req.chunk_index,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
