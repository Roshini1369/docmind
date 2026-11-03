"""FastAPI app: document management + streaming chat over the LangGraph pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, rag
from .agents import graph

app = FastAPI(title="DocMind", version="1.0.0")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list)


@app.get("/api/health")
def health():
    return {"status": "ok", "model": config.MODEL, "api_key_set": bool(config.ANTHROPIC_API_KEY)}


@app.get("/api/documents")
def documents():
    return rag.list_documents()


@app.post("/api/documents")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Allowed: {sorted(config.ALLOWED_EXTENSIONS)}")
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {config.MAX_UPLOAD_MB} MB")
    dest = config.UPLOAD_DIR / name
    dest.write_bytes(data)
    chunks = rag.ingest_file(dest, name)
    if chunks == 0:
        raise HTTPException(422, "No extractable text found (scanned PDF?)")
    return {"name": name, "chunks": chunks}


@app.delete("/api/documents/{name}")
def delete(name: str):
    name = Path(name).name
    rag.delete_document(name)
    (config.UPLOAD_DIR / name).unlink(missing_ok=True)
    return {"deleted": name}


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Server-Sent Events: one `step` event per agent, then a final `answer` event."""

    def stream():
        state = {"question": req.question, "history": req.history, "trace": []}
        final: dict = {}
        try:
            for update in graph.stream(state, stream_mode="updates"):
                for node, delta in update.items():
                    final.update(delta)
                    yield _sse("step", {"agent": node, "detail": (delta.get("trace") or [""])[-1]})
            sources = [
                {"n": i, "source": c["source"], "page": c["page"], "score": c["score"], "text": c["text"]}
                for i, c in enumerate(final.get("chunks", []), start=1)
            ]
            yield _sse("answer", {"answer": final.get("answer", ""), "sources": sources})
        except Exception as exc:  # surface errors to the UI instead of dropping the stream
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(stream(), media_type="text/event-stream")


# Serve the UI last so /api routes take priority
if config.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
