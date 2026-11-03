"""RAG layer: loading, chunking, embedding (via Chroma) and retrieval."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from . import config

_collection = None


def get_collection():
    """Lazily create the persistent Chroma collection (cosine similarity)."""
    global _collection
    if _collection is None:
        import chromadb

        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = client.get_or_create_collection(
            "docmind", metadata={"hnsw:space": "cosine"}
        )
    return _collection


# ---------- loading ----------
def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for a PDF / text / markdown file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(i + 1, (p.extract_text() or "")) for i, p in enumerate(reader.pages)]
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


# ---------- chunking ----------
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Sentence-aware chunking with character overlap between neighbouring chunks."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sent in sentences:
        # hard-split pathological sentences longer than a chunk
        while len(sent) > size:
            if current:
                chunks.append(" ".join(current))
                current, length = [], 0
            chunks.append(sent[:size])
            sent = sent[size - overlap:]
        if length + len(sent) + 1 > size and current:
            chunks.append(" ".join(current))
            # carry over trailing sentences as overlap
            carried: list[str] = []
            carried_len = 0
            for prev in reversed(current):
                if carried_len + len(prev) > overlap:
                    break
                carried.insert(0, prev)
                carried_len += len(prev) + 1
            current, length = carried, carried_len
        current.append(sent)
        length += len(sent) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


# ---------- ingestion ----------
def ingest_file(path: Path, source_name: str | None = None) -> int:
    """Chunk and index a file. Re-ingesting the same source replaces old chunks."""
    source = source_name or path.name
    pages = extract_pages(path)
    digest = hashlib.sha1(source.encode()).hexdigest()[:10]

    ids, docs, metas = [], [], []
    for page_no, page_text in pages:
        for i, chunk in enumerate(chunk_text(page_text)):
            ids.append(f"{digest}-p{page_no}-c{i}")
            docs.append(chunk)
            metas.append({"source": source, "page": page_no})

    col = get_collection()
    delete_document(source)
    if ids:
        col.add(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def delete_document(source: str) -> None:
    col = get_collection()
    if col.count():
        col.delete(where={"source": source})


# ---------- retrieval ----------
def search(query: str, k: int = config.TOP_K, source: str | None = None) -> list[dict]:
    col = get_collection()
    if col.count() == 0:
        return []
    res = col.query(
        query_texts=[query],
        n_results=min(k, col.count()),
        where={"source": source} if source else None,
    )
    out = []
    for cid, text, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append(
            {
                "id": cid,
                "text": text,
                "source": meta["source"],
                "page": meta["page"],
                "score": round(1 - dist, 4),  # cosine similarity
            }
        )
    return out


def list_documents() -> list[dict]:
    col = get_collection()
    if col.count() == 0:
        return []
    metas = col.get(include=["metadatas"])["metadatas"]
    counts: dict[str, int] = {}
    for m in metas:
        counts[m["source"]] = counts.get(m["source"], 0) + 1
    return [{"name": n, "chunks": c} for n, c in sorted(counts.items())]
