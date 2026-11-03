"""Offline tests: no API key or embedding model needed (LLM and vector search are stubbed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import agents, context, rag  # noqa: E402


def make_chunk(cid, text, score):
    return {"id": cid, "text": text, "source": "a.md", "page": 1, "score": score}


def test_chunking_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = rag.chunk_text(text, size=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    # overlap: the end of chunk i appears at the start of chunk i+1
    assert chunks[0].split(". ")[-1][:10] in chunks[1]


def test_chunk_empty():
    assert rag.chunk_text("   ") == []


def test_dedupe_removes_near_duplicates():
    a = make_chunk("1", "RAG retrieves chunks and passes them to the model", 0.9)
    b = make_chunk("2", "RAG retrieves chunks and passes them to the model.", 0.8)
    c = make_chunk("3", "Quantization reduces weight precision", 0.7)
    kept = context.dedupe([a, b, c])
    assert [k["id"] for k in kept] == ["1", "3"]


def test_pack_respects_budget():
    chunks = [make_chunk(str(i), "x" * 400, 1 - i / 10) for i in range(10)]  # ~100 tokens each
    packed, used = context.pack(chunks, budget=350)
    assert len(packed) == 3 and used <= 350
    assert packed[0]["score"] >= packed[-1]["score"]


def test_history_compaction():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(12)]
    out = context.compact_history(history, summarize=lambda t: "SUMMARY", keep_turns=2)
    assert len(out) == 2 + 4
    assert "SUMMARY" in out[0]["content"]
    assert out[-1]["content"] == "m11"


def test_graph_happy_path_and_revision(monkeypatch):
    chunks = [make_chunk("1", "RAG combines retrieval with generation.", 0.9)]
    monkeypatch.setattr(rag, "search", lambda q, **kw: chunks)

    critic_answers = iter([{"grounded": False, "issues": "claim 2 unsupported"}, {"grounded": True, "issues": ""}])

    def fake_json(system, user, **kw):
        if "planner" in system:
            return {"queries": ["what is rag"]}
        return next(critic_answers)

    monkeypatch.setattr(agents.llm, "complete_json", fake_json)
    monkeypatch.setattr(agents.llm, "complete", lambda system, messages, **kw: "RAG mixes retrieval and generation [1].")
    monkeypatch.setattr(agents.llm, "summarize", lambda t: "s")

    result = agents.graph.invoke({"question": "What is RAG?", "history": [], "trace": []})
    assert "[1]" in result["answer"]
    assert result["revisions"] == 1  # critic forced exactly one rewrite
    assert any("Critic: found issues" in t for t in result["trace"])


def test_graph_no_context(monkeypatch):
    monkeypatch.setattr(rag, "search", lambda q, **kw: [])
    monkeypatch.setattr(agents.llm, "complete_json", lambda *a, **kw: {"queries": ["q"]})
    result = agents.graph.invoke({"question": "anything", "history": [], "trace": []})
    assert "couldn't find" in result["answer"]
