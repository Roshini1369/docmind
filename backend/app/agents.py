"""Multi-agent pipeline built with LangGraph.

    memory -> planner -> researcher -> (no chunks?) -> no_context -> END
                                    -> writer -> critic -> (not grounded & revisions left?) -> writer
                                                        -> END

Agents
  memory     compacts long chat history (context engineering)
  planner    breaks the question into focused search queries
  researcher runs RAG for every query, dedupes and packs results into a token budget
  writer     drafts a cited answer using only the retrieved context
  critic     checks the draft is grounded in the sources; can trigger one revision
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from . import config, context, llm, rag


class State(TypedDict, total=False):
    question: str
    history: list[dict]
    queries: list[str]
    chunks: list[dict]
    context: str
    draft: str
    critique: dict
    revisions: int
    answer: str
    trace: Annotated[list[str], operator.add]


# ---------------- nodes ----------------
def memory_node(state: State) -> dict:
    history = state.get("history") or []
    compacted = context.compact_history(history, llm.summarize)
    note = (
        f"Compacted {len(history)} messages into {len(compacted)}"
        if len(compacted) != len(history)
        else f"History kept as-is ({len(history)} messages)"
    )
    return {"history": compacted, "revisions": 0, "trace": [note]}


PLANNER_SYSTEM = (
    "You are a research planner. Break the user's question into 1-3 short, self-contained "
    "search queries for a document search engine. Resolve pronouns using the conversation. "
    'Reply with JSON only: {"queries": ["...", "..."]}'
)


def planner_node(state: State) -> dict:
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in state.get("history", [])[-4:])
    data = llm.complete_json(
        PLANNER_SYSTEM,
        f"Conversation so far:\n{convo}\n\nQuestion: {state['question']}",
        model=config.FAST_MODEL,
    )
    queries = [q for q in data.get("queries", []) if isinstance(q, str) and q.strip()][:3]
    if not queries:
        queries = [state["question"]]
    return {"queries": queries, "trace": [f"Planned {len(queries)} search quer{'y' if len(queries) == 1 else 'ies'}: " + " | ".join(queries)]}


def researcher_node(state: State) -> dict:
    found: dict[str, dict] = {}
    for q in state["queries"]:
        for ch in rag.search(q):
            if ch["id"] not in found or ch["score"] > found[ch["id"]]["score"]:
                found[ch["id"]] = ch
    raw = list(found.values())
    unique = context.dedupe(raw)
    packed, used = context.pack(unique)
    return {
        "chunks": packed,
        "context": context.format_context(packed),
        "trace": [
            f"Retrieved {len(raw)} chunks -> {len(unique)} after dedupe -> {len(packed)} kept (~{used} tokens)"
        ],
    }


WRITER_SYSTEM = (
    "You are DocMind, a careful research assistant. Answer ONLY from the numbered context. "
    "Cite sources inline like [1] or [2][3]. If the context does not contain the answer, say so "
    "plainly instead of guessing. Be concise and well structured."
)


def writer_node(state: State) -> dict:
    critique = state.get("critique") or {}
    revision_note = ""
    if critique and not critique.get("grounded", True):
        revision_note = (
            "\n\nYour previous draft had problems. Fix them:\n"
            f"{critique.get('issues', '')}\nPrevious draft:\n{state.get('draft', '')}"
        )
    user = f"Context:\n{state['context']}\n\nQuestion: {state['question']}{revision_note}"
    messages = [*state.get("history", []), {"role": "user", "content": user}]
    draft = llm.complete(WRITER_SYSTEM, messages)
    label = "Revised draft" if revision_note else "Drafted answer"
    return {"draft": draft, "trace": [f"{label} ({len(draft)} chars)"]}


CRITIC_SYSTEM = (
    "You are a strict fact-checker. Given numbered context and a draft answer, decide whether every "
    "claim in the draft is supported by the context and cited correctly. "
    'Reply with JSON only: {"grounded": true|false, "issues": "short description or empty"}'
)


def critic_node(state: State) -> dict:
    data = llm.complete_json(
        CRITIC_SYSTEM,
        f"Context:\n{state['context']}\n\nDraft:\n{state['draft']}",
    )
    grounded = bool(data.get("grounded", True))
    revisions = state.get("revisions", 0) + (0 if grounded else 1)
    msg = "Critic: answer is grounded" if grounded else f"Critic: found issues - {data.get('issues', '')}"
    return {
        "critique": {"grounded": grounded, "issues": data.get("issues", "")},
        "revisions": revisions,
        "answer": state["draft"],
        "trace": [msg],
    }


def no_context_node(state: State) -> dict:
    return {
        "answer": "I couldn't find anything relevant in your documents. Try uploading a document first or rephrasing the question.",
        "chunks": [],
        "trace": ["No relevant chunks found - skipped writing"],
    }


# ---------------- routing ----------------
def after_research(state: State) -> str:
    return "writer" if state.get("chunks") else "no_context"


def after_critic(state: State) -> str:
    critique = state.get("critique", {})
    if not critique.get("grounded", True) and state.get("revisions", 0) <= config.MAX_REVISIONS:
        return "writer"
    return END


def build_graph():
    g = StateGraph(State)
    g.add_node("memory", memory_node)
    g.add_node("planner", planner_node)
    g.add_node("researcher", researcher_node)
    g.add_node("writer", writer_node)
    g.add_node("critic", critic_node)
    g.add_node("no_context", no_context_node)

    g.set_entry_point("memory")
    g.add_edge("memory", "planner")
    g.add_edge("planner", "researcher")
    g.add_conditional_edges("researcher", after_research, {"writer": "writer", "no_context": "no_context"})
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", after_critic, {"writer": "writer", END: END})
    g.add_edge("no_context", END)
    return g.compile()


graph = build_graph()
