<<<<<<< HEAD
# DocMind: Multi-Agent RAG Research Assistant

Upload PDFs, text or markdown files and ask questions. A team of AI agents plans the search,
retrieves evidence, writes a cited answer and fact-checks itself. You watch every agent work live.

**Tech used:** Claude SDK · LangGraph · RAG (ChromaDB) · Multi-Agent Systems · Context Engineering · MCP · LLM · FastAPI · JavaScript

## Architecture

```
Browser (index.html)  --SSE-->  FastAPI  -->  LangGraph pipeline
                                                memory -> planner -> researcher -> writer -> critic
                                                                          |                    |
                                                                       ChromaDB          revise once if
                                                                    (vector search)      answer not grounded

Claude Desktop / any MCP client --stdio--> MCP server --> same ChromaDB
```

| Agent | Job | Model |
|---|---|---|
| memory | summarises old chat turns, keeps recent ones | fast model |
| planner | splits the question into 1-3 search queries | fast model |
| researcher | RAG search, near-duplicate removal, token-budget packing | none (code) |
| writer | answers only from numbered context with `[n]` citations | main model |
| critic | checks every claim is supported; triggers one revision | main model |

**Context engineering** (`backend/app/context.py`): dedupe overlapping chunks, pack best chunks into a token budget,
numbered citations, history compaction.

## Run it

Requires Python 3.10+ and an Anthropic API key (https://console.anthropic.com).

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
# edit .env and paste your ANTHROPIC_API_KEY

uvicorn app.main:app --reload
```

Open http://localhost:8000, upload `sample_docs/ai_agents_primer.md`, and ask:
- "What is context engineering?"
- "How is RAG different from fine-tuning, and when should I use each?"  (multi-query planning)

The first upload downloads a small embedding model (about 80 MB) once.

### Tests (no API key needed)
```bash
cd backend
pytest -q
```

### MCP server
```bash
cd backend
python -m mcp_server.client_demo "what is quantization?"   # quick demo client
```
Use it from Claude Desktop by adding this to `claude_desktop_config.json` (use absolute paths):
```json
{
  "mcpServers": {
    "docmind": {
      "command": "/absolute/path/to/docmind/backend/.venv/bin/python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/docmind/backend"
    }
  }
}
```

## Project layout
```
backend/app/config.py      settings
backend/app/rag.py         loading, chunking, indexing, search
backend/app/context.py     context engineering
backend/app/llm.py         Claude SDK wrapper
backend/app/agents.py      LangGraph multi-agent pipeline
backend/app/main.py        FastAPI + SSE streaming
backend/mcp_server/        MCP server + demo client
backend/tests/             offline tests
frontend/index.html        chat UI (no build step)
```

## Resume bullets (edit the numbers to match what you measure)
- Built **DocMind**, a full-stack multi-agent RAG assistant (FastAPI, JavaScript) where a **LangGraph** pipeline of planner, researcher, writer and critic agents answers questions over uploaded PDFs with inline citations, using the **Claude SDK**.
- Implemented **context engineering**: near-duplicate chunk removal, token-budget packing and conversation-history compaction to keep prompts small and grounded.
- Added a self-check loop where a critic agent verifies each claim against retrieved sources and triggers a revision, reducing unsupported answers.
- Exposed the knowledge base as an **MCP server** (tools and resources) so Claude Desktop and other MCP clients can query it.
- Streamed live agent traces to the UI over Server-Sent Events; covered the pipeline with offline pytest tests using stubbed LLM calls.

## Ideas to extend (good for interviews)
- Add a web-search agent with an MCP tool and route between docs and web.
- Evaluate retrieval with a small Q&A set (hit rate, MRR) and tune chunk size.
- Swap Chroma for pgvector, add auth and per-user collections, deploy with Docker.
=======
# docmind
A Multi-Agent RAG Research Assistant Using LangGraph
>>>>>>> ab83a86b6e96453cf67e483d5f0dc591a5749885
