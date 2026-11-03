"""DocMind MCP server.

Exposes the document knowledge base as MCP tools so any MCP client
(Claude Desktop, Claude Code, your own agent) can search it.

Run (stdio transport) from the backend/ folder:
    python -m mcp_server.server

NOTE: stdio servers must never print to stdout - it carries the protocol.
"""
from mcp.server.fastmcp import FastMCP

from app import rag

mcp = FastMCP("docmind")


@mcp.tool()
def search_documents(query: str, top_k: int = 5, source: str | None = None) -> str:
    """Semantic search over the uploaded documents.

    Args:
        query: what to look for, in natural language
        top_k: number of chunks to return (1-10)
        source: optional file name to restrict the search to
    """
    top_k = max(1, min(top_k, 10))
    hits = rag.search(query, k=top_k, source=source)
    if not hits:
        return "No matching content found."
    return "\n\n".join(
        f"[{i}] {h['source']} (page {h['page']}, score {h['score']})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )


@mcp.tool()
def list_documents() -> str:
    """List all indexed documents and how many chunks each has."""
    docs = rag.list_documents()
    if not docs:
        return "No documents indexed yet."
    return "\n".join(f"- {d['name']} ({d['chunks']} chunks)" for d in docs)


@mcp.resource("docmind://documents")
def documents_resource() -> str:
    """Read-only resource: the current document index."""
    return list_documents()


if __name__ == "__main__":
    mcp.run()  # stdio transport
