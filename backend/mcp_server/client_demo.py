"""Tiny MCP client that launches the DocMind server and calls its tools.

Usage (from backend/):  python -m mcp_server.client_demo "what is RAG?"
"""
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main(query: str) -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            docs = await session.call_tool("list_documents", {})
            print("\nDocuments:\n" + docs.content[0].text)

            result = await session.call_tool("search_documents", {"query": query, "top_k": 3})
            print(f"\nSearch results for {query!r}:\n" + result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "What is retrieval augmented generation?"))
