import argparse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn

from hoid.extensions.mcp import MCPContext, MCPServer
from hoid.extensions.rag import RAGStore, backend_from_env

_MAX_CHARS = 8_000


@asynccontextmanager
async def lifespan(server: MCPServer) -> AsyncIterator[dict]:
    async with RAGStore.from_env(backend_from_env()) as rag:
        yield {"rag": rag}


mcp = MCPServer("knowledge", lifespan=lifespan)


# --- tools ---


@mcp.tool()
async def search_notes(query: str, ctx: MCPContext, limit: int = 3) -> str:
    """Search the knowledge base for relevant chunks.

    Args:
        query: Search query to find relevant knowledge chunks.
        limit: Maximum number of results to return (default 3).
    """
    rag: RAGStore = ctx.lifespan["rag"]
    results = await rag.search(query, limit=limit)
    if not results:
        return "No results found."
    text = "\n\n---\n\n".join(results)
    return text[:_MAX_CHARS] + ("\n...[truncated]" if len(text) > _MAX_CHARS else "")


@mcp.tool()
async def ingest_file(path: str, ctx: MCPContext) -> str:
    """Ingest a file into the knowledge base.

    Args:
        path: Absolute or relative path to the file to ingest.
    """
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(Path.home())
    except ValueError:
        raise PermissionError(
            f"Path '{resolved}' is outside the home directory sandbox"
        ) from None
    rag: RAGStore = ctx.lifespan["rag"]
    count = await rag.ingest_file(str(resolved))
    return f"Ingested {count} chunks from '{resolved}'."


# --- entry point ---


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if args.http:
        uvicorn.run(mcp.http_app(), host=args.host, port=args.port, log_level="warning")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
