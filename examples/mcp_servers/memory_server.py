import argparse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn

from hoid.extensions.mcp import MCPContext, MCPServer
from hoid.extensions.memory import MemoryStore


@asynccontextmanager
async def lifespan(server: MCPServer) -> AsyncIterator[dict]:
    async with MemoryStore.from_env() as store:
        yield {"store": store}


mcp = MCPServer("memory", lifespan=lifespan)


# --- tools ---


@mcp.tool()
async def save_memory(key: str, value: str, ctx: MCPContext) -> str:
    """Store a value under a named key for later recall.

    Args:
        key: Name under which to store the value.
        value: The content to remember.
    """
    store: MemoryStore = ctx.lifespan["store"]
    await store.save(key, value)
    return f"Saved '{key}'."


@mcp.tool()
async def recall_memory(key: str, ctx: MCPContext) -> str:
    """Retrieve a previously stored value by key.

    Args:
        key: The key to look up.
    """
    store: MemoryStore = ctx.lifespan["store"]
    value = store.load(key)
    return value if value is not None else f"No memory found for '{key}'."


@mcp.tool()
async def list_memories(ctx: MCPContext) -> str:
    """List all stored memory keys."""
    store: MemoryStore = ctx.lifespan["store"]
    keys = store.list_keys()
    return "\n".join(keys) if keys else "No memories stored."


@mcp.tool()
async def delete_memory(key: str, ctx: MCPContext) -> str:
    """Delete a stored memory by key.

    Args:
        key: The key to delete.
    """
    store: MemoryStore = ctx.lifespan["store"]
    await store.clear(key)
    return f"Deleted '{key}'."


# --- entry point ---


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if args.http:
        uvicorn.run(mcp.http_app(), host=args.host, port=args.port, log_level="warning")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
