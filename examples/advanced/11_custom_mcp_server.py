"""
11 — Custom MCP Server

Build and run your own MCP server from scratch.
This is the pattern for wrapping any external API, database,
or service so any agent can use it — without coupling the tools
to a specific agent or LLM.

What this example shows:
  - lifespan: create shared state once, available to all tools
  - @mcp.tool(): declare tools with rich docstrings and typed params
  - Context injection: access lifespan state inside a tool
  - Optional params with defaults
  - argparse main for stdio or HTTP transport

Run this server:
    python examples/11_custom_mcp_server.py --http --port 9000

Then connect from any agent:
    MCPClient.http("http://localhost:9000/mcp")

Or spawn it as a subprocess without --http:
    MCPClient.stdio("python", ["examples/11_custom_mcp_server.py"])
"""

import argparse
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from hoid.extensions.mcp import MCPContext, MCPServer

# --- Simulated in-memory "database" that lifespan manages ---


class NoteStore:
    def __init__(self):
        self._notes: dict[str, dict] = {}

    def add(self, title: str, body: str) -> str:
        note_id = f"note_{int(time.time() * 1000)}"
        self._notes[note_id] = {"id": note_id, "title": title, "body": body}
        return note_id

    def get(self, note_id: str) -> dict | None:
        return self._notes.get(note_id)

    def list_all(self) -> list[dict]:
        return list(self._notes.values())

    def delete(self, note_id: str) -> bool:
        return self._notes.pop(note_id, None) is not None


# --- Lifespan: create shared resources once per server process ---


@asynccontextmanager
async def lifespan(server: MCPServer) -> AsyncIterator[dict]:
    # Create the store once; all tool calls share this instance
    store = NoteStore()
    yield {"store": store}
    # Cleanup happens here after the server shuts down


mcp = MCPServer("notes-server", lifespan=lifespan)


# --- Tools: each gets ctx injected by MCPServer, stripped from the schema ---


@mcp.tool()
async def create_note(title: str, body: str, ctx: MCPContext) -> str:
    """Create a new note and return its ID.

    Args:
        title: Short title for the note.
        body: Full text content of the note.
    """
    store: NoteStore = ctx.lifespan["store"]
    note_id = store.add(title, body)
    return f"Note created with id '{note_id}'."


@mcp.tool()
async def get_note(note_id: str, ctx: MCPContext) -> str:
    """Retrieve a note by its ID.

    Args:
        note_id: The ID returned when the note was created.
    """
    store: NoteStore = ctx.lifespan["store"]
    note = store.get(note_id)
    if not note:
        return f"No note found with id '{note_id}'."
    return json.dumps(note)


@mcp.tool()
async def list_notes(ctx: MCPContext) -> str:
    "List all stored notes with their IDs and titles."
    store: NoteStore = ctx.lifespan["store"]
    notes = store.list_all()
    if not notes:
        return "No notes stored yet."
    return json.dumps([{"id": n["id"], "title": n["title"]} for n in notes])


@mcp.tool()
async def delete_note(note_id: str, ctx: MCPContext) -> str:
    """Delete a note by its ID.

    Args:
        note_id: The ID of the note to delete.
    """
    store: NoteStore = ctx.lifespan["store"]
    deleted = store.delete(note_id)
    return f"Deleted '{note_id}'." if deleted else f"No note found with id '{note_id}'."


# --- Entry point ---


def main():
    parser = argparse.ArgumentParser(description="Notes MCP server")
    parser.add_argument("--http", action="store_true", help="Run as HTTP server")
    parser.add_argument(
        "--port", type=int, default=9000, help="HTTP port (default 9000)"
    )
    args = parser.parse_args()

    if args.http:
        import uvicorn

        uvicorn.run(mcp.http_app(), host="0.0.0.0", port=args.port, log_level="warning")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
