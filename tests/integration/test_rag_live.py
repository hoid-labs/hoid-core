import json
from pathlib import Path

import pytest

from hoid.core import Agent, LLMClient
from hoid.extensions import MCPClient, MCPManager
from hoid.observability import set_hook

pytestmark = pytest.mark.integration

# The knowledge server is spawned as a subprocess — no manual server startup needed.
# Requires: uv pip install -e ".[rag,qdrant,mcp]"
# Requires: live embeddings endpoint at LLM_BASE_URL (set EMBED_MODEL if different from LLM_MODEL)

# Ingest the repo README as test data — always present, has real content
TEST_FILE = str(Path(__file__).parent.parent.parent / "README.md")


class PrintHook:
    async def emit(self, event):
        print(
            f"[{event.event_type.upper()}] {json.dumps(getattr(event, 'payload', None), default=str)}"
        )


async def main():
    set_hook(PrintHook())

    servers = [
        MCPClient.stdio("uv", ["run", "knowledge-server"]),
    ]

    async with (
        MCPManager(servers) as mcp,
        LLMClient.from_env() as client,
    ):
        mcp_tools = await mcp.get_all_tools()

        agent = Agent(client=client, tools=mcp_tools, max_steps=5)

        # ingest before querying — in-memory backend starts empty each run
        ingest_result = await agent.run(f"Ingest the file at path: {TEST_FILE}")
        print("\n--- INGEST ---")
        print(ingest_result["answer"])

        # now query what was ingested
        search_result = await agent.run(
            "Search the knowledge base and summarize what you find about the project architecture."
        )
        print("\n--- SEARCH ---")
        print(search_result["answer"])


async def test_main():
    await main()
