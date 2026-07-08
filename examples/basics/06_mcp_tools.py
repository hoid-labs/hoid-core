"""
06 — Remote Tools via MCP (multiple servers)

Connect to several out-of-process tool servers at once using MCPClient.
MCPManager aggregates all their tools into a single flat list — the agent
sees them the same way regardless of which server they come from.

New here: MCPClient.stdio, MCPManager, stacking multiple servers.
"""

import asyncio
import json
from pathlib import Path

from hoid.core import Agent, LLMClient
from hoid.extensions import MCPClient, MCPManager
from hoid.observability import set_hook

ROOT = Path(__file__).parent.parent.parent
EXAMPLES = Path(__file__).parent.parent
MEMORY_SERVER = str(EXAMPLES / "mcp_servers" / "memory_server.py")
KNOWLEDGE_SERVER = str(EXAMPLES / "mcp_servers" / "knowledge_server.py")


# Show which server each tool call came from by matching tool names.
_MEMORY_TOOLS = {"save_memory", "recall_memory", "list_memories", "delete_memory"}
_RAG_TOOLS = {"ingest_file", "search_notes"}


class TraceRemote:
    async def emit(self, event):
        if event.event_type == "action":
            tool = event.payload.get("tool", "")
            server = (
                "memory"
                if tool in _MEMORY_TOOLS
                else "rag" if tool in _RAG_TOOLS else "?"
            )
            print(
                f"  [{server}] {tool} <- {json.dumps(event.payload.get('args', {}), ensure_ascii=False)[:100]}"
            )
        elif event.event_type == "observation":
            print(f"  [result] {str(event.payload.get('content', ''))[:120]}")
        elif event.event_type == "tool_error":
            print(f"  [error]  {event.payload.get('error', '')}")


async def main():
    set_hook(TraceRemote())
    # Pass a list of bridges — MCPManager connects to all of them concurrently
    # and merges their tools. Each bridge is its own subprocess.
    # Alternative (pre-running HTTP server): MCPClient.http("http://localhost:8000/mcp", timeout=300.0)
    async with (
        MCPManager(
            [
                MCPClient.stdio("uv", ["run", "python", MEMORY_SERVER]),
                MCPClient.stdio("uv", ["run", "python", KNOWLEDGE_SERVER]),
                # MCPClient.http("http://localhost:8000/mcp"),
                # MCPClient.http("http://10.0.0.1:8000/mcp"),
            ]
        ) as mcp,
        LLMClient.from_env() as client,
    ):
        tools = await mcp.get_all_tools()
        print(f"Tools loaded ({len(tools)} total): {[t.__name__ for t in tools]}\n")

        agent = Agent(
            client=client,
            tools=tools,
            system_prompt=(
                "You are an assistant with memory and knowledge-base tools. "
                "You have no built-in knowledge of stored data — always use tools to look things up."
            ),
        )

        readme = str(ROOT / "README.md")
        await agent.run(f"Ingest the file at: {readme}")
        await agent.run(
            "Save under 'readme_summary' a one-sentence description of what this framework is."
        )

        result = await agent.run(
            "Search the knowledge base for optional extras, then recall what you saved under readme_summary."
        )

        print("\n--- ANSWER ---")
        print(result["answer"])
        print(
            f"[tokens] prompt={result['prompt_tokens']} completion={result['completion_tokens']} billable={result['total_billable_tokens']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
