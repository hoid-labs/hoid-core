"""
10 — RAG: Retrieval-Augmented Generation

Ingest documents into a vector store, then let the agent answer
questions by searching that knowledge base semantically. The knowledge
server handles chunking, embedding, and search — the agent decides
when to ingest and when to search.

New here: knowledge-server as subprocess, ingest + search tool pattern.

Requires: uv pip install -e ".[mcp,rag,qdrant]"
Also requires: EMBED_MODEL (or LLM_MODEL) pointing to an embeddings-capable endpoint.
"""

import asyncio
import json
from pathlib import Path

from hoid.core import Agent, LLMClient
from hoid.extensions import MCPClient, MCPManager
from hoid.observability import set_hook

KNOWLEDGE_SERVER = str(
    Path(__file__).parent.parent
    / "mcp_servers"
    / "knowledge_server.py"
)

# We'll ingest this repo's README as demo content — always present, real substance.
README = str(Path(__file__).parent.parent.parent / "README.md")


# Distinguish ingest from search visually — the two RAG operations look very different.
class TraceRag:
    async def emit(self, event):
        if event.event_type == "action":
            tool = event.payload.get("tool", "")
            args = event.payload.get("args", {})
            if "ingest" in tool:
                print(f"  [ingest] {str(args.get('file_path', args))[:100]}")
            elif "search" in tool:
                print(f"  [search] {str(args.get('query', args))[:100]}")
            else:
                print(f"  [call]   {tool} | {json.dumps(args, ensure_ascii=False)[:100]}")
        elif event.event_type == "observation":
            print(f"  [result] {str(event.payload.get('content', ''))[:160]}")
        elif event.event_type == "tool_error":
            print(f"  [error]  {event.payload.get('error', '')}")


async def main():
    set_hook(TraceRag())
    # Spawn the knowledge server as a subprocess.
    # Alternative (pre-running HTTP server): MCPClient.http("http://localhost:8082/mcp")
    async with (
        MCPManager([MCPClient.stdio("uv", ["run", "python", KNOWLEDGE_SERVER])]) as mcp,
        LLMClient.from_env() as client,
    ):
        tools = await mcp.get_all_tools()
        print(f"RAG tools: {[t.__name__ for t in tools]}\n")

        agent = Agent(
            client=client,
            tools=tools,
            max_tokens=2048,
            system_prompt=(
                "You answer questions from a knowledge base. "
                "Always search before answering. If the knowledge base is empty, ingest the relevant file first."
            ),
        )

        print("=== Step 1: ingest ===")
        await agent.run(f"Ingest the file at: {README}")

        print("\n=== Step 2: query ===")
        result = await agent.run(
            "What optional extras does the hoid provide and what does each one add?"
        )

        print("\n--- ANSWER ---")
        print(result["answer"])
        print(
            f"[tokens] prompt={result['prompt_tokens']} completion={result['completion_tokens']} billable={result['total_billable_tokens']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
