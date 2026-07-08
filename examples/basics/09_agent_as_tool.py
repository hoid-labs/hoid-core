"""
09 — Agent as a Tool (Remote Agent Delegation)

A running agent server exposes one tool — delegate_to_{name}(task) — over MCP.
Your local agent calls it like any other tool, with no knowledge of
the remote agent's tools, LLM, or system prompt.

New here: MCPClient spawning an agent server, timeout for long-running ReAct loops.
"""

import asyncio
from pathlib import Path

from hoid.core import Agent, LLMClient
from hoid.extensions import MCPClient, MCPManager
from hoid.observability import set_hook

# Resolve the agent server path relative to this file.
RESEARCHER = str(
    Path(__file__).parent.parent.parent
    / "examples"
    / "specialized_agents"
    / "researcher_agent.py"
)


# The interesting moment here is the delegation: task sent out, answer received.
class LogDelegation:
    async def emit(self, event):
        if event.event_type == "action":
            task = event.payload.get("args", {}).get("task", "")
            print(f"  [delegating to {event.payload.get('tool')}] {str(task)[:120]}")
        elif event.event_type == "observation":
            print(f"  [remote answer] {str(event.payload.get('content', ''))[:200]}")
        elif event.event_type == "tool_error":
            print(f"  [delegation failed] {event.payload.get('error', '')}")


async def main():
    set_hook(LogDelegation())
    # Spawn the researcher agent as a subprocess — it runs its own ReAct loop internally.
    # Alternative (pre-running HTTP server): MCPClient.http("http://localhost:8090/mcp", timeout=300.0)
    # The timeout=300.0 on HTTP gives the remote agent up to 5 minutes to complete.
    async with (
        MCPManager([MCPClient.stdio("uv", ["run", "python", RESEARCHER])]) as mcp,
        LLMClient.from_env() as client,
    ):
        # The remote agent's tool appears here like any local tool.
        remote_tools = await mcp.get_all_tools()
        print(f"Remote tools available: {[t.name for t in remote_tools]}\n")

        agent = Agent(client=client, tools=remote_tools, max_tokens=2048)

        result = await agent.run(
            "Research what Python asyncio is and give me a two-sentence explanation I can share with a junior developer."
        )

        print("\n--- ANSWER ---")
        print(result["answer"])
        print(
            f"[tokens] prompt={result['prompt_tokens']} completion={result['completion_tokens']} billable={result['total_billable_tokens']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
