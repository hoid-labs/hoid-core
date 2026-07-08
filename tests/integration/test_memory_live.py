import json

import pytest

from hoid.core import Agent, LLMClient
from hoid.extensions import MCPClient, MCPManager
from hoid.observability import set_hook

pytestmark = pytest.mark.integration


class PrintHook:
    async def emit(self, event):
        print(
            f"[{event.event_type.upper()}] {json.dumps(getattr(event, 'payload', None), default=str)}"
        )


async def main():
    set_hook(PrintHook())

    servers = [
        MCPClient.stdio("uv", ["run", "memory-server"]),
    ]

    async with (
        MCPManager(servers) as mcp,
        LLMClient.from_env() as client,
    ):
        mcp_tools = await mcp.get_all_tools()

        agent = Agent(client=client, tools=mcp_tools, max_steps=5)

        result = await agent.run(
            "Save the fact that my favorite color is blue, then recall it and tell me what it is."
        )
        print("\n--- ANSWER ---")
        print(result["answer"])


async def test_main():
    await main()
