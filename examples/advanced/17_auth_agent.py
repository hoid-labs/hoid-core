"""
17 — Authentication & Authorization

Role-based access control (RBAC) with per-user ACL overrides.
Three users run the same prompt — each sees only the tools their role allows.

New here: AuthGate, MemoryPolicyBackend, StaticAuthProvider, AuthContext,
          auth_gate on Agent, auth_context on agent.run().
"""

import asyncio

from examples.tools import add_numbers, fetch_url, get_current_datetime, write_file
from hoid.core import Agent, LLMClient
from hoid.extensions.auth import (
    AuthContext,
    AuthGate,
    MemoryPolicyBackend,
    StaticAuthProvider,
)
from hoid.observability import set_hook

POLICY = {
    "roles": {
        "admin": {"tools": ["*"]},
        "analyst": {"tools": ["fetch_url", "get_current_datetime"]},
        "viewer": {"tools": ["get_current_datetime"]},
    },
    "users": {
        # alice inherits analyst + one extra ACL grant
        "alice": {"roles": ["analyst"], "extra_tools": ["add_numbers"]},
        "bob": {"roles": ["viewer"]},
        "svc": {"roles": ["admin"]},
    },
}

ALL_TOOLS = [get_current_datetime, fetch_url, add_numbers, write_file]

provider = StaticAuthProvider(
    users={
        "alice": AuthContext(user_id="alice", roles={"analyst"}),
        "bob": AuthContext(user_id="bob", roles={"viewer"}),
        "svc": AuthContext(user_id="svc", roles={"admin"}),
    }
)


class Trace:
    async def emit(self, event):
        if event.event_type == "action":
            print(f"    -> {event.payload.get('tool')}()")
        elif event.event_type == "tool_error":
            print(f"    !! {event.payload.get('error')}")
        elif event.event_type == "answer":
            print(f"    {str(event.payload.get('content', ''))[:200]}")


async def main():
    set_hook(Trace())
    gate = AuthGate(MemoryPolicyBackend(POLICY))

    async with LLMClient.from_env() as client:
        agent = Agent(client, tools=ALL_TOOLS, auth_gate=gate, temperature=0.0)

        for username in ["bob", "alice", "svc"]:
            ctx = await provider.resolve({"type": "username", "username": username})
            visible = [
                s["function"]["name"] for s in gate.filter_schemas(agent.schemas, ctx)
            ]
            print(f"\n[{username}] roles={ctx.roles} | tools={visible}")
            await agent.run("What time is it? Then add 10 and 32.", auth_context=ctx)


if __name__ == "__main__":
    asyncio.run(main())
