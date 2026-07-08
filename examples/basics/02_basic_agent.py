"""
02 — Basic ReAct Agent

Introduce the Agent class. No tools yet — the agent answers from
its own knowledge. Watch the global observability hook to see how the
framework emits structured events at each step of the ReAct loop.

New here: Agent, observability hook, event shapes.
"""

import asyncio
import json

from hoid.core import Agent, LLMClient
from hoid.observability import set_hook


# print_event prints every field of every event as raw JSON — useful here
# because the whole point is to see what structure each event carries.
class PrintHook:
    async def emit(self, event):
        fields = {k: v for k, v in (event.payload or {}).items() if "tokens" not in k}
        print(
            f"  [{event.event_type}] {json.dumps(fields, ensure_ascii=False)[:160]}"
        )


async def main():
    set_hook(PrintHook())
    async with LLMClient.from_env() as client:
        # system_prompt sets the agent's persistent persona for all run() calls.
        agent = Agent(
            client=client,
            system_prompt="You are a concise assistant. Keep answers to one sentence.",
        )

        result = await agent.run("What year did the Berlin Wall fall?")

        # result always contains: answer, messages, prompt_tokens, completion_tokens
        print("\n--- ANSWER ---")
        print(result["answer"])
        print(
            f"[tokens] prompt={result['prompt_tokens']} completion={result['completion_tokens']} billable={result['total_billable_tokens']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
