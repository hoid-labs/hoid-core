"""
13 — Conversation History with HistoryBuffer

By default each agent.run() call is stateless — the agent has no memory
of previous calls. HistoryBuffer bridges that gap: it accumulates the
message history across calls and passes it back as prior_messages.

New here: HistoryBuffer, prior_messages in agent.run().
"""

import asyncio

from hoid.core import Agent, HistoryBuffer, LLMClient
from hoid.observability import set_hook


class ShowEvent:
    async def emit(self, event):
        if event.event_type == "answer":
            print(f"  [answer] {str(event.payload.get('content', ''))[:120]}")
        elif event.event_type == "task":
            print(f"  [task]   {str(event.payload.get('prompt', ''))[:120]}")


async def main():
    set_hook(ShowEvent())
    async with LLMClient.from_env() as client:
        agent = Agent(
            client=client,
            system_prompt="You are a concise assistant. Keep answers to two sentences.",
        )

        # max_tokens=2000 caps the history size by estimated token count.
        # max_messages=10 would cap by message count instead — pick one or both.
        buf = HistoryBuffer(max_tokens=2000)

        questions = [
            "My name is Alice. Please remember that.",
            "What is the capital of France?",
            # The agent should be able to answer this from history, not from guessing.
            "What is my name?",
        ]

        for question in questions:
            print(f"\nQ: {question}")
            # buf.get() returns the accumulated history; empty on the first call.
            result = await agent.run(question, prior_messages=buf.get())
            # extend() strips system messages — the agent re-injects its own each run.
            buf.extend(result["messages"])

        print(f"\nBuffer holds {len(buf.get())} messages after 3 turns.")
        print(
            f"Estimated tokens in buffer: "
            f"{sum(len(m.get('content') or '') // 4 for m in buf.get())}"
        )

        # clear() wipes the buffer — next run starts fresh.
        buf.clear()
        print(f"After clear: {len(buf.get())} messages.")


if __name__ == "__main__":
    asyncio.run(main())
