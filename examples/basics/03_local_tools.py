"""
03 — Agent with Local Tools

Add tools to the agent. The @tool decorator turns any typed function
into an OpenAI-compatible JSON schema automatically. The agent decides
when and which tools to call — you just provide them.

New here: @tool decorator, tools=[...] on Agent.
"""

import asyncio
import json

from hoid.core import Agent, LLMClient, tool
from hoid.observability import set_hook


# @tool reads the type hints and docstring to generate the JSON schema
# that the LLM uses to understand what each tool does and what it expects.
@tool
def celsius_to_fahrenheit(celsius: float) -> float:
    "Convert a temperature from Celsius to Fahrenheit."
    return celsius * 9 / 5 + 32


@tool
def word_count(text: str) -> int:
    "Count the number of words in a text string."
    return len(text.split())


# Arrow notation makes the call/result pair immediately obvious.
class ToolIOHook:
    async def emit(self, event):
        if event.event_type == "action":
            print(
                f"  -> {event.payload.get('tool')}({json.dumps(event.payload.get('args', {}), ensure_ascii=False)[:80]})"
            )
        elif event.event_type == "observation":
            print(f"  <- {str(event.payload.get('content', ''))[:80]}")


async def main():
    set_hook(ToolIOHook())
    async with LLMClient.from_env() as client:
        agent = Agent(
            client=client,
            tools=[celsius_to_fahrenheit, word_count],
        )

        result = await agent.run(
            "What is 22°C in Fahrenheit? Also, how many words are in this sentence: "
            "'The quick brown fox jumps over the lazy dog'?"
        )

        print("\n--- ANSWER ---")
        print(result["answer"])
        print(
            f"[tokens] prompt={result['prompt_tokens']} completion={result['completion_tokens']} billable={result['total_billable_tokens']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
