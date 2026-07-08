"""
14 — Cached Tool Calls

@cached_tool wraps a @tool so its return value is cached for the lifetime
of the process, keyed on the arguments. Use it for deterministic,
side-effect-free tools that a multi-step agent might call more than once
with the same inputs — file reads, schema lookups, API metadata, etc.

@cached_tool works on both sync and async functions. The cache is
per-function and does not expire; restart the process to clear it.

New here: @cached_tool decorator.
"""

import asyncio
import json

from hoid.core import Agent, LLMClient, cached_tool
from hoid.observability import set_hook

_call_count = {"get_pi": 0, "get_greeting": 0}


@cached_tool
def get_pi(decimal_places: int = 5) -> str:
    """Return Pi to the requested number of decimal places.

    Args:
        decimal_places: How many decimal places to return (1-15).
    """
    _call_count["get_pi"] += 1
    import math

    return f"{math.pi:.{min(decimal_places, 15)}f}"


@cached_tool
async def get_greeting(language: str) -> str:
    """Return a greeting word in the requested language.

    Args:
        language: A language name, e.g. 'Spanish', 'French', 'Japanese'.
    """
    _call_count["get_greeting"] += 1
    greetings = {"spanish": "Hola", "french": "Bonjour", "japanese": "Konnichiwa"}
    return greetings.get(language.lower(), f"Hello ({language})")


class ShowTools:
    async def emit(self, event):
        if event.event_type == "action":
            tool = event.payload.get("tool")
            print(
                f"  -> {tool}({json.dumps(event.payload.get('args', {}))}) [actual call #{_call_count.get(tool, '?')}]"
            )
        elif event.event_type == "observation":
            print(f"  <- {str(event.payload.get('content', ''))[:80]}")
        elif event.event_type == "answer":
            print(f"  [answer] {str(event.payload.get('content', ''))[:120]}")


async def main():
    set_hook(ShowTools())
    async with LLMClient.from_env() as client:
        agent = Agent(client=client, tools=[get_pi, get_greeting])

        # Ask something that will likely cause the agent to call get_pi twice.
        print("=== Run 1 ===")
        await agent.run(
            "What is Pi to 5 decimal places? Also, what is Pi to 3 decimal places?"
        )
        print(f"\nget_pi actual call count: {_call_count['get_pi']}")
        # First call (5 places) hits the function; second call (3 places) also hits —
        # different arguments = different cache key. Same args in a run would be cached.

        # Run again — both calls now hit the cache entirely.
        print("\n=== Run 2 (same questions, cache warm) ===")
        _call_count["get_pi"] = 0
        await agent.run(
            "What is Pi to 5 decimal places? Also, what is Pi to 3 decimal places?"
        )
        print(f"\nget_pi actual call count (should be 0): {_call_count['get_pi']}")


if __name__ == "__main__":
    asyncio.run(main())
