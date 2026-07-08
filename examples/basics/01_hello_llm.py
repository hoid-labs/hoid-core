"""
01 — Hello LLM

The absolute minimum: load config from .env, open an LLM client,
and send a raw chat completion request.

No agents, no tools, no framework abstractions — just the HTTP call.
"""

import asyncio

from hoid.core import LLMClient


async def main():
    # LLMClient.from_env() reads LLM_BASE_URL, LLM_API_KEY, LLM_MODEL from .env
    # or construct manually for multiple LLMs in the same app:
    # other = LLMClient(base_url="https://api.example.com", api_key="mykey", model="mymodel")
    async with LLMClient.from_env() as client:
        # await suspends this coroutine until the HTTP response arrives,
        # freeing the event loop to handle other work in the meantime.
        response = await client.chat_completions(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ]
        )

        content = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})

        print("\n--- ANSWER ---")
        print(content)
        print(
            f"[tokens] prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}"
        )


# asyncio.run() creates the event loop and runs the async entry point.
# All subsequent examples follow this same pattern.
if __name__ == "__main__":
    asyncio.run(main())
