"""
04 — Structured Output

Skip the agent loop entirely when you need a guaranteed JSON shape.
Pass response_format with a json_schema to chat_completions() and
parse the result yourself. Useful for extraction, classification,
or any task where you own the output contract.

New here: response_format, json_schema, no agent loop.
"""

import asyncio
import json

from hoid.core import LLMClient

# Define the exact shape you expect back. The LLM must conform to it.
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "movie_review",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "year": {"type": "integer"},
                "rating": {"type": "number"},
                "summary": {"type": "string"},
                "pros": {"type": "array", "items": {"type": "string"}},
                "cons": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "year", "rating", "summary", "pros", "cons"],
            "additionalProperties": False,
        },
    },
}


async def main():
    async with LLMClient.from_env() as client:
        response = await client.chat_completions(
            messages=[
                {
                    "role": "system",
                    "content": "You are a film critic. Always respond with valid JSON.",
                },
                {"role": "user", "content": "Review the movie Inception (2010)."},
            ],
            response_format=SCHEMA,
        )

        raw = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        review = json.loads(raw)

        print("\n--- ANSWER ---")
        print(f"Title:   {review['title']} ({review['year']})")
        print(f"Rating:  {review['rating']}/10")
        print(f"Summary: {review['summary']}")
        print(f"Pros:    {', '.join(review['pros'])}")
        print(f"Cons:    {', '.join(review['cons'])}")
        print(
            f"[tokens] prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}"
        )


if __name__ == "__main__":
    asyncio.run(main())
