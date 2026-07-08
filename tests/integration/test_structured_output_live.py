import json

import pytest

from hoid.core import LLMClient

pytestmark = pytest.mark.integration


async def main():
    async with LLMClient.from_env() as client:
        response = await client.chat_completions(
            messages=[
                {
                    "role": "user",
                    "content": "Give me a fictional person with a name and age.",
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name", "age"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        content = response["choices"][0]["message"]["content"]
        data = json.loads(content)
        print(data)


async def test_main():
    await main()
