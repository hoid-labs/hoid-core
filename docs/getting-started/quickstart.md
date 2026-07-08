# Quickstart

This guide walks through the shortest path to a working tool-using agent.
It focuses on the core workflow: load configuration, create a client, attach tools, and run a task.

## Before you run

1. Install the framework (see Installation page).
2. Create a `.env` file with your model endpoint settings.
3. Ensure your endpoint is OpenAI-compatible.

Minimum required variables:

```env
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=...
```

## Your First Agent

The script below loads config from your environment, opens the async model client, and gives the agent two filesystem tools.
The agent can decide when to call these tools as part of its reasoning loop.

```python
import asyncio
from hoid.core import LLMClient, Agent
from examples.tools.filesystem import list_directory, read_file

async def main():
    async with LLMClient.from_env() as client:

        agent = Agent(client=client, tools=[list_directory, read_file])

        result = await agent.run("List the files in my home directory.")
        print(result["answer"])

if __name__ == "__main__":
    asyncio.run(main())
```

## What this does

1. `LLMClient.from_env()` reads `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` and opens the HTTP client.
2. `LLMClient` handles chat completion calls and connection lifecycle.
3. `Agent` runs a ReAct loop and can invoke the provided tools.

## Typical next steps

- Add approval gating for sensitive tools.
- Add guardrails for input/output safety.
- Move shared tools to MCP servers when multiple agents need them.