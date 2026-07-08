import argparse
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from examples.tools.web_fetch import fetch_url
from hoid.core import Agent, LLMClient
from hoid.extensions.mcp import MCPContext, MCPServer

_MAX_CHARS = 8_000

_log = logging.getLogger("researcher")
_log.setLevel(logging.DEBUG)
# trace ReAct events to file
_log.addHandler(logging.FileHandler("researcher.log"))


@asynccontextmanager
async def lifespan(server: MCPServer) -> AsyncIterator[dict]:
    async with LLMClient.from_env() as client:
        yield {"client": client}


mcp = MCPServer("researcher-agent", lifespan=lifespan)


@mcp.tool()
async def delegate_to_researcher(task: str, ctx: MCPContext) -> str:
    """Research agent. Fetches URLs and synthesizes findings into a concise report.
    Delegate tasks that require gathering information from the web.

    Args:
        task: Natural language instruction describing what to research.
    """
    client: LLMClient = ctx.lifespan["client"]

    class JsonLogHook:
        async def emit(self, event):
            _log.debug(
                json.dumps(
                    {
                        "type": event.event_type,
                        "payload": event.payload,
                        "tokens": {
                            "prompt": event.tokens.prompt_tokens,
                            "completion": event.tokens.completion_tokens,
                        },
                    },
                    default=str,
                )
            )

    agent = Agent(
        client=client,
        tools=[fetch_url],
        max_steps=10,
        max_tokens=4096,
        system_prompt=(
            "You are a research assistant. Fetch relevant URLs and synthesize the findings "
            "into a concise intelligence report. Always ground your answer in what you actually retrieved."
        ),
        on_step=lambda e: _log.debug(json.dumps(e, default=str)),
    )
    from hoid.observability import set_hook

    set_hook(JsonLogHook())
    result = await agent.run(task)
    answer = result.get("answer", "(no answer)")
    return answer[:_MAX_CHARS] + (
        "\n...[truncated]" if len(answer) > _MAX_CHARS else ""
    )


# --- entry point ---


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    if args.http:
        import uvicorn

        uvicorn.run(mcp.http_app(), host="0.0.0.0", port=args.port, log_level="warning")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
