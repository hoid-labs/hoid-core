"""
08 — Multi-Agent Orchestration

The Orchestrator creates a supervisor agent with a delegate() tool
injected automatically. The supervisor decides which sub-agent to
call and what task to hand off — you just define the sub-agents
and give the supervisor a high-level goal.

New here: Orchestrator, sub_agents dict, make_step_printer factory for labeled output.
"""

import asyncio
import json
from pathlib import Path

from hoid.core import Agent, LLMClient, Orchestrator, tool
from hoid.observability import set_hook


# --- Sub-agent A: knows about the filesystem ---
@tool
def list_directory(path: str) -> str:
    "List the files and folders at the given path. Returns one entry per line."
    p = Path(path).expanduser()
    if not p.exists():
        return f"Path does not exist: {path}"
    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
    return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries)


@tool
def count_files(path: str) -> int:
    "Count the number of files (not directories) directly inside a directory."
    p = Path(path).expanduser()
    if not p.is_dir():
        return 0
    return sum(1 for e in p.iterdir() if e.is_file())


# --- Sub-agent B: knows about math ---
@tool
def calculate(expression: str) -> str:
    "Evaluate a safe arithmetic expression string and return the result."
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: unsafe expression."
    return str(eval(expression))  # noqa: S307 — constrained to arithmetic chars above


# Label each event with the agent that emitted it. Sub-agent events carry a
# delegated_to field in the payload (set by the Orchestrator).
def make_step_printer(label: str):
    tag = f"[{label.ljust(10)}]"

    class StepPrinter:
        async def emit(self, event):
            delegated = event.payload.get("delegated_to") if event.payload else None
            who = f"<{delegated}>" if delegated else tag
            if event.event_type == "thought":
                print(
                    f"  {who} [THOUGHT] {str(event.payload.get('content', ''))[:100]}"
                )
            elif event.event_type == "action":
                print(
                    f"  {who} [ACTION]  {event.payload.get('tool')} | {json.dumps(event.payload.get('args', {}))[:100]}"
                )
            elif event.event_type == "observation":
                print(
                    f"  {who} [OBS]     {str(event.payload.get('content', ''))[:100]}"
                )
            elif event.event_type == "tool_error":
                print(
                    f"  {who} [ERROR]   {event.payload.get('tool')} — {event.payload.get('error', '')}"
                )
            elif event.event_type == "answer":
                print(
                    f"  {who} [ANSWER]  {str(event.payload.get('content', ''))[:100]}"
                )

    return StepPrinter()


async def main():
    set_hook(make_step_printer("SUPERVISOR"))
    async with LLMClient.from_env() as client:
        fs_agent = Agent(
            client=client,
            tools=[list_directory, count_files],
            system_prompt="You are a filesystem assistant. Use the tools to inspect directories.",
        )

        math_agent = Agent(
            client=client,
            tools=[calculate],
            system_prompt="You are a calculator. Only perform arithmetic. Use the calculate tool.",
        )

        orchestrator = Orchestrator(
            client=client,
            sub_agents={"filesystem": fs_agent, "math": math_agent},
        )

        result = await orchestrator.run(
            "Count the files in ~/Desktop, then multiply that number by 42 and tell me the result."
        )

        print("\n--- ANSWER ---")
        print(result["answer"])
        print(
            f"[tokens] prompt={result['prompt_tokens']} completion={result['completion_tokens']} billable={result['total_billable_tokens']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
