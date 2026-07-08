"""Tests for the Agent ReAct loop: while-loop correctness and before_iteration_callback."""

from __future__ import annotations

import json
import pprint

from hoid.core import Agent
from hoid.core.tools import tool


@tool
def noop(x: str) -> str:
    """Do nothing; return input.

    Args:
        x: Any string.
    """
    return x


class _ToolCallClient:
    """Returns N tool calls then a final answer."""

    def __init__(self, tool_call_count: int, final: str = "done"):
        self._tool_call_count = tool_call_count
        self._call_count = 0
        self.calls: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def chat_completions(self, messages, tools=None, **kwargs):
        self.calls.append(messages)
        if self._call_count < self._tool_call_count:
            self._call_count += 1
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"tc{self._call_count}",
                                    "function": {
                                        "name": "noop",
                                        "arguments": json.dumps({"x": f"step-{self._call_count}"}),
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            }
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "done", "tool_calls": None}}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }


async def test_max_steps_respected_by_while_loop():
    """Agent must not exceed max_steps iterations."""
    client = _ToolCallClient(tool_call_count=100)
    agent = Agent(client, tools=[noop], max_steps=3)
    result = await agent.run("go")
    pprint.pprint(result)
    assert "answer" in result
    # 3 LLM calls: steps 1, 2, 3 (step 3 forces exit without a 4th call)
    assert len(client.calls) <= 3


async def test_before_iteration_callback_extends_max_steps():
    """Incrementing max_steps inside before_iteration_callback must extend the run.

    Without the while-loop fix, range(max_steps) would be evaluated once and
    the callback mutation would be silently ignored.
    """
    client = _ToolCallClient(tool_call_count=10)
    agent = Agent(client, tools=[noop], max_steps=3)

    steps_seen: list[int] = []

    async def _extend(step: int, max_steps: int) -> None:
        steps_seen.append(step)
        # On step 1 (0-indexed), extend by 2 — this gives steps 0..4 = 5 total
        if step == 1 and agent.max_steps == 3:
            agent.max_steps += 2

    agent.before_iteration_callback = _extend
    result = await agent.run("run with extension")
    pprint.pprint(result)
    assert "answer" in result
    # callback must have been invoked at least 5 times (steps 0-4)
    assert len(steps_seen) >= 5, f"only saw steps {steps_seen}"
    # The extension must have taken effect: we see step 4 (0-indexed)
    assert 4 in steps_seen


async def test_before_iteration_callback_sync_also_works():
    """A synchronous (non-async) callback must also be called without errors."""
    client = _ToolCallClient(tool_call_count=0)
    agent = Agent(client, max_steps=2)

    called_at: list[int] = []

    def _sync_cb(step: int, max_steps: int) -> None:
        called_at.append(step)

    agent.before_iteration_callback = _sync_cb
    result = await agent.run("one step")
    pprint.pprint(result)
    assert "answer" in result
    assert 0 in called_at


async def test_callback_receives_correct_max_steps():
    """max_steps passed to the callback must match agent.max_steps at that iteration."""
    client = _ToolCallClient(tool_call_count=0)
    agent = Agent(client, max_steps=4)

    received: list[tuple[int, int]] = []

    async def _record(step: int, max_steps: int) -> None:
        received.append((step, max_steps))

    agent.before_iteration_callback = _record
    await agent.run("check args")
    assert received
    step, ms = received[0]
    assert step == 0
    assert ms == 4
