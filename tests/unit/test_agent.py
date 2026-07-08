import json
import pprint

import pytest

from hoid.core import Agent
from hoid.core.tools import tool


@tool
def echo(text: str) -> str:
    "Return the input unchanged."
    return text


@tool
def boom(reason: str) -> str:
    """Always raises an exception.

    Args:
        reason: Why it should fail.
    """
    raise RuntimeError(reason)


def _tool_call_response(
    name: str,
    args: dict,
    call_id: str = "tc1",
    usage_overrides: dict | None = None,
) -> dict:
    "Build a raw LLM response that requests a tool call."
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    if usage_overrides:
        usage.update(usage_overrides)
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                }
            }
        ],
        "usage": usage,
    }


def _final_response(text: str, usage_overrides: dict | None = None) -> dict:
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    if usage_overrides:
        usage.update(usage_overrides)
    return {
        "choices": [
            {"message": {"role": "assistant", "content": text, "tool_calls": None}}
        ],
        "usage": usage,
    }


# --- construction ---


def test_registry_maps_name_to_callable():
    agent = Agent(object(), tools=[echo])
    print(f"registry: {list(agent.registry.keys())}")
    assert agent.registry["echo"] is echo


def test_schemas_built_from_tool_decorated_funcs():
    agent = Agent(object(), tools=[echo])
    names = [s["function"]["name"] for s in agent.schemas]
    print(f"schemas names: {names}")
    assert "echo" in names


def test_schemas_excludes_plain_callables():
    def plain(x: str) -> str:
        return x

    agent = Agent(object(), tools=[plain])
    names = [s["function"]["name"] for s in agent.schemas]
    assert "plain" not in names


def test_default_system_prompt_set():
    agent = Agent(object())
    assert agent.system_prompt


def test_custom_system_prompt_stored():
    agent = Agent(object(), system_prompt="You are a pirate.")
    assert agent.system_prompt == "You are a pirate."


def test_approval_tools_stored_as_set():
    agent = Agent(object(), approval_tools=["echo"])
    assert agent.approval_tools == {"echo"}


def test_approval_tools_none_means_all():
    agent = Agent(object(), approval_tools=None)
    assert agent.approval_tools is None


# --- run: basic flow ---


async def test_run_returns_answer(mock_llm):
    client = mock_llm(["hello back"])
    agent = Agent(client)
    result = await agent.run("say hello")
    print("agent.run result:")
    pprint.pprint(result)
    assert result["answer"] == "hello back"


async def test_run_calls_tool_and_returns_final_answer(mock_llm):
    client = mock_llm(
        [
            _tool_call_response("echo", {"text": "ping"}),
            _final_response("pong"),
        ]
    )
    agent = Agent(client, tools=[echo])
    result = await agent.run("ping")
    print(f"LLM call count: {len(client.calls)}  answer: {result['answer']!r}")
    assert result["answer"] == "pong"


# --- token accounting ---


async def test_total_billable_tokens_prompt_plus_completion_only(mock_llm):
    """reasoning_tokens is a subset of completion_tokens; billable = prompt + completion."""
    client = mock_llm(
        [
            _tool_call_response(
                "echo",
                {"text": "x"},
                usage_overrides={
                    "prompt_tokens": 10,
                    "completion_tokens": 100,
                    "completion_tokens_details": {"reasoning_tokens": 40},
                },
            ),
            _final_response("done", usage_overrides={"prompt_tokens": 0, "completion_tokens": 0}),
        ]
    )
    agent = Agent(client, tools=[echo])
    result = await agent.run("hi")
    print(
        f"prompt={result['prompt_tokens']} completion={result['completion_tokens']} "
        f"reasoning={result['reasoning_tokens']} billable={result['total_billable_tokens']}"
    )
    assert result["prompt_tokens"] == 10
    assert result["completion_tokens"] == 100
    assert result["reasoning_tokens"] == 40
    assert result["total_billable_tokens"] == 110


async def test_total_billable_tokens_cumulative_across_iterations(mock_llm):
    """billable is cumulative across ReAct iterations, not per-step."""
    client = mock_llm(
        [
            _tool_call_response(
                "echo",
                {"text": "a"},
                usage_overrides={
                    "prompt_tokens": 10,
                    "completion_tokens": 50,
                    "completion_tokens_details": {"reasoning_tokens": 20},
                },
            ),
            _tool_call_response(
                "echo",
                {"text": "b"},
                usage_overrides={
                    "prompt_tokens": 12,
                    "completion_tokens": 60,
                    "completion_tokens_details": {"reasoning_tokens": 25},
                },
            ),
            _final_response(
                "done",
                usage_overrides={
                    "prompt_tokens": 8,
                    "completion_tokens": 30,
                },
            ),
        ]
    )
    agent = Agent(client, tools=[echo])
    result = await agent.run("multi-step")
    print(
        f"prompt={result['prompt_tokens']} completion={result['completion_tokens']} "
        f"reasoning={result['reasoning_tokens']} billable={result['total_billable_tokens']}"
    )
    assert result["prompt_tokens"] == 10 + 12 + 8
    assert result["completion_tokens"] == 50 + 60 + 30
    assert result["reasoning_tokens"] == 20 + 25
    assert result["total_billable_tokens"] == (10 + 12 + 8) + (50 + 60 + 30)


# --- input/output guards ---


async def test_input_guard_applied_before_llm(mock_llm):
    captured = {}

    def capture_guard(text: str) -> str:
        captured["input"] = text
        return text.upper()

    client = mock_llm(["ok"])
    agent = Agent(client, input_guards=[capture_guard])
    await agent.run("hello")
    # the guard must have transformed the prompt
    assert captured["input"] == "hello"
    sent_user_msg = client.calls[0]["messages"][-1]["content"]
    print(
        f"guard captured input: {captured['input']!r}  sent to LLM: {sent_user_msg!r}"
    )
    assert sent_user_msg == "HELLO"


async def test_input_guard_raising_propagates(mock_llm):
    def blocking_guard(text: str) -> str:
        raise ValueError("blocked by guard")

    client = mock_llm()
    agent = Agent(client, input_guards=[blocking_guard])
    with pytest.raises(ValueError, match="blocked by guard"):
        await agent.run("some prompt")


async def test_output_guard_applied_to_final_answer(mock_llm):
    def redact_guard(text: str) -> str:
        return text.replace("secret", "[REDACTED]")

    client = mock_llm(["the secret is safe"])
    agent = Agent(client, output_guards=[redact_guard])
    result = await agent.run("what is the secret")
    print(f"output after redact guard: {result['answer']!r}")
    assert result["answer"] == "the [REDACTED] is safe"


# --- approval ---


async def test_approval_callback_called_before_tool(mock_llm):
    calls = []

    def approve(name, args):
        calls.append((name, args))
        return True

    client = mock_llm(
        [_tool_call_response("echo", {"text": "hi"}), _final_response("done")]
    )
    agent = Agent(client, tools=[echo], approval_callback=approve)
    await agent.run("go")
    print(f"approval calls: {calls}")
    assert any(name == "echo" for name, _ in calls)


async def test_approval_callback_false_skips_tool(mock_llm):
    executed = []

    @tool
    def track(msg: str) -> str:
        "Track a message."
        executed.append(msg)
        return "tracked"

    def deny(name, args):
        return False

    client = mock_llm(
        [_tool_call_response("track", {"msg": "hi"}), _final_response("done")]
    )
    agent = Agent(client, tools=[track], approval_callback=deny)
    await agent.run("go")
    print(f"executed (should be empty, approval denied): {executed}")
    assert executed == []


# --- max_steps ---


async def test_max_steps_limits_iterations(mock_llm):
    # always returns a tool call — forces the agent to loop until max_steps

    class InfiniteToolCallClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def chat_completions(self, messages, tools=None, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "t1",
                                    "function": {
                                        "name": "echo",
                                        "arguments": json.dumps({"text": "loop"}),
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            }

    agent = Agent(InfiniteToolCallClient(), tools=[echo], max_steps=3)
    result = await agent.run("run forever")
    # agent must terminate and return something — not hang
    print("result after max_steps=3:")
    pprint.pprint(result)
    assert "answer" in result


# --- observability hook ---


async def test_delegated_to_propagates_to_emitted_events(mock_llm, recording_hook):
    "When the Orchestrator calls a sub-agent, every emitted event carries delegated_to in its payload."
    client = mock_llm(["done"])
    agent = Agent(client)
    await agent.run("hi", delegated_to="orchestrator.researcher")
    print(f"event count: {len(recording_hook.events)}")
    for e in recording_hook.events:
        print(f"  {e.event_type}: delegated_to={e.payload.get('delegated_to')!r}")
    assert len(recording_hook.events) >= 1
    assert all(
        e.payload.get("delegated_to") == "orchestrator.researcher"
        for e in recording_hook.events
    )


async def test_delegated_to_cleared_after_run(mock_llm, recording_hook):
    "A second run without delegated_to must not inherit state from a prior delegated run."
    client = mock_llm(["done"])
    agent = Agent(client)
    await agent.run("hi", delegated_to="orchestrator.x")
    recording_hook.events.clear()
    await agent.run("plain")
    assert all(
        e.payload.get("delegated_to") is None for e in recording_hook.events
    )


async def test_recording_hook_receives_task_event(mock_llm, recording_hook):
    client = mock_llm(["done"])
    agent = Agent(client)
    await agent.run("my task")
    event_types = [e.event_type for e in recording_hook.events]
    print(f"event types: {event_types}")
    assert "task" in event_types


async def test_recording_hook_receives_action_and_observation(mock_llm, recording_hook):
    client = mock_llm(
        [_tool_call_response("echo", {"text": "ping"}), _final_response("pong")]
    )
    agent = Agent(client, tools=[echo])
    await agent.run("ping")
    event_types = [e.event_type for e in recording_hook.events]
    print(f"event types: {event_types}")
    assert "action" in event_types
    assert "observation" in event_types


# --- tool error handling ---


async def test_tool_exception_returned_as_observation(mock_llm, recording_hook):
    client = mock_llm(
        [
            _tool_call_response("boom", {"reason": "test error"}),
            _final_response("handled"),
        ]
    )
    agent = Agent(client, tools=[boom])
    result = await agent.run("do it")
    # agent must survive the tool exception and complete
    print(f"answer after tool exception: {result['answer']!r}")
    print("tool_error events:")
    pprint.pprint([e for e in recording_hook.events if e.event_type == "tool_error"])
    assert result["answer"] == "handled"
    assert any(e.event_type == "tool_error" for e in recording_hook.events)


async def test_unknown_tool_returns_error_observation(mock_llm, recording_hook):
    client = mock_llm([_tool_call_response("nonexistent", {}), _final_response("ok")])
    agent = Agent(client, tools=[])
    result = await agent.run("call unknown")
    print(f"answer after unknown tool: {result['answer']!r}")
    print("tool_error events:")
    pprint.pprint([e for e in recording_hook.events if e.event_type == "tool_error"])
    assert result["answer"] == "ok"
    assert any(e.event_type == "tool_error" for e in recording_hook.events)
