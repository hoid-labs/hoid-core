import pytest

from hoid.observability import clear_hook, set_hook


class MockLLMClient:
    """Scriptable LLM client for unit tests.

    Accepts a list of responses that are returned in order. Each response is
    either a plain string (final answer with no tool calls) or a dict that is
    returned verbatim as the API response body.

    Attributes:
        calls: List of {"messages": ..., "tools": ...} dicts recorded per call.
    """

    def __init__(self, responses: list | None = None, final_answer: str = "done"):
        # fall back to a single final-answer response when none are scripted
        self._responses: list = responses if responses is not None else [final_answer]
        self._index = 0
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def chat_completions(self, messages, tools=None, **kwargs) -> dict:
        self.calls.append({"messages": list(messages), "tools": list(tools or [])})
        if self._index >= len(self._responses):
            # once scripted responses are exhausted keep returning the last one
            response = self._responses[-1]
        else:
            response = self._responses[self._index]
            self._index += 1

        if isinstance(response, str):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": response,
                            "tool_calls": None,
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        # caller supplied a raw response dict — return it unchanged
        return response


class RecordingHook:
    "Observability hook that appends every event to a list. Tests can inspect `events` after the run."

    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, event) -> None:
        self.events.append(event)


@pytest.fixture(autouse=True)
def _newline_before_output():
    """Print a blank line so test stdout starts on its own line (not glued to the test name)."""
    print()


@pytest.fixture
def mock_llm():
    "Return a factory for MockLLMClient; call it with an optional responses list."
    return MockLLMClient


@pytest.fixture
def recording_hook():
    "Install a global RecordingHook for the duration of the test; clear it on teardown."
    hook = RecordingHook()
    set_hook(hook)
    yield hook
    clear_hook()
