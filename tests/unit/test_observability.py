import asyncio
from dataclasses import FrozenInstanceError

import pytest

from hoid.observability import (
    AgentStepEvent,
    EmbeddingEvent,
    Event,
    GuardrailEvent,
    LLMCallEvent,
    MCPEvent,
    ObservabilityContext,
    ObservabilityHook,
    OrchestratorEvent,
    PipelineStepEvent,
    RAGEvent,
    TokenUsage,
    clear_hook,
    emit,
    get_context,
    get_hook,
    reset_context,
    set_context,
    set_hook,
)

# --- context ---


def test_context_default_is_none():
    clear_contexts()
    assert get_context() is None


def test_set_and_get_context():
    clear_contexts()
    ctx = ObservabilityContext({"caller_id": "u1", "request_id": "r1"})
    token = set_context(ctx)
    try:
        assert get_context() is ctx
    finally:
        reset_context(token)
    assert get_context() is None


def test_context_propagates_through_asyncio_create_task():
    clear_contexts()
    ctx = ObservabilityContext({"caller_id": "u1", "request_id": "r1"})
    token = set_context(ctx)
    try:

        async def inner():
            return get_context()

        result = asyncio.run(inner())
        assert result is ctx
    finally:
        reset_context(token)


def test_context_is_default_empty_dict():
    ctx = ObservabilityContext()
    assert ctx.context == {}


def test_context_accepts_arbitrary_keys():
    """The context is a free-form tag bag — any key is preserved verbatim."""
    ctx = ObservabilityContext(
        {
            "caller_id": "alice",
            "request_id": "abc-123",
            "source": "agents_api",
            "weird_custom_key": {"nested": [1, 2, 3]},
        }
    )
    assert ctx.context["caller_id"] == "alice"
    assert ctx.context["weird_custom_key"] == {"nested": [1, 2, 3]}


def test_context_is_frozen():
    ctx = ObservabilityContext({"caller_id": "u1"})
    with pytest.raises(FrozenInstanceError):
        ctx.context = {}  # type: ignore[misc]


# --- hook ---


def test_hook_default_is_none():
    clear_hook()
    assert get_hook() is None


def test_set_and_get_hook():
    clear_hook()
    hook = _RecorderHook()
    set_hook(hook)
    try:
        assert get_hook() is hook
    finally:
        clear_hook()
    assert get_hook() is None


def test_observability_hook_is_a_protocol():
    # The Protocol is structural; any class with `async def emit(self, event)` matches.
    class MyHook:
        def __init__(self):
            self.events = []

        async def emit(self, event):
            self.events.append(event)

    h = MyHook()
    assert isinstance(h, ObservabilityHook)


# --- emit ---


def test_emit_no_op_without_hook():
    clear_hook()
    clear_contexts()
    asyncio.run(emit(LLMCallEvent(op="chat_completions", model="m1")))
    # nothing to assert; the call must not raise


def test_emit_forwards_to_hook_with_ctx():
    clear_hook()
    clear_contexts()
    recorder = _RecorderHook()
    set_hook(recorder)
    ctx = ObservabilityContext({"caller_id": "u1", "request_id": "r1"})
    token = set_context(ctx)
    try:
        asyncio.run(
            emit(LLMCallEvent(op="chat_completions", model="m1", messages_count=3))
        )
    finally:
        reset_context(token)
        clear_hook()
    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert isinstance(event, LLMCallEvent)
    assert event.model == "m1"
    assert event.ctx is ctx


def test_emit_preserves_explicit_ctx():
    """If the caller already attached a ctx, emit() should not overwrite it."""
    clear_hook()
    clear_contexts()
    recorder = _RecorderHook()
    set_hook(recorder)
    ambient = ObservabilityContext({"caller_id": "ambient"})
    explicit = ObservabilityContext({"caller_id": "explicit"})
    token = set_context(ambient)
    try:
        asyncio.run(emit(LLMCallEvent(op="chat_completions", model="m1", ctx=explicit)))
    finally:
        reset_context(token)
        clear_hook()
    assert recorder.events[0].ctx is explicit


def test_emit_swallows_hook_exceptions():
    clear_hook()
    clear_contexts()
    set_hook(_RaisingHook())
    asyncio.run(emit(LLMCallEvent(op="chat_completions", model="m1")))
    # must not raise


def test_emit_swallows_sync_hook_exceptions():
    """Sync hooks are not supported by the protocol, but emit() must still not raise."""
    clear_hook()
    clear_contexts()
    set_hook(_SyncRaisingHook())
    asyncio.run(emit(LLMCallEvent(op="chat_completions", model="m1")))


# --- event dataclasses ---


def test_token_usage_defaults():
    u = TokenUsage()
    assert u.prompt_tokens == 0
    assert u.completion_tokens == 0
    assert u.reasoning_tokens == 0
    assert u.total_tokens == 0
    assert u.context_tokens == 0


@pytest.mark.parametrize(
    "event",
    [
        LLMCallEvent(op="chat_completions", model="m1"),
        EmbeddingEvent(op="ingest", model="m1"),
        RAGEvent(op="search", collection_id="c1"),
        MCPEvent(server="s1", tool="t1"),
        GuardrailEvent(guard_type="g1", verdict="allow"),
        AgentStepEvent(event_type="task"),
        OrchestratorEvent(event_type="delegated"),
        PipelineStepEvent(event_type="step_started"),
    ],
)
def test_event_is_a_protocol(event):
    assert isinstance(event, Event)


def test_event_is_frozen():
    e = LLMCallEvent(op="chat_completions", model="m1")
    with pytest.raises(FrozenInstanceError):
        e.model = "m2"  # type: ignore[misc]


def test_attach_ctx_returns_new_instance():
    """Frozen dataclasses must produce a new instance, not mutate."""
    import dataclasses

    e = LLMCallEvent(op="chat_completions", model="m1")
    ctx = ObservabilityContext({"caller_id": "u1"})
    e2 = dataclasses.replace(e, ctx=ctx)
    assert e is not e2
    assert e.ctx is None
    assert e2.ctx is ctx


# --- helpers ---


def clear_contexts():
    """Best-effort reset of any leftover context from prior tests."""
    _context_var_reset()


def _context_var_reset():
    from hoid.observability import _context_var

    _context_var.set(None)


class _RecorderHook:
    def __init__(self):
        self.events: list = []

    async def emit(self, event):
        self.events.append(event)


class _RaisingHook:
    async def emit(self, event):
        raise RuntimeError("boom")


class _SyncRaisingHook:
    def emit(self, event):
        raise RuntimeError("boom")
