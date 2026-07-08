"""Observability primitives for the LLM framework.

A small, dependency-free vocabulary for emitting and routing events from core
call sites. No I/O happens here — this module is pure data and dispatch. A
user of the library who wants to observe LLM calls registers an
`ObservabilityHook` implementation and decides where events go (storage,
format, transport are all up to the consumer).

Public surface:
    - `ObservabilityContext` — free-form request-scoped metadata
    - `set_context` / `get_context` / `reset_context` — ContextVar helpers
    - `ObservabilityHook` — protocol an observer must implement
    - `set_hook` / `get_hook` / `clear_hook` — global hook registration
    - `emit(event)` — attach current context to an event and forward to the hook
    - `TokenUsage` and the event dataclasses — what gets emitted
    - `Event` — common protocol on every event

Backward compatibility:
    - All hooks are opt-in. With no hook registered, `emit()` is a no-op.
    - All new parameters on existing core classes default to `None` and add no overhead.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

log = logging.getLogger(__name__)


# --- context ---


@dataclass(frozen=True)
class ObservabilityContext:
    """Request-scoped metadata attached to every event emitted from this async context.

    A free-form tag bag.

    Example:
        set_context(ObservabilityContext({
            "caller_id": "alice",
            "request_id": "abc-123",
            "source": "agents_api",
        }))
    """

    context: dict[str, Any] = field(default_factory=dict)


_context_var: ContextVar[ObservabilityContext | None] = ContextVar(
    "hoid_observability_context", default=None
)


def get_context() -> ObservabilityContext | None:
    """Return the current `ObservabilityContext` or `None` if none is set."""
    return _context_var.get()


def set_context(ctx: ObservabilityContext) -> Token:
    """Set the context for the current async scope. Return a token to pass to `reset_context`."""
    return _context_var.set(ctx)


def reset_context(token: Token) -> None:
    """Restore the previous context using the token returned by `set_context`."""
    _context_var.reset(token)


# --- hook ---


@runtime_checkable
class ObservabilityHook(Protocol):
    """An observer that receives events emitted by core call sites.

    Implementations decide what to do with events: write to a database, print
    to stdout, ship to a metrics backend, or ignore. Core wraps the call in
    try/except so observers must not raise, but a defensive implementation
    should still catch its own exceptions.
    """

    async def emit(self, event: Event) -> None: ...


_hook_var: ObservabilityHook | None = None


def set_hook(hook: ObservabilityHook | None) -> None:
    """Register a global observability hook. Pass `None` to clear."""
    global _hook_var
    _hook_var = hook


def get_hook() -> ObservabilityHook | None:
    """Return the currently registered hook or `None`."""
    return _hook_var


def clear_hook() -> None:
    """Clear the global hook (equivalent to `set_hook(None)`)."""
    set_hook(None)


# --- events ---


@dataclass(frozen=True)
class TokenUsage:
    """Token accounting for a single LLM call. Mirrors what core already extracts."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    context_tokens: int = 0


@runtime_checkable
class Event(Protocol):
    """Common protocol every emitted event implements.

    Concrete events are frozen dataclasses; `ctx` is attached by `emit()` and
    is not part of the event's own state. Structural — any object with a
    matching attribute set counts.
    """

    ctx: ObservabilityContext | None


@dataclass(frozen=True)
class LLMCallEvent:
    """One LLM chat-completion call (or, for embeddings, an LLM embed call with op='embeddings')."""

    op: Literal["chat_completions", "embeddings"]
    model: str
    messages_count: int = 0
    response_id: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    error: str | None = None
    ctx: ObservabilityContext | None = None


@dataclass(frozen=True)
class EmbeddingEvent:
    """Embedding-batch call. Distinct from `LLMCallEvent` so observers can bucket embed traffic separately from chat completions."""

    op: Literal["ingest", "search"]
    model: str
    batch_size: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None
    ctx: ObservabilityContext | None = None


@dataclass(frozen=True)
class RAGEvent:
    """Higher-level RAG operation: ingest or search on a collection."""

    op: Literal["ingest", "search"]
    collection_id: str
    doc_id: str | None = None
    chunk_count: int = 0
    latency_ms: float = 0.0
    error: str | None = None
    ctx: ObservabilityContext | None = None


@dataclass(frozen=True)
class MCPEvent:
    """One MCP tool call: server name, tool name, latency, error."""

    server: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None
    ctx: ObservabilityContext | None = None


@dataclass(frozen=True)
class GuardrailEvent:
    """A guardrail verdict: allow or block, with the guard type and reason."""

    guard_type: str
    verdict: Literal["allow", "block"]
    input_hash: str | None = None
    policy: str | None = None
    latency_ms: float = 0.0
    reason: str | None = None
    ctx: ObservabilityContext | None = None


@dataclass(frozen=True)
class AgentStepEvent:
    """One ReAct-loop event from an Agent (task, thought, action, observation, answer, error)."""

    event_type: str
    step: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    ctx: ObservabilityContext | None = None


@dataclass(frozen=True)
class OrchestratorEvent:
    """Orchestrator-level event: supervisor or sub-agent step."""

    event_type: str
    supervisor_id: str | None = None
    sub_agent_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    ctx: ObservabilityContext | None = None


@dataclass(frozen=True)
class PipelineStepEvent:
    """Pipeline step boundary event: started, step, finished, errored."""

    event_type: str
    pipeline_id: str | None = None
    step_index: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    ctx: ObservabilityContext | None = None


# --- emit ---


async def emit(event: Any) -> None:
    """Attach the current context to an event and forward to the registered hook.

    No-op if no hook is registered. Never raises — observability failure is
    logged via stdlib and discarded.
    """
    hook = _hook_var
    if hook is None:
        return
    ctx = _context_var.get()
    try:
        # attach ctx if the event has a `ctx` attribute and doesn't already have one
        if hasattr(event, "ctx") and event.ctx is None and ctx is not None:
            event = _attach_ctx(event, ctx)
        await hook.emit(event)
    except Exception:
        log.debug("observability hook raised; dropping event", exc_info=True)


def emit_sync(event: Any) -> None:
    """Synchronous emit: schedules the async emit on the current event loop if one is running.

    Use this from sync code paths that cannot await (e.g. inside a callback that
    is itself called from a non-async context). If no loop is running, the
    event is silently dropped.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(emit(event))


def _attach_ctx(event: Any, ctx: ObservabilityContext) -> Any:
    """Return a copy of `event` with `ctx` set. Frozen dataclasses require `object.__setattr__` or `dataclasses.replace`."""
    if dataclasses.is_dataclass(event) and not isinstance(event, type):
        return dataclasses.replace(event, ctx=ctx)  # type: ignore[misc]
    # fallback for non-dataclass events
    return event


# --- ready-to-use hooks ---


class _PrintHook:
    "Built-in hook that prints each event to a stream. Useful for examples and ad-hoc debugging."

    def __init__(self, stream: Any = None) -> None:
        self._stream = stream

    async def emit(self, event: Any) -> None:
        import json

        # event_type and tokens are typed attributes on the dataclass; everything
        # else lives in payload. Fall back to str() for events without a payload.
        payload = getattr(event, "payload", None)
        tokens = getattr(event, "tokens", None)
        line = f"[{event.event_type}] {json.dumps(payload, default=str) if payload else ''}"
        if tokens and (tokens.prompt_tokens or tokens.completion_tokens):
            line += f" tokens=p{tokens.prompt_tokens}/c{tokens.completion_tokens}"
        print(line, file=self._stream)


def print_hook(stream: Any = None) -> ObservabilityHook:
    "Return an `ObservabilityHook` that prints each event to a stream (default stdout)."
    return _PrintHook(stream)
