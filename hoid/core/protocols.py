"""Structural Protocols for collaborators that `Agent` depends on.

These Protocols let `core/` declare the shape of the objects it accepts
(client, auth gate, approval callback, etc.) without importing the concrete
implementations from `extensions/`. Any class that structurally matches is
accepted, preserving the duck-typed design while making mypy honest.

Why Protocols and not ABCs:
    - No runtime dependency from `core` on `extensions` (one-way dependency).
    - The concrete `AuthGate`, `AuthContext`, etc. in `extensions/auth/` do not
      need to inherit from anything here — they satisfy the Protocol structurally.
    - `runtime_checkable` is enabled only where `isinstance` checks are useful
      (e.g. debugging); it is not required for the type system to do its job.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


@runtime_checkable
class AuthContextProtocol(Protocol):
    """Structural interface for an authenticated request context.

    The concrete `AuthContext` dataclass in `extensions/auth/_context.py`
    satisfies this Protocol — it has `user_id: str`, `roles: set[str]`, and
    `attributes: dict[str, Any]`. The Protocol exists so that `core/agent.py`
    can reference the shape without importing the dataclass at runtime.
    """

    user_id: str
    roles: set[str]
    attributes: dict[str, Any]


@runtime_checkable
class AuthGateProtocol(Protocol):
    """Structural interface for an authorization gate.

    The concrete `AuthGate` in `extensions/auth/__init__.py` satisfies this
    Protocol — it implements `filter_schemas` and `authorize` with the
    signatures declared below.

    Both methods must accept `None` for the context (meaning "no caller
    authenticated" / "treat as fully authorized"), matching the behavior of
    the concrete `AuthGate`.
    """

    def filter_schemas(
        self,
        schemas: list[dict[str, Any]],
        context: AuthContextProtocol | None,
    ) -> list[dict[str, Any]]: ...

    def authorize(
        self,
        tool_name: str,
        context: AuthContextProtocol | None,
    ) -> bool: ...


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


class LLMClientProtocol(Protocol):
    """Structural interface for an OpenAI-compatible chat completion client.

    The concrete `LLMClient` in `core/llm.py` satisfies this Protocol.
    Declared here so `Agent` can annotate its `client` parameter without
    importing the concrete class (which would be a same-package circular ref).
    """

    async def chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_retries: int = 3,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class EmbeddingClientProtocol(Protocol):
    """Structural interface for an OpenAI-compatible embeddings client.

    The concrete `LLMClient` in `core/llm.py` satisfies this Protocol.
    Declared here so `RAGStore` can annotate its `llm_client` parameter
    without importing the concrete class.
    """

    async def embeddings(
        self,
        input_texts: list[str],
        max_retries: int = 3,
    ) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Agent runner (used by HistoryBuffer to avoid importing Agent directly)
# ---------------------------------------------------------------------------


class AgentRunProtocol(Protocol):
    """Minimal interface for objects that can be run by HistoryBuffer.

    Satisfied by `Agent` and any compatible wrapper.
    """

    async def run(
        self,
        prompt: str,
        system_prompt: str | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        auth_context: Any = None,
    ) -> dict[str, Any]: ...


__all__ = [
    "AgentRunProtocol",
    "AuthContextProtocol",
    "AuthGateProtocol",
    "EmbeddingClientProtocol",
    "LLMClientProtocol",
]
