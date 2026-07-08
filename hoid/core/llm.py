import asyncio
import logging
import os
import time
from typing import Any

import httpx

from hoid._env import load_env
from hoid.core.observability import (
    EmbeddingEvent,
    LLMCallEvent,
    TokenUsage,
    emit,
)

log = logging.getLogger(__name__)


def _usage_from_response(payload: dict[str, Any]) -> TokenUsage:
    """Extract a TokenUsage from a chat-completions response payload. Defaults to zeros."""
    usage = payload.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return TokenUsage(
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        reasoning_tokens=int(details.get("reasoning_tokens") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
    )


def _usage_from_embeddings_response(payload: dict[str, Any]) -> int:
    """Extract total_tokens from an embeddings response payload, defaulting to 0."""
    usage = payload.get("usage") or {}
    return int(usage.get("total_tokens") or 0)


class LLMClient:
    "Async HTTP client for OpenAI-compatible chat completion and embeddings endpoints."

    # --- construction ---

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        verify: str | bool = True,
    ):
        """
        Args:
            base_url: Base URL of the OpenAI-compatible API endpoint.
            api_key: API key; sent as both Bearer token and x-api-key for cross-provider compatibility.
            model: Model identifier to use for all requests.
            timeout: HTTP request timeout in seconds (default 120).
            verify: TLS verification — True for system CAs, False to skip, or a path to a CA bundle.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        # both formats for cross-provider compatibility
        headers = {
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }
        self._http = httpx.AsyncClient(timeout=timeout, headers=headers, verify=verify)

    @classmethod
    def from_env(cls) -> "LLMClient":
        "Construct from env vars; reads LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, CA_BUNDLE_PATH."
        load_env()
        ca_bundle = os.getenv("CA_BUNDLE_PATH") or None
        if ca_bundle and not os.path.isfile(ca_bundle):
            raise FileNotFoundError(
                f"CA_BUNDLE_PATH is set to '{ca_bundle}' but that file does not exist. "
                "Fix the path or unset CA_BUNDLE_PATH to use the system certificate store."
            )
        return cls(
            base_url=os.getenv("LLM_BASE_URL", ""),
            api_key=os.getenv("LLM_API_KEY", ""),
            model=os.getenv("LLM_MODEL", ""),
            verify=ca_bundle or True,
        )

    # --- API calls ---

    async def models(self) -> list[dict[str, Any]]:
        r = await self._http.get(f"{self.base_url}/models")
        self._check_response(r)
        return r.json().get("data", [])

    async def chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_retries: int = 3,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request with exponential backoff on transient server errors.

        Args:
            messages: Conversation history in OpenAI message format.
            tools: Optional list of JSON function schemas to expose to the model.
            temperature: Sampling temperature (default 0.7); use 0.0 for deterministic output.
            max_tokens: Maximum tokens in the response.
            max_retries: Retry attempts on 429/5xx errors with exponential backoff.
            response_format: Optional structured output schema (e.g. a json_schema object).

        Returns:
            Raw OpenAI-compatible response dict with choices, usage, and model fields.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format

        start = time.perf_counter()
        response: dict[str, Any] | None = None
        err: str | None = None
        try:
            for attempt in range(max_retries):
                try:
                    r = await self._http.post(
                        f"{self.base_url}/chat/completions", json=payload
                    )
                    self._check_response(r)
                    response = r.json()
                    return response
                except httpx.HTTPStatusError as e:
                    err = f"http_{e.response.status_code}"
                    # 429/5xx are transient; back off exponentially
                    if (
                        e.response.status_code in (429, 500, 502, 503, 504)
                        and attempt < max_retries - 1
                    ):
                        await asyncio.sleep(2**attempt)
                        continue
                    raise
                except httpx.RequestError as e:
                    err = f"request_{type(e).__name__}"
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            usage = _usage_from_response(response) if response else TokenUsage()
            response_id = (response or {}).get("id")
            await emit(
                LLMCallEvent(
                    op="chat_completions",
                    model=self.model,
                    messages_count=len(messages),
                    response_id=response_id,
                    usage=usage,
                    latency_ms=latency_ms,
                    error=err,
                )
            )
        raise RuntimeError(f"chat_completions failed after {max_retries} attempts")

    async def embeddings(
        self, input_texts: list[str], max_retries: int = 3
    ) -> list[list[float]]:
        """Return one embedding vector per input text.

        Args:
            input_texts: List of strings to embed.
            max_retries: Retry attempts on 429/5xx errors with exponential backoff.

        Returns:
            List of embedding vectors, one per input string, in the same order.
        """
        payload = {
            "model": self.model,
            "input": input_texts,
        }
        start = time.perf_counter()
        result: list[list[float]] | None = None
        err: str | None = None
        total_tokens = 0
        try:
            for attempt in range(max_retries):
                try:
                    r = await self._http.post(
                        f"{self.base_url}/embeddings", json=payload
                    )
                    self._check_response(r)
                    body = r.json()
                    data = body.get("data", [])
                    result = [item["embedding"] for item in data]
                    total_tokens = _usage_from_embeddings_response(body)
                    return result
                except httpx.HTTPStatusError as e:
                    err = f"http_{e.response.status_code}"
                    # 429/5xx are transient; back off exponentially
                    if (
                        e.response.status_code in (429, 500, 502, 503, 504)
                        and attempt < max_retries - 1
                    ):
                        await asyncio.sleep(2**attempt)
                        continue
                    raise
                except httpx.RequestError as e:
                    err = f"request_{type(e).__name__}"
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            await emit(
                EmbeddingEvent(
                    op="ingest",
                    model=self.model,
                    batch_size=len(input_texts),
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    error=err,
                )
            )
        raise RuntimeError(f"embeddings failed after {max_retries} attempts")

    # --- internal ---

    def _check_response(self, response: httpx.Response) -> None:
        # log body before raising so errors are diagnosable
        if not response.is_success:
            log.error("API error %s: %s", response.status_code, response.text)
        response.raise_for_status()

    # --- lifecycle ---

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
