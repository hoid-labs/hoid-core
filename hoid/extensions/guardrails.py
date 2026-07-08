import json
import re
import time
from typing import Any

import httpx

from hoid.core.observability import GuardrailEvent


class GuardrailBlockedError(ValueError):
    """Raised by a guard when content is blocked by policy."""


# Structured output schema requesting a strict JSON response with status and reason fields.
_POLICY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "policy_evaluation",
        "strict": True,  # prevents the model from returning undeclared fields
        "schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["ALLOW", "BLOCK"],
                    "description": "The final decision. Must be ALLOW or BLOCK.",
                },
                "reason": {
                    "type": "string",
                    "description": "A short reason for the violation. Leave as an empty string if ALLOW.",
                },
            },
            "required": ["status", "reason"],
            "additionalProperties": False,
        },
    },
}


def block_keywords(words: list[str]):
    "Return a guard that raises if any blocked keyword appears in the text (case-insensitive)."
    pattern = re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)

    def guard(text: str) -> str:
        match = pattern.search(text)
        if match:
            raise GuardrailBlockedError(f"Blocked keyword detected: '{match.group()}'")
        return text

    return guard


def strip_pii():
    "Return a guard that replaces email addresses and phone numbers with redaction placeholders."
    # covers common email and phone patterns only
    email = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    phone = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")

    def guard(text: str) -> str:
        text = email.sub("[EMAIL]", text)
        text = phone.sub("[PHONE]", text)
        return text

    return guard


def llm_guard(client, policy: str, on_guard: Any | None = None):
    """Return an async guard that evaluates text against a natural language policy using an LLM.

    Structured output (``response_format``) is attempted first. Two fallback signals
    are handled, covering spec-compliant and non-compliant endpoint behaviour:

    - **HTTP 400** — the endpoint explicitly rejects ``response_format`` per the
      OpenAI-compatible API specification.
    - **Empty content with HTTP 200** — the endpoint silently ignores the parameter
      and returns an empty response body instead of an error.

    In both cases the guard retries without ``response_format``, relying on the
    system prompt instruction to produce well-formed JSON. All other errors propagate
    normally.

    Args:
        client: An `LLMClient` instance.
        policy: Natural-language policy description passed to the system prompt.
        on_guard: Optional observability callback receiving `GuardrailEvent` on each guard invocation.
    """
    _messages = [
        {
            "role": "system",
            "content": (
                f"You are a content policy enforcer. Policy: {policy}\n"
                "Evaluate the user's text against the policy.\n"
                "Respond with status=ALLOW if the text complies and should be permitted.\n"
                "Respond with status=BLOCK if the text violates the policy and should be rejected.\n"
                "When status is BLOCK you MUST provide a short reason explaining the violation.\n"
                "When status is ALLOW leave reason as an empty string.\n"
                'Always respond with a JSON object only, for example: {"status": "BLOCK", "reason": "Mentions blocked brand"}'
            ),
        },
    ]

    async def _call(text: str, response_format: dict | None) -> dict:
        kwargs: dict = dict(
            messages=_messages + [
                # XML tags prevent the evaluated text from being injected as instructions
                {"role": "user", "content": f"<evaluate>{text}</evaluate>"},
            ],
            temperature=0.0,
            # 512 tokens to leave room for models that spend tokens on internal
            # reasoning before emitting their output.
            max_tokens=512,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format
        return await client.chat_completions(**kwargs)

    # LLM evaluation catches contextual violations regex misses
    async def guard(text: str) -> str:
        start = time.perf_counter()
        try:
            # Attempt 1: structured output — enforced by endpoints that support it.
            response = await _call(text, _POLICY_SCHEMA)
            verdict_raw = response["choices"][0]["message"]["content"] or ""
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 400:
                raise
            # Per the OpenAI-compatible spec, HTTP 400 signals that response_format
            # is not supported. Fall back to prompt-only JSON instruction.
            response = await _call(text, None)
            verdict_raw = response["choices"][0]["message"]["content"] or ""

        if not verdict_raw.strip():
            # Some proxy implementations silently ignore unsupported response_format
            # parameters and return an empty response instead of HTTP 400. Fall back
            # to prompt-only so the system prompt instruction produces the JSON.
            response = await _call(text, None)
            verdict_raw = response["choices"][0]["message"]["content"] or ""

        try:
            verdict_data = json.loads(verdict_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Guardrail LLM returned invalid JSON: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000.0
        verdict_str = verdict_data["status"].upper()
        block = verdict_str == "BLOCK"
        reason_str = verdict_data.get("reason", "").strip() or None
        if block and not reason_str:
            reason_str = "Blocked by policy"
        if on_guard:
            await on_guard(
                GuardrailEvent(
                    guard_type="llm_guard",
                    verdict="block" if block else "allow",
                    policy=policy[:500] if policy else None,
                    latency_ms=latency_ms,
                    reason=reason_str,
                )
            )
        if block:
            raise GuardrailBlockedError(f"Guard blocked: {reason_str}")
        return text

    return guard
