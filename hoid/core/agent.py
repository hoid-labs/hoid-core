from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from hoid.core.observability import AgentStepEvent, TokenUsage
from hoid.core.observability import emit as _obs_emit

from .protocols import AuthContextProtocol, AuthGateProtocol, LLMClientProtocol

if TYPE_CHECKING:
    pass

_MAX_CHARS = 20_000


class Agent:
    "ReAct agent that iterates tool calls until it reaches a final text answer."

    # --- construction ---

    def __init__(
        self,
        client: LLMClientProtocol,
        tools: list | None = None,
        max_steps: int = 10,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        max_retries: int = 3,
        system_prompt: str | None = None,
        input_guards: list | None = None,
        output_guards: list | None = None,
        approval_callback: Callable[..., object] | None = None,
        approval_tools: list[str] | None = None,
        auth_gate: AuthGateProtocol | None = None,
        before_iteration_callback: Callable[[int, int], object] | None = None,
        on_step: Callable[[AgentStepEvent], object] | None = None,
    ) -> None:
        """
        Args:
            client: An `LLMClient` instance for sending chat completion requests.
            tools: List of `@tool`-decorated callables available during the ReAct loop.
            max_steps: Maximum reasoning iterations before returning a fallback answer (default 10).
            max_tokens: Maximum tokens per LLM response (default 4096). Increase when the model must reproduce large tool results verbatim.
            temperature: Sampling temperature (default 0.7). Use 0.0 for deterministic tasks like structured extraction or KQL generation.
            max_retries: LLM request retry attempts with exponential backoff on 429/5xx errors (default 3).
            system_prompt: Persistent persona injected as the system message on every `run()` call; overridable per-call.
            input_guards: List of callables `(str) -> str` applied to the prompt before the LLM sees it. Raise to block, return to pass/transform.
            output_guards: List of callables `(str) -> str` applied to the final answer. Raise to block, return to pass/transform.
            approval_callback: Sync or async `(name: str, args: dict) -> bool` called before each tool runs. Return `False` to deny.
            approval_tools: Set of tool names that require approval. If `None`, every tool requires approval.
            auth_gate: An `AuthGate` instance that filters tool schemas and enforces access at execution time. `None` disables auth (all tools allowed).
            before_iteration_callback: Sync or async callable `(step: int, max_steps: int) -> None` invoked at the start of each ReAct iteration. Enables callers to mutate `max_steps` at runtime to extend or shorten the iteration limit without subclassing.
            on_step: Optional callable receiving `AgentStepEvent` at each ReAct iteration. Fires in addition to the global observability hook. Use the global hook (`hoid.observability.set_hook()`) for typical observability wiring.
        """
        self.client = client
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        # stored as the agent's default persona; run() can override per-call
        self.system_prompt = (
            system_prompt or "Solve tasks step-by-step. Use tools when necessary."
        )
        self.tools = tools or []
        # registry maps function name → callable for dispatch during tool calls
        self.registry = {f.__name__: f for f in (tools or [])}
        # schemas is the list of JSON function descriptors sent to the LLM
        self.schemas = [f.schema for f in (tools or []) if hasattr(f, "schema")]
        self.input_guards = input_guards or []
        self.output_guards = output_guards or []
        self.approval_callback = approval_callback
        # None means every tool requires approval; a set scopes it to specific names
        self.approval_tools = (
            set(approval_tools) if approval_tools is not None else None
        )
        # None disables auth — all tools are available (backward-compatible default)
        self.auth_gate = auth_gate
        self.before_iteration_callback = before_iteration_callback
        self.on_step = on_step

    # --- event emission ---

    async def _emit(self, event: AgentStepEvent) -> None:
        # tag sub-agent events with the delegating agent name (set by Orchestrator)
        if getattr(self, "_current_delegated_to", None):
            event = dataclasses.replace(
                event,
                payload={**event.payload, "delegated_to": self._current_delegated_to},
            )
        if self.on_step:
            result = self.on_step(event)
            if asyncio.iscoroutine(result):
                await result
        await _obs_emit(event)

    def _step_event(
        self,
        event_type: str,
        payload: dict,
        *,
        step: int = 0,
        tokens: TokenUsage | None = None,
    ) -> AgentStepEvent:
        return AgentStepEvent(
            event_type=event_type,
            step=step,
            payload=payload,
            tokens=tokens or TokenUsage(),
        )

    # --- tool validation and execution ---

    def _validate_args(self, name: str, args: dict[str, Any]) -> str | None:
        # skip validation for tools without a schema (e.g. MCP proxies before schema injection)
        func = self.registry.get(name)
        if not func or not hasattr(func, "schema"):
            return None
        params = func.schema["function"]["parameters"]
        required = params.get("required", [])
        missing = [p for p in required if p not in args]
        if missing:
            return f"Missing required arguments for '{name}': {missing}"
        return None

    async def _execute_tool(
        self,
        name: str,
        args: dict[str, Any],
        auth_context: AuthContextProtocol | None = None,
    ) -> str:
        # registry lookup
        if name not in self.registry:
            error = f"Tool '{name}' not found."
            await self._emit(self._step_event("tool_error", {"tool": name, "error": error}))
            return f"Error: {error}"

        # argument validation against the tool's JSON schema
        validation_error = self._validate_args(name, args)
        if validation_error:
            await self._emit(
                self._step_event("tool_error", {"tool": name, "error": validation_error})
            )
            return f"Error: {validation_error}"

        # execution-time authorization check — defense-in-depth against prompt injection
        if self.auth_gate and not self.auth_gate.authorize(name, auth_context):
            error = f"Access denied: '{name}' is not permitted for this user."
            await self._emit(
                self._step_event(
                    "tool_error",
                    {"tool": name, "error": error, "reason": "access_denied"},
                )
            )
            return f"Error: {error}"

        # HITL approval gate — pauses execution until the callback responds
        needs_approval = self.approval_callback and (
            self.approval_tools is None or name in self.approval_tools
        )
        if needs_approval:
            await self._emit(
                self._step_event("waiting_for_approval", {"tool": name, "args": args})
            )
            approved = self.approval_callback(name, args)  # type: ignore[misc]
            if asyncio.iscoroutine(approved):
                approved = await approved
            if not approved:
                error = f"Tool '{name}' was denied by the approval callback."
                await self._emit(self._step_event("tool_error", {"tool": name, "error": error}))
                return f"Error: {error}"

        # execution — supports both sync and async tools
        try:
            func = self.registry[name]
            result = func(**args)
            if asyncio.iscoroutine(result):
                result = await result
            result = result if isinstance(result, str) else json.dumps(result)
            # truncate large observations so they don't flood the context window
            if len(result) > _MAX_CHARS:
                return (
                    result[:_MAX_CHARS] + "\n...[OBSERVATION TRUNCATED DUE TO LENGTH]"
                )
            return result
        except Exception as e:
            error = str(e)
            await self._emit(self._step_event("tool_error", {"tool": name, "error": error}))
            return f"Error executing {name}: {error}"

    async def _run_tool_call(
        self,
        tc: dict[str, Any],
        step: int,
        tok: TokenUsage,
        auth_context: AuthContextProtocol | None = None,
    ) -> dict[str, Any]:
        # unpack the LLM's tool call request
        fn_name = tc["function"]["name"]
        try:
            fn_args = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError as exc:
            error = f"Malformed tool arguments from model: {exc}"
            await self._emit(
                self._step_event(
                    "tool_error", {"tool": fn_name, "error": error}, step=step + 1, tokens=tok
                )
            )
            return {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": f"Error: {error}",
            }
        # announce the action before executing so the event fires even if the tool raises
        await self._emit(
            self._step_event(
                "action",
                {"tool": fn_name, "args": fn_args},
                step=step + 1,
                tokens=tok,
            )
        )
        observation = await self._execute_tool(fn_name, fn_args, auth_context)
        await self._emit(
            self._step_event(
                "observation",
                {"tool": fn_name, "content": observation},
                step=step + 1,
                tokens=tok,
            )
        )
        # return in the format the OpenAI messages API expects for tool results
        return {
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": observation,
        }

    # --- ReAct loop helpers ---

    def _apply_auth_filter(
        self,
        schemas: list[dict[str, object]],
        context: AuthContextProtocol | None,
    ) -> list[dict[str, object]]:
        if self.auth_gate:
            return self.auth_gate.filter_schemas(schemas, context)
        return schemas

    async def _apply_input_guards(self, prompt: str) -> str:
        for guard in self.input_guards:
            name = getattr(guard, "__name__", type(guard).__name__)
            try:
                result = guard(prompt)
                if asyncio.iscoroutine(result):
                    result = await result
                prompt = result
                await self._emit(self._step_event("guardrail", {"scope": "input", "guard": name, "verdict": "pass"}))
            except Exception as exc:
                await self._emit(self._step_event("guardrail", {"scope": "input", "guard": name, "verdict": "block", "reason": str(exc)}))
                raise
        return prompt

    async def _apply_output_guards(self, content: str) -> str:
        for guard in self.output_guards:
            name = getattr(guard, "__name__", type(guard).__name__)
            try:
                result = guard(content)
                if asyncio.iscoroutine(result):
                    result = await result
                content = result
                await self._emit(self._step_event("guardrail", {"scope": "output", "guard": name, "verdict": "pass"}))
            except Exception as exc:
                await self._emit(self._step_event("guardrail", {"scope": "output", "guard": name, "verdict": "block", "reason": str(exc)}))
                raise
        return content

    async def _run_loop(
        self,
        messages: list[dict[str, Any]],
        auth_context: AuthContextProtocol | None,
    ) -> dict[str, Any]:
        tok = TokenUsage()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_reasoning_tokens = 0
        context_tokens = 0

        step = 0
        while step < self.max_steps:
            if self.before_iteration_callback:
                cb_result = self.before_iteration_callback(step, self.max_steps)
                if asyncio.iscoroutine(cb_result):
                    await cb_result
            visible_schemas = self._apply_auth_filter(self.schemas, auth_context)

            response = await self.client.chat_completions(
                messages,
                tools=visible_schemas or None,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                max_retries=self.max_retries,
            )

            usage = response.get("usage", {})
            context_tokens = usage.get("prompt_tokens", 0)
            total_prompt_tokens += context_tokens
            total_completion_tokens += usage.get("completion_tokens", 0)
            total_reasoning_tokens += usage.get("completion_tokens_details", {}).get(
                "reasoning_tokens", 0
            )
            tok = TokenUsage(
                context_tokens=context_tokens,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                reasoning_tokens=total_reasoning_tokens,
            )

            msg: dict[str, Any] = response["choices"][0]["message"]
            history_msg = {
                k: v
                for k, v in msg.items()
                if k not in ("reasoning_content", "thought", "reasoning")
            }
            messages.append(history_msg)

            reasoning = (
                msg.get("reasoning_content")
                or msg.get("thought")
                or msg.get("reasoning")
            )
            if reasoning:
                await self._emit(
                    self._step_event(
                        "thought",
                        {"kind": "reasoning", "content": reasoning.strip()},
                        step=step + 1,
                        tokens=tok,
                    )
                )

            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls")

            if content and tool_calls:
                await self._emit(
                    self._step_event(
                        "thought",
                        {"kind": "plan", "content": content.strip()},
                        step=step + 1,
                        tokens=tok,
                    )
                )

            if not tool_calls:
                return {
                    "answer": content,
                    "messages": messages,
                    "context_tokens": context_tokens,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "reasoning_tokens": total_reasoning_tokens,
                    "total_billable_tokens": total_prompt_tokens
                    + total_completion_tokens,
                }

            tool_messages: list[dict[str, Any]] = list(
                await asyncio.gather(
                    *[
                        self._run_tool_call(tc, step, tok, auth_context)
                        for tc in tool_calls
                    ]
                )
            )
            messages.extend(tool_messages)
            step += 1

        await self._emit(
            self._step_event("error", {"reason": "max_steps_reached"}, tokens=tok)
        )
        return {
            "answer": "(reached maximum steps without a final answer)",
            "messages": messages,
            "context_tokens": context_tokens,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "reasoning_tokens": total_reasoning_tokens,
            "total_billable_tokens": total_prompt_tokens + total_completion_tokens,
        }

    # --- ReAct loop ---

    async def run(
        self,
        prompt: str,
        system_prompt: str | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        auth_context: AuthContextProtocol | None = None,
        delegated_to: str | None = None,
    ) -> dict[str, Any]:
        """Run the agent on a prompt and return the final answer with cumulative token usage.

        Args:
            prompt: The user's task or question.
            system_prompt: Override the agent's default persona for this call only.
            prior_messages: Previous conversation turns to prepend; use with `HistoryBuffer` for multi-turn conversations.
            auth_context: An `AuthContext` identifying the caller. When combined with `auth_gate`, unauthorized tools are hidden from the LLM and rejected at execution time.
            delegated_to: Set by the `Orchestrator` to tag this run as initiated by a parent agent. The tag appears in every emitted event payload (and in the global observability hook) so observers can correlate sub-agent activity with the delegating agent.

        Returns:
            Dict with keys:
            - `answer` (str) — the final answer
            - `messages` (list) — full ReAct-loop message history
            - `context_tokens` (int) — size of the most recent request
            - `prompt_tokens` (int) — cumulative input tokens across all iterations
            - `completion_tokens` (int) — cumulative output tokens, INCLUDING reasoning (OpenAI-compatible)
            - `reasoning_tokens` (int) — subset of `completion_tokens`; do NOT add to the other fields
            - `total_billable_tokens` (int) — `prompt + completion`; the actual amount billed
        """
        system_prompt = system_prompt or self.system_prompt
        prompt = await self._apply_input_guards(prompt)

        messages: list[dict[str, Any]] = (
            [{"role": "system", "content": system_prompt}]
            + (prior_messages or [])
            + [{"role": "user", "content": prompt}]
        )

        # tag every event this run emits with delegated_to (cleared on exit)
        self._current_delegated_to = delegated_to
        try:
            await self._emit(self._step_event("task", {"prompt": prompt}))
            result = await self._run_loop(messages, auth_context)

            answer = str(result.get("answer", ""))
            answer = await self._apply_output_guards(answer)
            result["answer"] = answer

            tok_for_emit = TokenUsage(
                context_tokens=int(result.get("context_tokens", 0)),
                prompt_tokens=int(result.get("prompt_tokens", 0)),
                completion_tokens=int(result.get("completion_tokens", 0)),
                reasoning_tokens=int(result.get("reasoning_tokens", 0)),
            )
            await self._emit(
                self._step_event("answer", {"content": answer}, tokens=tok_for_emit)
            )
            return result
        finally:
            self._current_delegated_to = None
