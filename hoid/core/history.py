from __future__ import annotations

from typing import Any

from .protocols import AgentRunProtocol


class HistoryBuffer:
    "Rolling message buffer that preserves the system message while trimming oldest history."

    def __init__(self, max_messages: int | None = None, max_tokens: int | None = None) -> None:
        """
        Args:
            max_messages: Maximum number of messages to retain. Oldest action/observation pairs are evicted first.
            max_tokens: Soft token budget (char÷4 approximation with a 20% safety margin). The buffer trims to 80% of this value.
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self._messages: list[dict] = []

    @property
    def messages(self) -> list[dict]:
        "Current buffered messages after trimming."
        return list(self._messages)

    def extend(self, messages: list[dict[str, Any]]) -> None:
        "Append messages (from an agent.run() result) and trim to configured limits."
        # system message is re-added by the agent on each run()
        non_system = [m for m in messages if m.get("role") != "system"]
        self._messages.extend(non_system)
        self._trim()

    def _trim(self) -> None:
        if self.max_messages is not None:
            while len(self._messages) > self.max_messages:
                self._evict_oldest_pair()
        if self.max_tokens is not None:
            # 20% margin absorbs char÷4 estimation error without an external tokenizer
            while (
                self._token_estimate() > int(self.max_tokens * 0.8) and self._messages
            ):
                self._evict_oldest_pair()

    def _evict_oldest_pair(self) -> None:
        # evict the full action/observation group so history stays coherent
        for i, m in enumerate(self._messages):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                self._messages.pop(i)
                # drop all tool results belonging to this action
                while (
                    i < len(self._messages) and self._messages[i].get("role") == "tool"
                ):
                    self._messages.pop(i)
                return
        # no tool call — evict the oldest user/assistant pair so Q&A history stays coherent
        for i, m in enumerate(self._messages):
            if m.get("role") == "user":
                self._messages.pop(i)
                if (
                    i < len(self._messages)
                    and self._messages[i].get("role") == "assistant"
                ):
                    self._messages.pop(i)
                return
        if self._messages:
            self._messages.pop(0)

    def _token_estimate(self) -> int:
        total = 0
        for m in self._messages:
            # role string + ~4 tokens of JSON message envelope overhead
            total += len(m.get("role", "")) // 4 + 4
            total += len(m.get("content") or "") // 4
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                # id field is sent in every tool call message
                total += len(tc.get("id", "")) // 4
                total += len(fn.get("name", "")) // 4
                total += len(fn.get("arguments", "")) // 4
            # tool result messages reference their originating call id
            if m.get("role") == "tool":
                total += len(m.get("tool_call_id", "")) // 4
        return total

    def get(self) -> list[dict[str, Any]]:
        "Return a copy of the current message history for use as prior_messages."
        return list(self._messages)

    async def run(self, agent: AgentRunProtocol, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Run the agent with buffered history, then extend the buffer automatically.

        Args:
            agent: The `Agent` instance to run.
            prompt: The user's task or question.
            **kwargs: Forwarded to `agent.run()` (e.g. `system_prompt`).

        Returns:
            The agent's result dict (same shape as `agent.run()`).
        """
        result = await agent.run(prompt, prior_messages=self.get(), **kwargs)
        self.extend(result["messages"])
        return result

    def clear(self) -> None:
        "Reset the buffer."
        self._messages = []
