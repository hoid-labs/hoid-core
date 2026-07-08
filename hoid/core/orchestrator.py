from __future__ import annotations

import contextlib
from typing import Any

from hoid.core.observability import OrchestratorEvent, emit

from .agent import Agent
from .history import HistoryBuffer
from .protocols import AuthContextProtocol, LLMClientProtocol
from .tools import tool


class Orchestrator:
    "Supervisor agent that delegates tasks to named sub-agents via an injected delegate() tool."

    # --- construction ---

    def __init__(
        self,
        client: LLMClientProtocol,
        sub_agents: dict[str, Agent],
        max_steps: int = 10,
        max_tokens: int = 4096,
        max_retries: int = 3,
        history_max_tokens: int = 8000,
    ) -> None:
        """
        Args:
            client: An `LLMClient` instance for the supervisor agent.
            sub_agents: Mapping of agent name to `Agent` instance. Names are exposed to the supervisor as delegation targets.
            max_steps: Maximum supervisor reasoning iterations (default 10).
            max_tokens: Maximum tokens per supervisor response (default 4096).
            max_retries: Retry attempts on transient API errors forwarded to the supervisor agent (default 3).
            history_max_tokens: Token budget for the supervisor history buffer (default 8000).
        """
        agents_list = ", ".join(sub_agents.keys())

        # binds sub_agents in closure without module-level state
        @tool
        async def delegate(agent_name: str, task: str) -> str:
            "Delegate a task to a named sub-agent and return its answer."
            if agent_name not in sub_agents:
                return f"Unknown agent '{agent_name}'. Available: {agents_list}"
            agent = sub_agents[agent_name]
            with contextlib.suppress(Exception):
                await emit(
                    OrchestratorEvent(
                        event_type="delegated",
                        sub_agent_id=agent_name,
                        payload={"task_preview": task[:200]},
                    )
                )
            # pass delegated_to so the sub-agent tags its events with the orchestrator as caller
            result = await agent.run(task, delegated_to=f"orchestrator.{agent_name}")
            with contextlib.suppress(Exception):
                await emit(
                    OrchestratorEvent(
                        event_type="delegation_finished",
                        sub_agent_id=agent_name,
                        payload={
                            "answer_preview": (result or {}).get("answer", "")[:200]
                        },
                    )
                )
            return str(result.get("answer", "(no answer)"))

        self._system_prompt = (
            f"You are a supervisor. Delegate tasks to sub-agents as needed. "
            f"Available agents: {agents_list}."
        )
        self._agent = Agent(
            client,
            tools=[delegate],
            max_steps=max_steps,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        # retains supervisor context across multiple run() calls
        self._history = HistoryBuffer(max_tokens=history_max_tokens)

    async def run(
        self,
        prompt: str,
        system_prompt: str | None = None,
        auth_context: AuthContextProtocol | None = None,
    ) -> dict[str, Any]:
        with contextlib.suppress(Exception):
            await emit(
                OrchestratorEvent(
                    event_type="supervisor_started",
                    payload={"prompt_preview": prompt[:200]},
                )
            )
        try:
            result = await self._history.run(
                self._agent,
                prompt,
                system_prompt=system_prompt or self._system_prompt,
                auth_context=auth_context,
            )
        except Exception as exc:
            with contextlib.suppress(Exception):
                await emit(
                    OrchestratorEvent(
                        event_type="supervisor_errored",
                        payload={"error": str(exc)[:200]},
                    )
                )
            raise
        with contextlib.suppress(Exception):
            await emit(
                OrchestratorEvent(
                    event_type="supervisor_finished",
                    payload={"answer_preview": (result or {}).get("answer", "")[:200]},
                )
            )
        return result
