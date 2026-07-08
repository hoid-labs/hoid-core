# Core

The core module contains the runtime primitives you use in almost every project.
It is intentionally small and dependency-light, so it can be the stable foundation for both simple scripts and larger multi-agent systems.

If you are unsure where to start, start in core: create an `LLMClient`, build an `Agent`, and add tools only as needed.

## LLMClient

`LLMClient` is the provider-agnostic HTTP client for chat completions and embeddings.
Use it directly when you need low-level model calls, or pass it to `Agent` and `Orchestrator` for higher-level workflows.

### ::: hoid.core.llm.LLMClient

---

## Agent

`Agent` implements the ReAct loop: reason, call tools, observe, and iterate to a final answer.
It is the main entry point for tool-using assistants and supports approvals, guardrails, and auth-aware tool filtering.

### ::: hoid.core.agent.Agent

---

## Orchestrator

`Orchestrator` is a supervisor/delegate layer for multi-agent task routing.
Use it when one prompt should be split across specialized agents with different prompts, tools, or responsibilities.

### ::: hoid.core.orchestrator.Orchestrator

---

## HistoryBuffer

`HistoryBuffer` maintains rolling multi-turn context with token-budget trimming.
Use it to keep conversations coherent while controlling prompt growth and cost.

### ::: hoid.core.history.HistoryBuffer

---

## Observability

The observability module is a small, dependency-free vocabulary for emitting and routing structured events from any call site.
No I/O happens here — consumers register a hook and decide where events go (storage, format, transport are all up to the consumer).

The pattern is: `emit(LLMCallEvent(...))` (or any of the typed event dataclasses) from a call site, and the global hook receives it. With no hook registered, `emit` is a no-op, so adding events never costs anything for users who don't subscribe.

```python
from hoid.core.observability import set_hook, print_hook

set_hook(print_hook())  # events now print to stdout
```

### ::: hoid.core.observability

---

## Decorators

The decorators define how Python callables become model-callable tools.
Use `@tool` for schema generation from type hints and docstrings, and `@cached_tool` when deterministic tool calls should reuse prior results.

### ::: hoid.core.tools.tool

### ::: hoid.core.tools.cached_tool
