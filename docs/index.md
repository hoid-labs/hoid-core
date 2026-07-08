# hoid

Minimal Python library for building LLM-powered agents with a low dependency footprint.

- **`core/`** needs only `httpx` and `python-dotenv`. Always available.
- **`extensions/`** are optional extras. Pull only what you need.
- Every feature targets the "goldilocks" zone: use a library where it saves meaningful complexity, write from scratch where it adds unnecessary dependencies.

---

## Where to go next

* **[Installation](getting-started/installation.md)**: Learn how to install the core library and its optional extensions.
* **[Quickstart](getting-started/quickstart.md)**: Build your first agent in less than 15 lines of code.
* **[Patterns](patterns.md)**: Architecture guidance for choosing `examples/tools/` or `examples/mcp_servers/`.

---

## API Reference Overview

- **[Core](api/core.md)** — `LLMClient`, `Agent`, `Orchestrator`, `HistoryBuffer`, `@tool`, `@cached_tool`
- **[Extensions](api/extensions.md)** — `MCPClient`, `MCPManager`, `MCPServer`, `MCPContext`, `MemoryStore`, `RAGStore`, auth gate, `OIDCAuthProvider`, guardrails, vector stores
- **[Configuration](api/config.md)** — `Config` dataclass

Reference tool and MCP-server implementations live under [`examples/`](https://github.com/hoid-labs/hoid-core/tree/main/examples).