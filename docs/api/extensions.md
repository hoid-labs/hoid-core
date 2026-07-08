# Extensions

Extensions are optional modules that add capabilities beyond the core runtime while keeping dependencies explicit.
Use extensions when you need integration features like MCP connectivity, RAG storage, policy-based auth, or output safeguards.
Install the `[std]` extra (`uv pip install 'hoid[std]'`) for the recommended full-featured set (`[rag]` + `[oidc]`).

## MCP

MCP extensions connect your agents to out-of-process tool servers over stdio or streamable HTTP.
Use `MCPClient`/`MCPManager` to consume tools from any MCP server. Use `MCPServer`/`MCPContext` to build and expose your own.

!!! note
    `MCPClient` and `MCPServer` stdio transport have no extra dependencies. The MCP server HTTP transport requires the `[mcp]` extra: `uv pip install 'hoid[mcp]'`.

### ::: hoid.extensions.mcp.MCPClient

### ::: hoid.extensions.mcp.MCPManager

### ::: hoid.extensions.mcp.MCPServer

### ::: hoid.extensions.mcp.MCPContext

---

## Memory

`MemoryStore` is a lightweight key-value persistence layer for recalled context.
It is useful for user preferences, session facts, and other simple memory patterns without adding heavy dependencies.

### ::: hoid.extensions.memory.MemoryStore

---

## Auth

Auth primitives separate authentication (who the caller is) from authorization (which tools they can use).
Use these types when your agent must enforce role-based or user-based access to tools.
`OIDCAuthProvider` requires the `[oidc]` extra: `uv pip install 'hoid[oidc]'`.

### ::: hoid.extensions.auth.AuthContext

### ::: hoid.extensions.auth.AuthGate

### ::: hoid.extensions.auth.backends.file.FilePolicyBackend

### ::: hoid.extensions.auth.backends.memory.MemoryPolicyBackend

### ::: hoid.extensions.auth.providers.static.StaticAuthProvider

### ::: hoid.extensions.auth.providers.oidc.OIDCAuthProvider

---

## RAG

`RAGStore` handles ingest, chunking, embedding, and semantic retrieval workflows.
Use it when your agent should ground answers in your own documents rather than only model priors.
Requires the `[rag]` extra: `uv pip install 'hoid[rag]'`. For OIDC auth support, add `[oidc]`. Install both via `[std]`.

### ::: hoid.extensions.rag.RAGStore

---

## Vector Store

Vector backends provide the storage/search engine behind RAG retrieval.
`SqliteVecBackend` is the default (no server required, file or in-memory). `QdrantBackend` is available for production scale via the `[qdrant]` extra.
Use `backend_from_env()` to select at runtime via the `VECTOR_BACKEND` env var.

### ::: hoid.extensions.rag.vector_store.sqlite.SqliteVecBackend

### ::: hoid.extensions.rag.vector_store.qdrant.QdrantBackend

---

## Guardrails

Guardrails are composable filters that validate or transform agent input and output.
Use them to block unsafe prompts, redact sensitive output, or enforce policy boundaries.

### ::: hoid.extensions.guardrails.block_keywords

### ::: hoid.extensions.guardrails.strip_pii

### ::: hoid.extensions.guardrails.llm_guard
