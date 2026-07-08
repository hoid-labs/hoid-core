# Installation

This page covers practical installation paths for different workflows.

Use source install if you are developing in this repository.
Use release install if you want to consume a published version from PyPI.

## Choose an installation path

- **Source install (recommended for contributors)**: best for local development, examples, and tests.
- **Release install (PyPI)**: best for consumers who want a pinned published version.

## Source install (recommended for this repo)

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

Install optional capabilities only when needed:

```bash
uv pip install -e ".[mcp]"      # fastapi (HTTP transport for MCP servers)
uv pip install -e ".[rag]"      # pypdf, semantic-text-splitter, sqlite-vec (RAG with sqlite backend)
uv pip install -e ".[qdrant]"   # qdrant-client (Qdrant vector backend; set VECTOR_BACKEND=qdrant)
uv pip install -e ".[oidc]"     # PyJWT[crypto] (OIDC Authorization Code flow)
uv pip install -e ".[std]"      # rag + oidc; recommended full install
```

This keeps dependencies explicit and aligned with the framework's low-footprint philosophy.

| Extra | Adds |
|---|---|
| *(none)* | `httpx`, `python-dotenv` |
| `[mcp]` | `fastapi` — HTTP transport for MCP servers |
| `[rag]` | `pypdf`, `semantic-text-splitter`, `sqlite-vec` |
| `[oidc]` | `PyJWT[crypto]` |
| `[std]` | `rag` + `oidc` — recommended full install |
| `[qdrant]` | `qdrant-client` — Qdrant vector backend; requires `VECTOR_BACKEND=qdrant`; install alongside `[rag]` |

## Release install (PyPI)

`hoid` is published to PyPI. Install with `uv pip` or `pip`:

```bash
pip install hoid              # base install (httpx, python-dotenv, defusedxml)
pip install "hoid[std]"       # recommended full install (rag + oidc)
```

Pull only what you need to keep your dependency footprint small:

```bash
pip install "hoid[mcp]"        # fastapi (HTTP transport for MCP servers)
pip install "hoid[rag]"        # pypdf, semantic-text-splitter, sqlite-vec
pip install "hoid[qdrant]"     # qdrant-client (Qdrant backend; set VECTOR_BACKEND=qdrant)
pip install "hoid[oidc]"       # PyJWT[crypto] (OIDC Authorization Code flow)
```

After installation, continue with the Quickstart to verify your environment and run a first agent.