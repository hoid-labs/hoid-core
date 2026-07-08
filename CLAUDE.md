# CLAUDE.md

Instructions for AI agents (Claude Code, GitHub Copilot, Cursor, etc.) working in this repository. Single source of truth — keep it current.

## Project

Minimal Python library for building LLM-powered agents. Wrapped by [`hoid-studio`](https://github.com/hoid-labs/hoid-studio) (FastAPI + Svelte 5) and consumed by plugin packages via the `hoid.plugins` entry-point group. Priority: low dependency footprint and supply chain safety.

- `hoid/` — the library
  - `_env.py` — `load_env()` helper; reads `.env` from cwd
  - `_optional.py` — `EXTRAS_MAP` + `require()` for optional-dep sentinels
  - `core/` — httpx + dotenv + defusedxml only; always available
  - `extensions/` — optional; install only what you need via extras
- `tests/` — `unit/`, `integration/`, `test_packaging.py`
- `examples/` — runnable demos (`basics/`, `chats/`, `specialized_agents/`, plus `tools/` and `mcp_servers/` reference implementations)
- `tools/diagnose.sh` — diagnostic toolchain (ruff, bandit, radon, mypy, deptry, pytest-cov, pydeps)
- `docs/` — mkdocs site (`docs/api/`, `docs/patterns.md`, `docs/environment-variables.md`)

This file is the upstream source for sibling repos. When core's rules change, `hoid-studio/CLAUDE.md` and `hoid-plugins/CLAUDE.md` update to match.

## Quick Start

```sh
uv venv && source .venv/bin/activate
uv pip install -e ".[std]"            # core + rag + oidc (recommended)
uv pip install -e ".[mcp]"            # + fastapi (MCP server HTTP transport)
uv pip install -e ".[qdrant]"         # + qdrant-client (set VECTOR_BACKEND=qdrant)
uv sync --all-extras --group dev      # full dev environment
```

| Extra | Adds |
|---|---|
| *(none)* | `httpx`, `python-dotenv`, `defusedxml` |
| `[mcp]` | `fastapi` |
| `[rag]` | `pypdf`, `semantic-text-splitter`, `sqlite-vec` |
| `[oidc]` | `PyJWT[crypto]` |
| `[qdrant]` | `qdrant-client` (pair with `[rag]`) |
| `[std]` | `[rag]` + `[oidc]` (recommended full install) |

Environment: copy `.env.example` to `.env`. Required: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`. Optional: `CA_BUNDLE_PATH`, `VECTOR_BACKEND`, `SQLITE_PATH`, `QDRANT_*`, `MEMORY_PATH`, `OIDC_*`.

The reference MCP servers (`knowledge-server`, `memory-server`) live in `examples/`. Run from inside `examples/`:

```sh
cd examples && uv sync
uv run knowledge-server --http --port 8082
uv run memory-server --http --port 8083
```

## Tests

```sh
uv run pytest tests/unit tests/test_packaging.py -v                  # no LLM required
uv run pytest tests/unit/test_agent.py::test_name -v                 # single test
uv run pytest tests/integration -v -m integration                    # live LLM endpoint required
```

## Diagnostics

```sh
uv run ruff check hoid tests                                # lint
uv run ruff check --fix hoid tests                          # lint + autofix
uv run mypy hoid                                            # type-check (library only)
uv run bandit -r hoid -ll                                   # security scan
uv run deptry hoid tests                                    # unused/missing deps
bash tools/diagnose.sh                                               # full toolchain
```

## Architecture

Three layers with one-way imports (`core` never imports from `extensions`). The public API is what `hoid.core.__init__` re-exports (`LLMClient`, `Agent`, `Orchestrator`, `HistoryBuffer`, `tool`, `cached_tool`); everything else is reached via explicit submodule paths.

### `hoid/core/`

httpx + dotenv + defusedxml only. Always importable.

- `llm.py` — async HTTP client for OpenAI-compatible chat + embeddings
- `observability.py` — events, hook, `ContextVar`; `emit()` is a no-op when no hook is registered
- `tools.py` — `@tool` decorator (type hints + docstring → JSON schema); `@cached_tool` LRU variant
- `agent.py` — ReAct loop: chat → tool calls → observations → repeat
- `history.py` — `HistoryBuffer`: rolling multi-turn history with token-budget trimming
- `orchestrator.py` — multi-agent supervisor / delegate pattern
- `protocols.py` — structural `Protocol`s; lets `core` declare shapes without importing `extensions`

### `hoid/extensions/`

Optional; gated by pip extras. Never imports from `core/__init__.py` — targets submodules directly.

- `memory.py` — stdlib JSON-backed key-value store (zero new deps; single-process only)
- `auth/` — `AuthContext`, `AuthGate`, `PolicyBackend`, `FilePolicyBackend`, `MemoryPolicyBackend`, OIDC provider
- `mcp/` — `MCPClient` (stdio + streamable HTTP), `MCPManager`, `MCPServer`, `MCPContext`
- `rag/` — `RAGStore`, `BaseStorageBackend` protocol; sqlite-vec and Qdrant backends
- `guardrails.py` — composable input/output guard functions (zero new deps)

### `examples/tools/`

`@tool`-decorated reference implementations shipped as runnable examples (filesystem, web_fetch, shell, memory, builtins). Import via `examples.tools.X` after `cd examples && uv sync`.

### `examples/mcp_servers/`

`@mcp.tool()` servers; entry-point scripts. Run from inside `examples/`:

- `knowledge_server.py` — RAG-backed search (`uv run knowledge-server`)
- `memory_server.py` — key-value store (`uv run memory-server`)

For API reference: `docs/api/`. For usage patterns: `docs/patterns.md`. For environment variables: `docs/environment-variables.md` (auto-rendered from `.env.example`).

## Coding Rules

These are non-negotiable. See `.github/copilot-instructions.md` for the parallel "strict Python standards" set.

**Comments.** Single-line only. Explain *why*, not *what*. No numbered comments, no emojis.

**Naming.** snake_case for functions/variables, PascalCase for classes, UPPER_CASE for module-level constants. Names must be provider-/backend-agnostic — no hardcoded service/library names in comments or variable names.

**Type hints.** 100% annotations on all signatures and class attributes. `from __future__ import annotations` at the top of every module. Prefer `T | None` (PEP 604). `Any` only at protocol boundaries. Banned: mutable default arguments. Banned: `getattr`/`hasattr` for inter-module contracts — use `Protocol` or explicit type checks.

**Docstrings.** Required on `@tool` / `@mcp.tool()` functions (used for schema auto-generation): one-line summary + `Args:` block per parameter. Full docstrings on public service methods and Protocol interface methods when side effects / return shape / contract are not obvious from the name. Include `Returns:` and `Raises:` when non-trivial.

**Dependencies.** Never edit `pyproject.toml` by hand. Use `uv add`: `uv add <pkg>`, `uv add --optional <extra-name> <pkg>` (one named extra per feature — never bundle unrelated deps), `uv add --group dev <pkg>`. For upgrades: `uv add --upgrade <pkg>`.

**Architecture.** Strict DAG imports: `core` must not import from `extensions` at runtime; use `Protocol` + `TYPE_CHECKING` for type-only references. Submodules import from explicit submodule paths, never from the top-level package `__init__.py` (prevents recursive execution loops). Catch specific exceptions — `except Exception:` is a code smell. Protocol methods must be fully implemented; never stub with `return None`.

**Optional deps.** For *third-party* optional packages, use the three-step sentinel pattern: (1) add `"pkg": "extra_name"` to `EXTRAS_MAP` in `hoid/_optional.py`, (2) declare a top-level `try: import pkg / except ImportError: pkg = None  # type: ignore[assignment]`, (3) call `_require("pkg", pkg)` at instantiation or first use (not at import time). Never use lazy imports inside function bodies — the sentinel pattern defers the failure to the call site with an install hint.

**Consistency.** Any pattern introduced in one file of a group (`examples/tools/`, `examples/mcp_servers/`, `extensions/*/`) must be applied to all equivalent files in that group immediately — error surfacing, `__main__` CLI blocks, logging, `_MAX_CHARS` truncation, etc. Pareto principle: handle the 80% case cleanly.

## Key Patterns

**`@tool` decorator.** Type hints + docstring auto-build the JSON schema. Supports sync and async. Use `@cached_tool(maxsize=128)` for LRU caching.

```python
from hoid.core import tool

@tool
def search_notes(query: str, limit: int = 5) -> str:
    """Search the knowledge base for relevant chunks.

    Args:
        query: Description shown to the model.
        limit: Max results to return.
    """
    ...
```

**Optional-dep sentinel.** Add the extra to `pyproject.toml`:

```toml
[project.optional-dependencies]
my_feature = ["my_pkg>=1.0"]
```

**Structural Protocol for cross-layer contracts.** When `core` needs to type-hint a collaborator from `extensions`:

```python
# in core/protocols.py
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from hoid.extensions.my_feature import MyService

class MyServiceProtocol(Protocol):
    async def do_thing(self, x: str) -> dict: ...
```

The concrete class in `extensions/` satisfies the Protocol structurally — no inheritance required.

**Observability event.** Every meaningful call site emits one. No-op when no hook is registered.

```python
from hoid.observability import emit, MyEvent

await emit(MyEvent(event_type="started", payload={"x": 42}))
```

The global hook is opt-in. Observability is the right place for cross-cutting concerns; the per-instance `on_step` callback is for tight integration with a specific agent.

**Public API surface.** `from hoid.core import X` re-exports only the six names listed in `core/__init__.py`. New public symbols in `core/` need to be added to that `__all__` list to be reachable via the top-level import. Extensions are re-exported from `extensions/__init__.py` (see `MemoryStore`, `RAGStore`, `AuthGate`, `MCPClient`, etc.).

## Anti-patterns

Curated list of mistakes that have actually happened in this repo. Don't replicate.

- **Demo functions live in `examples/`, not the library.** The library ships `@tool` decorators and infrastructure; concrete tool implementations are reference examples.
- **Don't reach into another module's internals** to inject state (e.g. `Orchestrator` mutating a sub-agent's `on_event`). Use the public API or the observability hook.
- **Don't use lazy imports** inside function bodies. Use the sentinel pattern.
- **Don't double-count tokens.** Reasoning tokens are a *subset* of completion tokens, not additive.
- **Don't bake IdP-specific fallbacks** into generic OIDC code. Either commit to a specific IdP (and say so) or remove the fallbacks.
- **Don't `print()` for observability.** Use the observability hook or stdlib logging.
- **Don't catch `Exception`** to silence errors. Catch the specific type you expect.
- **Don't add new top-level package files** without updating the Architecture map above.

## How to Add Things

### A tool

1. Create or edit a file in `examples/tools/` (or `examples/basics/` if it's a demo).
2. `from hoid.core import tool`; decorate with `@tool`.
3. One-line docstring + `Args:` block per parameter. Re-export in `examples/tools/__init__.py`.

### An extension

1. Create `hoid/extensions/my_feature.py` (single file) or `extensions/my_feature/` (package).
2. Zero new deps → import unconditionally in `extensions/__init__.py`.
3. Has optional deps → use the sentinel pattern. Add to `_optional.EXTRAS_MAP`. Declare a new named extra in `pyproject.toml`.

### An MCP server (exposing tools to any MCP client)

1. Create `examples/mcp_servers/my_server.py`.
2. Define a `lifespan` async context manager that yields shared clients/resources.
3. Decorate tools with `@mcp.tool()`. Access shared state via `ctx: MCPContext` parameter.
4. Add `argparse` CLI with `--http/--port` flags.
5. Add an entry point in `examples/pyproject.toml [project.scripts]`.
6. Connect from agents with `MCPClient.stdio("uv", ["run", "my-server"])` (run from inside `examples/`) or `MCPClient.http(url)`.

### An agent server (agent-as-a-tool)

Contract: every agent server exposes exactly one tool named `delegate_to_{name}(task: str) -> str`. Uniform interface, unique name when aggregated. Place in `examples/specialized_agents/my_agent.py`; `lifespan` builds the `LLMClient` and any tool list; `logging.FileHandler` captures ReAct traces to `{name}.log`; callers connect with `MCPClient.http(url, timeout=300.0)`.

### A vector backend

Create `extensions/rag/vector_store/my_backend.py` implementing `async def upsert(...)` and `async def search(...)`. No base class — `BaseStorageBackend` is a structural Protocol. Add a branch to `backend_from_env()` in `extensions/rag/__init__.py`. Add a pyproject extra if the backend needs a new package.

### An auth provider

Implement `async def resolve(self, credentials: dict) -> AuthContext | None`. No base class — `AuthProvider` is a structural Protocol. `credentials` is transport-agnostic. Return `None` when credentials are unrecognized.

## Testing Patterns

- `pytest-asyncio` is in auto mode — async test functions run without `@pytest.mark.asyncio`.
- `tests/conftest.py` provides shared fixtures:
  - `mock_llm` — factory for `MockLLMClient`: scriptable fake that returns canned responses in order. Accepts a list of strings (final answers) or raw response dicts.
  - `recording_hook` — installs a global `RecordingHook` that collects all observability events into `.events` for assertion.
- Unit tests must not require a live LLM. Use `MockLLMClient` to script tool-call sequences.

## Pre-PR Checklist

1. `uv run ruff check hoid tests` — clean
2. `uv run mypy hoid` — clean
3. `uv run pytest tests/unit tests/test_packaging.py` — all green
4. `tools/diagnose.sh` — clean (or document any intentional warning)
5. New env vars have `.env.example` entries in the exact format `# [Required|Optional] One-line description.` on the line above
6. New top-level files under `hoid/` are reflected in the Architecture map above
7. New MCP/agent servers under `README.md` ("MCP Tools Servers" / "Agent Servers" tables)
8. `CHANGELOG.md` has an `[Unreleased]` entry under the right section (`Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` / `Security`)

## Commit Format

See the `/propose-commits` command for the full format rules, types list, body shape, and worked examples. It lives at `.claude/commands/propose-commits.md` in the workspace root — invoke it as `/propose-commits <core|studio|plugins>` when composing any commit message.

Workflow reminders (not in the skill — always true):

- Before writing the message, run `git diff HEAD` (not just `git status --short`) so the bullets come from the actual diff, not from memory. `git status` only lists file names; it is not enough.
- Never commit or push without explicit user instruction. Never amend already-pushed commits. Propose both the `git add` and the `git commit` commands — never just the message.
- When the user asks for commits, output ready-to-copy `git add <files> && git commit -m "..."` blocks.

## Core-specific Rules

- `ruff` line-length 88; rules `E/F/W/I/UP/B/SIM`; `E501` ignored.
- `mypy` is strict on `hoid/` only (tests are exercised by `pytest`).
- Optional deps used inside the library must be declared as sentinels — never imported at function-body level.
- `core/` re-exports exactly six symbols from `__init__.py` (`LLMClient`, `Agent`, `Orchestrator`, `HistoryBuffer`, `tool`, `cached_tool`). Add new public symbols there deliberately — it is the contract with downstream consumers.
- `extensions/__init__.py` re-exports the public surface of each optional subsystem. New extension classes get added there; otherwise they are reachable only via explicit submodule paths.
- Do not commit `.env`, `*.log`, `data/`. `.gitignore` covers these.

## Release flow

Every push to `main` runs `.github/workflows/auto-tag.yaml`, which computes the next version, creates the tag via the GitHub API, and dispatches a `release` repository event. The dispatch event triggers `release.yaml`, which builds the wheel + sdist via `uv build` and attaches them to a GitHub Release (no PyPI publish). Release notes are auto-generated from commit subjects by `softprops/action-gh-release@v2`.

The version bump follows conventional commits:
- `[release: major]` / `[release: minor]` / `[release: patch]` footers override auto-detection.
- Subjects ending in `!:` or bodies containing `BREAKING CHANGE:` trigger a major bump.
- `feat:` subjects trigger a minor bump.
- Anything else triggers a patch bump.

Add `[skip release]` or `[skip ci]` to any commit body to suppress tagging for that push. To ship a specific version out-of-band, push the tag manually: `git tag vX.Y.Z && git push origin vX.Y.Z` — the `push: tags: v*` trigger on `release.yaml` still handles this path. The wheel version is read from the tag at build time by `hatch-vcs` (`pyproject.toml` `[tool.hatch.version] source = "vcs"`).

## Open Work (v1.0 gate)

Items from the private review notes (`issues/REVIEW_*.md`, gitignored) that should be tackled before v1.0:

1. **Decide on OIDC IdP coupling** — either commit to a specific IdP and rename, or remove the `v1` and `appid` fallbacks.
2. **Add CI for tests + lint + types** — currently only docs deploy runs on push.
3. **`MemoryStore` concurrency** — JSON-as-a-DB has no multi-process safety. Document the limit or swap to SQLite.
4. **`RAGStore` hardcoded home-dir sandbox** — make the root a constructor argument.

For the full list, see the private review notes (in `issues/REVIEW_*.md`, gitignored).
