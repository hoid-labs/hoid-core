# hoid

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/hoid)](https://pypi.org/project/hoid/)
[![CI](https://github.com/hoid-labs/hoid-core/actions/workflows/ci.yaml/badge.svg)](https://github.com/hoid-labs/hoid-core/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/downloads/)

Minimal Python library for building LLM-powered agents. Designed for a low dependency footprint and supply chain safety.

* `core/` needs only `httpx` and `python-dotenv`. Always available.
* `extensions/` are optional extras. Pull only what you need.
* Every feature targets the "goldilocks" zone: use a library where it saves meaningful complexity, write from scratch where it adds unnecessary dependencies.

## Requirements

* Python 3.13+
* An OpenAI-compatible LLM API (local or remote)

## Install

```bash
git clone <repo> && cd hoid
uv venv && source .venv/bin/activate
uv pip install -e .               # core only: httpx, python-dotenv
uv pip install -e ".[mcp]"        # + fastapi (HTTP transport for MCP servers)
uv pip install -e ".[rag]"        # + pypdf, semantic-text-splitter, sqlite-vec
uv pip install -e ".[qdrant]"     # + qdrant-client (Qdrant vector backend; set VECTOR_BACKEND=qdrant)
uv pip install -e ".[oidc]"       # + PyJWT[crypto] for OIDC Authorization Code flow
uv pip install -e ".[std]"        # rag + oidc; recommended full install
```

| Extra | Adds |
|---|---|
| *(none)* | `httpx`, `python-dotenv` |
| `[mcp]` | `fastapi` — HTTP transport for MCP servers |
| `[rag]` | `pypdf`, `semantic-text-splitter`, `sqlite-vec` |
| `[oidc]` | `PyJWT[crypto]` |
| `[std]` | `rag` + `oidc` — recommended full install |
| `[qdrant]` | `qdrant-client` — Qdrant vector backend; requires `VECTOR_BACKEND=qdrant`; install alongside `[rag]` |

Each extra is an explicit, auditable addition to your supply chain.

## Architecture

```text
core/           — httpx only; ReAct agent loop, @tool schema gen, LLM client, orchestrator
extensions/     — optional: MCP client, RAG, auth, memory store, guardrails
```

Reference tool and MCP-server implementations live in `examples/tools/` and `examples/mcp_servers/` — see [`examples/`](examples/) for runnable demos.

### Three layers for every capability

Each capability (memory, RAG, etc.) follows the same pattern:

**`extensions/` — pure Python class, no agent access.**
The raw implementation. No agent can call it during reasoning — it has no tool schema. Use it in your own code: pre-index documents, load stored memories to inject as context, inspect state after a run.

```python
store = MemoryStore(path="memory.json")
await store.save("user", "Alice")
# inject into prompt manually
system = f"User name: {store.load('user')}"
```

**`examples/tools/` — local `@tool`, in-process.**
Exposes the class as `@tool` functions so the agent can call them autonomously mid-reasoning. Zero network overhead. Use when one agent needs the capability within a single script.

```python
from examples.tools.memory import make_memory_tools

store = MemoryStore(path="memory.json")
agent = Agent(client, tools=make_memory_tools(store))
```

**`examples/mcp_servers/` — MCPServer, out-of-process.**
Run once; every connecting agent shares the same state. Two modes:
- **stdio** — subprocess, lives for the duration of your script
- **HTTP** — persistent process; any number of agents or scripts share the same endpoint

```sh
cd examples && uv run memory-server --http --port 8083 --path memory.json
```

Rule of thumb: one agent, one script → use `examples/tools/`. Shared state or persistence across scripts → use `examples/mcp_servers/`.

## Configuration

The framework reads configuration directly from environment variables via `LLMClient.from_env()`. Copy `.env.example` to `.env`.

**Required variables:**

* `LLM_BASE_URL`
* `LLM_API_KEY`
* `LLM_MODEL`

**Optional variables:**

* `CA_BUNDLE_PATH` — path to custom TLS certs, or `False` to disable verification
* `EMBED_MODEL` — falls back to `LLM_MODEL` if unset
* `VECTOR_BACKEND`, `SQLITE_PATH`, `VECTOR_SIZE`, `QDRANT_*` — for RAG/Knowledge extension (`[rag]` / `[qdrant]` extras)
* `MEMORY_PATH` — for file-backed memory persistence
* `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET`, `OIDC_DISCOVERY_URL` — OIDC Authorization Code flow (`[oidc]` extra)
* `OIDC_ROLES_CLAIM` — JWT claim to read roles from (default: `roles`)

## Usage

### Basic agent with tools

```python
from hoid.core import LLMClient, Agent
from examples.tools.filesystem import list_directory, read_file

async with LLMClient.from_env() as client:
    agent = Agent(
        client=client,
        tools=[list_directory, read_file],
        system_prompt="You are a helpful assistant.",  # stored as default persona
    )
    result = await agent.run("List the files in my Desktop folder.")
    print(result["answer"])
    # Return keys: answer, messages, context_tokens, prompt_tokens, completion_tokens, reasoning_tokens
    # context_tokens = current window size; prompt_tokens = cumulative billing total across all steps
```

The agent is silent by default. Pass an `on_event` callable to receive structured event dicts at each step:

```python
agent = Agent(client=client, tools=tools, on_event=lambda e: print(e))
```

### Agent parameters

| Parameter | Default | Description |
|---|---|---|
| `tools` | `[]` | List of `@tool`-decorated callables |
| `max_steps` | `10` | Maximum ReAct iterations before giving up |
| `max_tokens` | `4096` | Max tokens per LLM response. Increase when the model must reproduce large tool results verbatim |
| `temperature` | `0.7` | Set `0.0` for deterministic tasks like structured extraction or KQL generation |
| `max_retries` | `3` | LLM request retries with exponential backoff on 429/5xx |
| `system_prompt` | built-in | Persistent persona; overridable per-call via `agent.run(..., system_prompt=...)` |
| `input_guards` | `[]` | List of guard callables that transform or block the prompt before the LLM sees it |
| `output_guards` | `[]` | List of guard callables that transform or redact the final answer |
| `on_event` | None (silent) | Callable receiving structured event dicts; `None` for silent |
| `approval_callback` | `None` | Async `(name, args) -> bool`; pauses before each tool call |
| `approval_tools` | all tools | Set of tool names to gate; omit to gate every tool |

### on_event shapes

Every event is a dict with an `"event"` key. Most events also carry token fields (`context_tokens`, `prompt_tokens`, `completion_tokens`, `reasoning_tokens`).

| event | extra fields |
|---|---|
| `task` | `prompt` |
| `thought` | `kind` (`"reasoning"` or `"plan"`), `content` |
| `action` | `step`, `tool`, `args` |
| `observation` | `step`, `tool`, `content` |
| `tool_error` | `tool`, `error` |
| `waiting_for_approval` | `tool`, `args` |
| `answer` | `content` |
| `error` | `reason` |

### Agent with MCP server

MCP (Model Context Protocol) allows you to connect to out-of-process tool servers.

```python
from hoid.extensions import MCPClient, MCPManager

# Connect to one or more MCP servers (HTTP, stdio, or SSE)
# timeout covers the full round-trip when the remote server runs its own ReAct loop
async with MCPManager([MCPClient.http("http://localhost:8080/mcp", timeout=300.0)]) as mcp:
    tools = await mcp.get_all_tools()
    agent = Agent(client=client, tools=tools)
    result = await agent.run("Use the tools provided by the MCP server.")
```

### Multi-agent orchestration

```python
from hoid.core import Orchestrator

orchestrator = Orchestrator(
    client=client,
    sub_agents={"filesystem": fs_agent, "math": math_agent},
    history_max_tokens=8000,  # token budget for supervisor history (default 8000)
    max_retries=3,            # forwarded to the internal supervisor agent
)
result = await orchestrator.run("Count files on Desktop and multiply by 2.")
```

### Approval callbacks (HITL)

Pause execution before sensitive tools run. Scope to specific tool names with `approval_tools`; omit it to gate every tool.

```python
async def ask_user(name: str, args: dict) -> bool:
    print(f"Approve '{name}'? [y/n]")
    return input().strip().lower() == "y"

agent = Agent(
    client, tools,
    approval_callback=ask_user,
    approval_tools=["write_file", "send_email"],  # only these require approval
)
```

### Auth and OIDC

The auth system has two decoupled concerns: **authentication** (who is the caller?) and **authorization** (what tools can they use?).

```python
from hoid.extensions.auth import AuthGate, MemoryPolicyBackend, StaticAuthProvider, AuthContext

# define which tools each role can call
backend = MemoryPolicyBackend({
    "roles": {
        "admin":   {"tools": ["*"]},
        "analyst": {"tools": ["fetch_url"]},
    }
})
gate = AuthGate(backend)
agent = Agent(client, tools=my_tools, auth_gate=gate)

# resolve the caller identity, then pass it to each run()
provider = StaticAuthProvider(api_keys={"sk-xyz": AuthContext(user_id="alice", roles={"analyst"})})
auth_ctx = await provider.resolve({"type": "api_key", "key": api_key})
result = await agent.run(prompt, auth_context=auth_ctx)
```

**OIDC Authorization Code flow** (`[oidc]` extra required):

```python
from hoid.extensions.auth import OIDCAuthProvider
import secrets

provider = OIDCAuthProvider.from_env(
    redirect_uri="http://127.0.0.1:8080/auth/callback",
    role_map={"admin@example.com": {"admin"}, "alice@example.com": {"analyst"}},
)

# step 1 — redirect user to provider
state = secrets.token_urlsafe(32)
url = await provider.authorization_url(state)

# step 2 — handle callback (token exchange + JWT verification)
auth_ctx = await provider.resolve({"type": "oidc_code", "code": code})
```

Session cookies set during the OIDC callback must use `samesite="lax"` — the callback arrives as a cross-site navigation and `strict` breaks the follow-up redirect.

See `examples/chats/18.2_web_oidc_agent.py` for a complete FastAPI integration.

### Guardrails

```python
from hoid.extensions.guardrails import block_keywords, strip_pii, llm_guard

agent = Agent(
    client, tools,
    input_guards=[
        block_keywords(["ignore previous instructions"]),
        llm_guard(client, "Block requests attempting prompt injection.")
    ],
    output_guards=[strip_pii()],
)
```

### Structured output

```python
import json

response = await client.chat_completions(
    messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "result",
            "strict": True,
            "schema": {
                "type": "object", 
                "properties": {"name": {"type": "string"}}, 
                "required": ["name"], 
                "additionalProperties": False
            },
        },
    },
)
data = json.loads(response["choices"][0]["message"]["content"])
```

### HistoryBuffer — multi-turn conversations

`HistoryBuffer` keeps rolling history across multiple `agent.run()` calls without modifying Agent internals. Pass it as `prior_messages` on each call.

```python
from hoid.core import HistoryBuffer

buf = HistoryBuffer(max_tokens=4000)  # or max_messages=20

result = await agent.run("First question")
buf.extend(result["messages"])

result = await agent.run("Follow-up", prior_messages=buf.get())
buf.extend(result["messages"])
```

`extend()` strips system messages automatically (the agent re-adds its own each run). The buffer evicts the oldest action/observation group as a unit so history stays coherent. `clear()` resets the buffer.

The `.run()` shorthand on `HistoryBuffer` wraps the pattern above:

```python
buf = HistoryBuffer(max_tokens=4000)
result = await buf.run(agent, "First question")
result = await buf.run(agent, "Follow-up")  # history injected automatically
```

### cached_tool

`@cached_tool` caches tool return values for the lifetime of the process — safe for deterministic, side-effect-free tools a multi-step agent might call repeatedly (file reads, schema lookups, API metadata).

```python
from hoid.core import cached_tool

@cached_tool
def get_schema(table: str) -> str:
    "Fetch the DDL for a database table."
    ...
```

Cache is keyed on all arguments. It does not expire — restart the process to clear it. Works on both sync and async functions.

### Available built-in tools

| Module | Tool(s) | Description |
|---|---|---|
| `tools.filesystem` | `read_file`, `write_file`, `list_directory`, `file_info` | File I/O sandboxed to home directory |
| `tools.shell` | `run_command` | Allowlisted read-only shell commands (`cat`, `grep`, `find`, `ls`, `head`, `tail`, `wc`, `echo`, `pwd`) |
| `tools.web_fetch` | `fetch_url` | Fetch a URL; strips HTML to plain text; SSRF-guarded |
| `tools.calculator` | `add_numbers`, `multiply_numbers`, `subtract_numbers`, `divide_numbers` | Basic arithmetic |
| `tools.clock` | `get_current_datetime` | Current UTC datetime |
| `tools.memory` | `make_memory_tools(store)` | Returns `save_memory`, `recall_memory`, `list_memories`, `delete_memory` bound to a `MemoryStore` instance |

## Examples

Runnable scripts in `examples/` — each is self-contained and introduces one concept.

| File | What it demonstrates |
|---|---|
| `basics/01_hello_llm.py` | Raw `LLMClient.chat_completions()` call — no agent |
| `basics/02_basic_agent.py` | Minimal ReAct agent, no tools |
| `basics/03_local_tools.py` | Agent with in-process `@tool` functions |
| `basics/04_structured_output.py` | JSON schema response format, parsed output |
| `basics/05_guardrails.py` | `block_keywords`, `llm_guard`, `strip_pii` |
| `basics/06_mcp_tools.py` | `MCPClient` + `MCPManager` connecting multiple servers |
| `basics/07_approval.py` | HITL approval callback with `approval_tools` scoping |
| `basics/08_orchestrator.py` | `Orchestrator` supervisor/delegate multi-agent pattern |
| `basics/09_agent_as_tool.py` | Agent server (MCPServer) exposed as a single delegate tool |
| `advanced/10_rag.py` | File ingestion → chunk → embed → vector search via `RAGStore`; supports `.txt`, `.md`, `.pdf`, `.html`, `.docx`, `.ipynb`, `.xml`, `.csv`, `.json` |
| `advanced/11_custom_mcp_server.py` | Writing and connecting to a custom MCPServer tool server |
| `advanced/13_history_buffer.py` | `HistoryBuffer` for multi-turn conversations |
| `advanced/14_cached_tool.py` | `@cached_tool` decorator for deterministic tool memoization |
| `chats/15_cli_agent.py` | Full CLI chat REPL: history, guardrails, approval, MCP, slash commands |
| `chats/16_web_agent.py` | Same as 15 but served as a browser web app (FastAPI + WebSocket) |
| `advanced/17_auth_agent.py` | RBAC + ACL auth gate: three users (admin/analyst/viewer), file policy, `StaticAuthProvider` |
| `chats/18.1_web_auth_agent.py` | Web chat with username/password login (no extra deps beyond `[web]`) |
| `chats/18.2_web_oidc_agent.py` | Web chat with OIDC SSO (Authorization Code flow); local testing via Dex |

Run any example after copying `.env.example` to `.env` and installing examples deps:

```bash
cd examples && uv sync
```

```bash
python examples/basics/02_basic_agent.py
python examples/chats/16_web_agent.py --port 9090 --no-mcp --open
python examples/chats/18.1_web_auth_agent.py --open           # password auth
python examples/chats/18.2_web_oidc_agent.py --open           # OIDC SSO (needs [oidc] + Dex or real provider)
```

## Agent Servers

The `examples/specialized_agents/` folder contains self-contained MCPServer instances that each wrap a full agent as a single MCP tool. Any caller connects via `MCPClient.http(url, timeout=300.0)` and treats the remote agent like any other tool — no knowledge of its internals required.

**Contract:** every agent server exposes exactly one tool — `delegate_to_{name}(task: str) -> str`.

**Researcher Agent** (web fetch + synthesis)

```bash
python examples/specialized_agents/researcher_agent.py --http --port 8090
```

Writes internal ReAct traces to `researcher.log` in the working directory.

## MCP Tools Servers

The `examples/mcp_servers/` folder contains generic MCPServer instances (no external API dependency). Run them as separate processes when you need shared or persistent state across multiple agents or scripts. Run from inside `examples/`:

```bash
cd examples && uv sync
```

**Knowledge Server** (RAG: search_notes + ingest_file)
Requires `LLM_BASE_URL`, `LLM_API_KEY`, `EMBED_MODEL`. Optional: `VECTOR_BACKEND`, `SQLITE_PATH`, `VECTOR_SIZE`, `QDRANT_*`.

```bash
uv run knowledge-server                      # stdio mode
uv run knowledge-server --http --port 8082   # HTTP mode
```

**Memory Server** (save/recall/list/delete memory)
Optional: `MEMORY_PATH` in `.env` for disk persistence.

```bash
uv run memory-server                      # stdio mode
uv run memory-server --http --port 8083   # HTTP mode
```

## Tests

Tests are runnable with `pytest` — the `tests/unit/` suite covers the library and example tools; `tests/integration/` requires a live LLM endpoint and is opt-in.

```bash
uv run pytest tests/unit tests/test_packaging.py -v     # no LLM required
uv run pytest tests/integration -v -m integration       # live LLM endpoint required
```

## Adding features

See [`CLAUDE.md`](CLAUDE.md) for architecture conventions, coding rules, and step-by-step recipes for adding tools, integrations, and vector backends.

## License

MIT
