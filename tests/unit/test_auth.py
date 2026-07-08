import json
import pathlib
import pprint
import tempfile

from hoid.core import Agent
from hoid.core.tools import tool
from hoid.extensions.auth import (
    AuthContext,
    AuthGate,
    FilePolicyBackend,
    MemoryPolicyBackend,
    StaticAuthProvider,
)


@tool
def tool_a() -> str:
    "Tool A."
    return "a"


@tool
def tool_b() -> str:
    "Tool B."
    return "b"


@tool
def tool_c() -> str:
    "Tool C."
    return "c"


ALL_TOOLS = [tool_a, tool_b, tool_c]

POLICY = {
    "roles": {
        "admin": {"tools": ["*"]},
        "analyst": {"tools": ["tool_a", "tool_b"]},
        "viewer": {"tools": ["tool_a"]},
    },
    "users": {
        "alice": {"roles": ["analyst"]},
        "bob": {"roles": ["viewer"], "extra_tools": ["tool_c"]},
        "svc": {"roles": ["admin"]},
        "dave": {"roles": []},
    },
}


def make_gate(policy: dict) -> AuthGate:
    return AuthGate(MemoryPolicyBackend(policy))


def visible_names(gate: AuthGate, schemas: list, ctx: AuthContext) -> list[str]:
    return [s["function"]["name"] for s in gate.filter_schemas(schemas, ctx)]


def test_schema_filtering_analyst():
    gate = make_gate(POLICY)
    ctx = AuthContext(user_id="alice", roles={"analyst"})
    schemas = [t.schema for t in ALL_TOOLS]
    names = visible_names(gate, schemas, ctx)
    print(f"analyst sees: {names}")
    assert names == ["tool_a", "tool_b"]


def test_schema_filtering_viewer_with_extra_acl():
    gate = make_gate(POLICY)
    ctx = AuthContext(user_id="bob", roles={"viewer"})
    schemas = [t.schema for t in ALL_TOOLS]
    names = visible_names(gate, schemas, ctx)
    print(f"viewer+extra_acl sees: {sorted(names)}")
    assert set(names) == {"tool_a", "tool_c"}


def test_schema_filtering_admin_wildcard():
    gate = make_gate(POLICY)
    ctx = AuthContext(user_id="svc", roles={"admin"})
    schemas = [t.schema for t in ALL_TOOLS]
    names = visible_names(gate, schemas, ctx)
    print(f"admin sees: {sorted(names)}")
    assert set(names) == {"tool_a", "tool_b", "tool_c"}


def test_schema_filtering_no_grants():
    gate = make_gate(POLICY)
    ctx = AuthContext(user_id="dave", roles=set())
    schemas = [t.schema for t in ALL_TOOLS]
    names = visible_names(gate, schemas, ctx)
    print(f"no-grants user sees: {names}")
    assert names == []


def test_pass_through_when_no_context():
    gate = make_gate(POLICY)
    schemas = [t.schema for t in ALL_TOOLS]
    names = visible_names(gate, schemas, None)
    assert set(names) == {"tool_a", "tool_b", "tool_c"}


def test_authorize_allows_permitted_tool():
    gate = make_gate(POLICY)
    ctx = AuthContext(user_id="alice", roles={"analyst"})
    assert gate.authorize("tool_a", ctx) is True
    assert gate.authorize("tool_b", ctx) is True


def test_authorize_denies_forbidden_tool():
    gate = make_gate(POLICY)
    ctx = AuthContext(user_id="alice", roles={"analyst"})
    assert gate.authorize("tool_c", ctx) is False


def test_authorize_admin_wildcard():
    gate = make_gate(POLICY)
    ctx = AuthContext(user_id="svc", roles={"admin"})
    assert gate.authorize("tool_a", ctx) is True
    assert gate.authorize("tool_c", ctx) is True
    assert gate.authorize("anything_unknown", ctx) is True


def test_denied_tools_acl():
    policy = {
        "roles": {"editor": {"tools": ["tool_a", "tool_b", "tool_c"]}},
        "users": {"carol": {"roles": ["editor"], "denied_tools": ["tool_c"]}},
    }
    gate = make_gate(policy)
    ctx = AuthContext(user_id="carol", roles={"editor"})
    schemas = [t.schema for t in ALL_TOOLS]
    names = visible_names(gate, schemas, ctx)
    print(f"editor with denied_tools=['tool_c'] sees: {sorted(names)}")
    assert "tool_c" not in names
    assert set(names) == {"tool_a", "tool_b"}


async def test_agent_schema_filtering_integration(mock_llm):
    gate = make_gate(POLICY)
    client = mock_llm()
    ctx = AuthContext(user_id="alice", roles={"analyst"})

    agent = Agent(client, tools=ALL_TOOLS, auth_gate=gate)
    await agent.run("do something", auth_context=ctx)

    sent_names = [t["function"]["name"] for t in client.calls[0]["tools"]]
    print(f"tools sent to LLM (as analyst): {sorted(sent_names)}")
    assert set(sent_names) == {"tool_a", "tool_b"}


async def test_agent_execution_time_denial(recording_hook):
    gate = make_gate(POLICY)

    class ForcedToolCallClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        _step = 0

        async def chat_completions(self, messages, tools=None, **kwargs):
            self._step += 1
            if self._step == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "tool_c",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "done",
                            "tool_calls": None,
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    ctx = AuthContext(user_id="alice", roles={"analyst"})
    agent = Agent(ForcedToolCallClient(), tools=ALL_TOOLS, auth_gate=gate)
    await agent.run("do something", auth_context=ctx)
    errors = [e for e in recording_hook.events if e.event_type == "tool_error"]
    print("errors captured:")
    pprint.pprint(errors)
    assert any(
        "access_denied" in str(e.payload.get("reason", "")) for e in errors
    )


async def test_backward_compat_no_auth_gate(mock_llm):
    client = mock_llm()
    agent = Agent(client, tools=ALL_TOOLS)
    await agent.run("do something")
    sent_names = {t["function"]["name"] for t in client.calls[0]["tools"]}
    print(f"no-auth agent sees all tools: {sorted(sent_names)}")
    assert sent_names == {"tool_a", "tool_b", "tool_c"}


async def test_file_policy_backend():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(POLICY, f)
        path = pathlib.Path(f.name)

    try:
        backend = FilePolicyBackend(path)
        gate = AuthGate(backend)
        ctx = AuthContext(user_id="svc", roles={"admin"})
        schemas = [t.schema for t in ALL_TOOLS]
        names = visible_names(gate, schemas, ctx)
        print(f"file-policy admin sees: {sorted(names)}")
        assert set(names) == {"tool_a", "tool_b", "tool_c"}
    finally:
        path.unlink(missing_ok=True)


async def test_static_auth_provider():
    provider = StaticAuthProvider(
        api_keys={"sk-abc": AuthContext(user_id="svc", roles={"admin"})},
        users={"alice": AuthContext(user_id="alice", roles={"analyst"})},
    )
    ctx_api = await provider.resolve({"type": "api_key", "key": "sk-abc"})
    ctx_user = await provider.resolve({"type": "username", "username": "alice"})
    ctx_miss = await provider.resolve({"type": "api_key", "key": "wrong"})
    print(f"api_key ctx: {ctx_api}")
    print(f"username ctx: {ctx_user}")
    print(f"wrong key ctx: {ctx_miss}")
    assert ctx_api is not None and ctx_api.user_id == "svc"
    assert ctx_user is not None and ctx_user.user_id == "alice"
    assert ctx_miss is None
