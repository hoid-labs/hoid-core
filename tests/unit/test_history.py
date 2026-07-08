import pprint

from hoid.core import HistoryBuffer


def _user(text):
    return {"role": "user", "content": text}


def _assistant(text):
    return {"role": "assistant", "content": text}


def _tool_call(call_id):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }
        ],
    }


def _tool_result(call_id):
    return {"role": "tool", "tool_call_id": call_id, "content": "ok"}


def test_extend_and_get():
    buf = HistoryBuffer()
    buf.extend([_user("hello"), _assistant("hi")])
    msgs = buf.get()
    print("messages:")
    pprint.pprint(msgs)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_system_messages_stripped_on_extend():
    buf = HistoryBuffer()
    buf.extend(
        [{"role": "system", "content": "instructions"}, _user("q"), _assistant("a")]
    )
    msgs = buf.get()
    print(f"roles after strip: {[m['role'] for m in msgs]}")
    assert all(m["role"] != "system" for m in msgs)
    assert len(msgs) == 2


def test_clear_resets_buffer():
    buf = HistoryBuffer()
    buf.extend([_user("x"), _assistant("y")])
    buf._messages.clear()
    assert buf.get() == []


def test_max_messages_eviction():
    buf = HistoryBuffer(max_messages=2)
    buf.extend([_user("first"), _assistant("reply1")])
    buf.extend([_user("second"), _assistant("reply2")])
    msgs = buf.get()
    contents = [m.get("content") for m in msgs]
    print(f"remaining contents after eviction (max_messages=2): {contents}")
    assert len(msgs) <= 2
    assert "first" not in contents


def test_max_tokens_budget_trims():
    buf = HistoryBuffer(max_tokens=10)
    long_content = "x" * 400
    for _ in range(3):
        buf.extend([_user(long_content), _assistant(long_content)])
    token_estimate = buf._token_estimate()
    print(f"token_estimate after trim: {token_estimate} (budget: {int(10 * 0.8)})")
    assert token_estimate <= int(10 * 0.8)


def test_tool_call_pair_evicted_as_unit():
    buf = HistoryBuffer(max_messages=2)
    buf.extend([_tool_call("c1"), _tool_result("c1")])
    buf.extend([_user("follow-up"), _assistant("answer")])
    msgs = buf.get()
    print(f"roles after tool-pair eviction: {[m.get('role') for m in msgs]}")
    assert all(m.get("role") != "tool" for m in msgs)


def test_get_returns_copy():
    buf = HistoryBuffer()
    buf.extend([_user("a"), _assistant("b")])
    copy = buf.get()
    copy.append(_user("injected"))
    assert len(buf.get()) == 2
