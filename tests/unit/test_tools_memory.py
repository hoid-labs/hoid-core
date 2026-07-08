from examples.tools.memory import make_memory_tools
from hoid.extensions.memory import MemoryStore


def _store():
    return MemoryStore()


def test_make_memory_tools_returns_four_callables():
    tools = make_memory_tools(_store())
    names = {t.name for t in tools}
    print(f"tool names: {sorted(names)}")
    assert len(tools) == 4
    assert names == {"save_memory", "recall_memory", "list_memories", "delete_memory"}


def test_all_tools_have_schema():
    tools = make_memory_tools(_store())
    for t in tools:
        assert hasattr(t, "schema"), f"{t.name} missing schema"


async def test_save_and_recall_roundtrip():
    store = _store()
    tools = {t.name: t for t in make_memory_tools(store)}
    await tools["save_memory"](key="color", value="blue")
    result = tools["recall_memory"](key="color")
    print(f"recall('color') = {result!r}")
    assert result == "blue"


async def test_recall_missing_key_returns_message():
    tools = {t.name: t for t in make_memory_tools(_store())}
    result = tools["recall_memory"](key="nonexistent")
    print(f"recall('nonexistent') = {result!r}")
    assert "No memory" in result


async def test_list_memories_shows_saved_key():
    store = _store()
    tools = {t.name: t for t in make_memory_tools(store)}
    await tools["save_memory"](key="item", value="value")
    result = tools["list_memories"]()
    print(f"list_memories() = {result!r}")
    assert "item" in result


async def test_list_memories_empty_store():
    tools = {t.name: t for t in make_memory_tools(_store())}
    result = tools["list_memories"]()
    print(f"list_memories() on empty store = {result!r}")
    assert "No memories" in result


async def test_delete_memory_removes_key():
    store = _store()
    tools = {t.name: t for t in make_memory_tools(store)}
    await tools["save_memory"](key="temp", value="x")
    await tools["delete_memory"](key="temp")
    result = tools["recall_memory"](key="temp")
    print(f"recall('temp') after delete = {result!r}")
    assert "No memory" in result
