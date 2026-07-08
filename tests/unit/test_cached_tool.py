from hoid.core import cached_tool


def test_sync_cached_calls_fn_once():
    count = {"n": 0}

    @cached_tool
    def lookup(key: str) -> str:
        "Return a value for a key."
        count["n"] += 1
        return f"value:{key}"

    assert lookup("foo") == "value:foo"
    assert lookup("foo") == "value:foo"
    print(f"call_count after two identical calls: {count['n']} (expected 1)")
    assert count["n"] == 1


def test_different_args_produce_different_entries():
    count = {"n": 0}

    @cached_tool
    def lookup(key: str) -> str:
        "Return a value for a key."
        count["n"] += 1
        return f"value:{key}"

    lookup("foo")
    lookup("bar")
    print(f"call_count after 'foo' then 'bar': {count['n']} (expected 2)")
    assert count["n"] == 2


async def test_async_tool_is_cached():
    count = {"n": 0}

    @cached_tool
    async def async_lookup(key: str) -> str:
        "Return a value for a key asynchronously."
        count["n"] += 1
        return f"async:{key}"

    assert await async_lookup("x") == "async:x"
    assert await async_lookup("x") == "async:x"
    print(f"async call_count after two identical calls: {count['n']} (expected 1)")
    assert count["n"] == 1


def test_lru_eviction():
    count = {"n": 0}

    @cached_tool(maxsize=2)
    def tiny(key: str) -> str:
        "Return a value for a key with tiny LRU cache."
        count["n"] += 1
        return f"v:{key}"

    tiny("a")
    tiny("b")
    before = count["n"]
    tiny("c")  # evicts "a" (LRU)
    tiny("a")  # cache miss — must recompute
    print(f"call_count: {count['n']}, expected before({before}) + 2 = {before + 2}")
    assert count["n"] == before + 2


def test_schema_attributes_retained():
    @cached_tool
    def lookup(key: str) -> str:
        "Return a value for a key."
        return f"value:{key}"

    assert hasattr(lookup, "schema")
    assert hasattr(lookup, "name")
    assert hasattr(lookup, "description")
    print(f"name={lookup.name!r}  description={lookup.description!r}")
    assert lookup.name == "lookup"
