from hoid.extensions.memory import MemoryStore


async def test_save_and_load_roundtrip():
    store = MemoryStore()
    await store.save("greeting", "hello")
    value = store.load("greeting")
    print(f"load('greeting') = {value!r}")
    assert value == "hello"


async def test_load_missing_key_returns_none():
    store = MemoryStore()
    value = store.load("no_such_key")
    print(f"load('no_such_key') = {value!r}")
    assert value is None


async def test_list_keys_returns_saved_keys():
    store = MemoryStore()
    await store.save("a", "1")
    await store.save("b", "2")
    keys = store.list_keys()
    print(f"list_keys() = {keys}")
    assert "a" in keys
    assert "b" in keys


async def test_clear_removes_key():
    store = MemoryStore()
    await store.save("temp", "value")
    await store.clear("temp")
    value = store.load("temp")
    print(f"load('temp') after clear = {value!r}")
    assert value is None


async def test_key_normalisation():
    "Variant spellings of the same key map to the same entry."
    store = MemoryStore()
    await store.save("My Key", "value")
    v1 = store.load("my_key")
    v2 = store.load("My-Key")
    print(f"load('my_key') = {v1!r}  load('My-Key') = {v2!r}")
    assert v1 == "value"
    assert v2 == "value"


async def test_overwrite_existing_key():
    store = MemoryStore()
    await store.save("item", "first")
    await store.save("item", "second")
    assert store.load("item") == "second"


async def test_file_persistence(tmp_path):
    "Data written to a path-backed store survives creating a new store from the same path."
    path = tmp_path / "memory.json"
    store1 = MemoryStore(str(path))
    await store1.save("key", "persisted")

    store2 = MemoryStore(str(path))
    value = store2.load("key")
    print(f"reloaded from {path}: {value!r}")
    assert value == "persisted"


async def test_in_memory_store_not_persisted(tmp_path):
    "In-memory store does not create any files."
    store = MemoryStore()
    await store.save("key", "val")
    assert not any(tmp_path.iterdir())


async def test_clear_nonexistent_key_is_silent():
    store = MemoryStore()
    await store.clear("never_set")  # must not raise
