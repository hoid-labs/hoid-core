from hoid.core import tool
from hoid.extensions.memory import MemoryStore


def make_memory_tools(store: MemoryStore) -> list:
    "Create @tool functions bound to a MemoryStore instance; no global state."

    @tool
    async def save_memory(key: str, value: str) -> str:
        """Save a value to memory under the given key.

        Args:
            key: Identifier for the memory entry.
            value: The value to store.
        """
        await store.save(key, value)
        return f"Saved '{key}'."

    @tool
    def recall_memory(key: str) -> str:
        """Retrieve a value from memory by key.

        Args:
            key: The key to look up.
        """
        value = store.load(key)
        if value is None:
            return f"No memory found for '{key}'."
        return value

    @tool
    def list_memories() -> str:
        "List all stored memory keys."
        keys = store.list_keys()
        if not keys:
            return "No memories stored."
        return "\n".join(keys)

    @tool
    async def delete_memory(key: str) -> str:
        """Delete a memory entry by key.

        Args:
            key: The key to delete.
        """
        await store.clear(key)
        return f"Deleted '{key}'."

    return [save_memory, recall_memory, list_memories, delete_memory]
