import asyncio
import json
import os
from pathlib import Path

from hoid._env import load_env


class MemoryStore:
    "Key-value memory store backed by an in-memory dict or an optional JSON file."

    # --- construction ---

    def __init__(self, path: str | None = None):
        """
        Args:
            path: Optional path to a JSON file for persistence. Omit for in-memory only (data lost on process exit).
        """
        self._path = Path(path) if path else None
        # hydrate from disk if path provided
        self._data = (
            json.loads(self._path.read_text())
            if self._path and self._path.exists()
            else {}
        )
        # prevents interleaved writes from concurrent async callers in the MCP server
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(cls) -> "MemoryStore":
        load_env()
        return cls(path=os.getenv("MEMORY_PATH"))

    # --- lifecycle ---

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    # --- operations ---

    @staticmethod
    def _normalize(key: str) -> str:
        # collapse whitespace/hyphens so variant spellings map to the same key
        return "_".join(key.lower().split()).replace("-", "_")

    def _persist(self) -> None:
        if self._path:
            self._path.write_text(json.dumps(self._data, indent=2))

    async def save(self, key: str, value: str):
        async with self._lock:
            self._data[self._normalize(key)] = value
            self._persist()

    def load(self, key: str) -> str | None:
        return self._data.get(self._normalize(key))

    def list_keys(self) -> list[str]:
        return list(self._data.keys())

    async def clear(self, key: str):
        async with self._lock:
            self._data.pop(self._normalize(key), None)
            self._persist()
