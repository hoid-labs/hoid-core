from __future__ import annotations

# Maps the importable Python package name to the pyproject extra that installs it.
# Keep this in sync with pyproject.toml [project.optional-dependencies].
# tests/test_packaging.py asserts that every key here matches a declared extra.
EXTRAS_MAP: dict[str, str] = {
    "fastapi": "mcp",
    "jwt": "oidc",
    "semantic_text_splitter": "rag",
    "pypdf": "rag",
    "sqlite_vec": "rag",
    "qdrant_client": "qdrant",
}


def require[T](name: str, obj: T | None) -> T:
    """Return *obj* unchanged, or raise ImportError with an install hint if it is None.

    Args:
        name: Importable Python package name — must be a key in EXTRAS_MAP.
        obj: The sentinel that was set to None when the package failed to import.
    """
    if obj is None:
        extra = EXTRAS_MAP.get(name, name)
        raise ImportError(
            f"'{name}' is required but not installed. "
            f"Install it with: uv pip install 'hoid[{extra}]'"
        )
    return obj
