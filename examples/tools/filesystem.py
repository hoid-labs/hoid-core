import pathlib

from hoid.core import tool

_SAFE_ROOT = pathlib.Path.home()
_MAX_CHARS = 20_000
_MAX_CONTENT_BYTES = 10_485_760


def _safe_path(path):
    # raises if outside home dir to sandbox the agent
    p = pathlib.Path(path).expanduser().resolve()
    try:
        p.relative_to(_SAFE_ROOT)
    except ValueError:
        raise PermissionError(
            f"Path '{p}' is outside allowed root '{_SAFE_ROOT}'"
        ) from None
    return p


@tool
def read_file(path: str, encoding: str = "utf-8") -> str:
    """Read a text file and return its contents.

    Args:
        path: Path to the file to read.
        encoding: Text encoding (default utf-8).
    """
    content = _safe_path(path).read_text(encoding=encoding)
    return content[:_MAX_CHARS] + (
        "\n...[truncated]" if len(content) > _MAX_CHARS else ""
    )


@tool
def write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """Write text to a file, creating parent directories if needed.

    Args:
        path: Destination file path.
        content: Text to write.
        encoding: Text encoding (default utf-8).
    """
    if len(content.encode(encoding)) > _MAX_CONTENT_BYTES:
        raise ValueError(
            f"Content exceeds the {_MAX_CONTENT_BYTES // 1_048_576} MB write limit"
        )
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding=encoding)
    return f"Written {len(content)} chars to {p}"


@tool
def list_directory(path: str) -> list:
    """List the contents of a directory.

    Args:
        path: Path to the directory to list.
    """
    p = _safe_path(path)
    if not p.is_dir():
        raise NotADirectoryError(f"'{p}' is not a directory")
    entries = []
    for entry in sorted(p.iterdir()):
        entries.append(entry.name + ("/" if entry.is_dir() else ""))
    return entries


@tool
def file_info(path: str) -> dict:
    """Return metadata about a file or directory.

    Args:
        path: Path to the file or directory.
    """
    p = _safe_path(path)
    s = p.stat()
    return {
        "path": str(p),
        "type": "directory" if p.is_dir() else "file",
        "size_bytes": s.st_size,
        "modified": s.st_mtime,  # Unix timestamp — seconds since epoch
    }
