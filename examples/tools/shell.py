import asyncio
import pathlib

from hoid.core import tool

_SAFE_ROOT = pathlib.Path.home()
_MAX_CHARS = 20_000

# allowlist — only safe, read-only commands permitted
_ALLOWED_COMMANDS = {
    "cat",
    "echo",
    "find",
    "grep",
    "head",
    "ls",
    "pwd",
    "tail",
    "wc",
}


def _safe_cwd(cwd: str | None) -> pathlib.Path:
    # raises if cwd escapes the home directory sandbox
    if cwd is None:
        return _SAFE_ROOT
    p = pathlib.Path(cwd).expanduser().resolve()
    try:
        p.relative_to(_SAFE_ROOT)
    except ValueError:
        raise PermissionError(
            f"cwd '{p}' is outside allowed root '{_SAFE_ROOT}'"
        ) from None
    return p


def _safe_args(args: list[str]) -> None:
    # absolute paths in args can escape the cwd sandbox — validate each one
    for arg in args:
        if arg.startswith("/"):
            p = pathlib.Path(arg).resolve()
            try:
                p.relative_to(_SAFE_ROOT)
            except ValueError:
                raise PermissionError(
                    f"Path argument '{p}' is outside allowed root '{_SAFE_ROOT}'"
                ) from None


@tool
async def run_command(
    command: str,
    args: list[str] | None = None,
    cwd: str | None = None,
    timeout: float = 30.0,
) -> str:
    """Run an allowlisted shell command and return its combined stdout and stderr.

    Args:
        command: The command to run. Must be in the allowed list (cat, echo, find, grep, head, ls, pwd, tail, wc).
        args: Optional list of arguments to pass to the command.
        cwd: Working directory for the command; must be within the home directory.
        timeout: Maximum seconds to wait before killing the process (default 30).
    """
    if command not in _ALLOWED_COMMANDS:
        raise PermissionError(
            f"Command '{command}' is not allowed. Allowed: {sorted(_ALLOWED_COMMANDS)}"
        )
    work_dir = _safe_cwd(cwd)
    _safe_args(args or [])
    proc = await asyncio.create_subprocess_exec(
        command,
        *(args or []),
        cwd=work_dir,
        stdout=asyncio.subprocess.PIPE,
        # stderr merged so the caller sees all output in one string
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.communicate()
        raise TimeoutError(
            f"Command '{command}' timed out after {timeout:.0f}s"
        ) from None
    output = stdout.decode(errors="replace")
    return output[:_MAX_CHARS] + (
        "\n...[truncated]" if len(output) > _MAX_CHARS else ""
    )
