import asyncio
import contextlib
import copy
import inspect
import json
import logging
import os
import time
from typing import Any

import httpx

from hoid.core.observability import MCPEvent

log = logging.getLogger(__name__)

_PROTO_VERSION = "2025-03-26"
_INIT_PARAMS = {
    "protocolVersion": _PROTO_VERSION,
    "capabilities": {},
    "clientInfo": {"name": "hoid", "version": "1.0.0"},
}


class MCPClient:
    "Manages a single MCP server connection over stdio or streamable-HTTP transport."

    # --- construction ---

    def __init__(
        self,
        transport: str,
        timeout: float | None = None,
        on_mcp: Any | None = None,
        **kwargs,
    ):
        self._transport = transport
        self._timeout = timeout
        self.on_mcp = on_mcp
        # stdio state
        self._command: str = kwargs.get("command", "")
        self._args: list = kwargs.get("args", [])
        self._env: dict | None = kwargs.get("env")
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._id_counter = 0
        # http state
        self._url: str = kwargs.get("url", "")
        self._http: httpx.AsyncClient | None = None

    @classmethod
    def stdio(
        cls,
        command: str,
        args: list[str] | None = None,
        env: dict | None = None,
        timeout: float = 60.0,
        on_mcp: Any | None = None,
    ):
        """Connect to an MCP server running as a subprocess over stdio.

        Args:
            command: Executable to launch (e.g. `"uv"`).
            args: Command-line arguments (e.g. `["run", "python", "examples/mcp_servers/memory_server.py"]`).
            env: Optional environment variables merged into the subprocess environment.
            timeout: Per-call timeout in seconds (default 60). Subprocess is killed on context exit.
            on_mcp: Optional observability callback receiving `MCPEvent` for each tool call.
        """
        return cls(
            "stdio",
            timeout=timeout,
            on_mcp=on_mcp,
            command=command,
            args=args or [],
            env=env,
        )

    @classmethod
    def http(cls, url: str, timeout: float = 300.0, on_mcp: Any | None = None):
        """Connect to an MCP server over streamable HTTP.

        Args:
            url: Full URL to the MCP endpoint (e.g. `http://localhost:8080/mcp`).
            timeout: Request timeout in seconds (default 300). Must cover the full remote ReAct loop round-trip.
            on_mcp: Optional observability callback receiving `MCPEvent` for each tool call.
        """
        return cls("http", timeout=timeout, on_mcp=on_mcp, url=url)

    # --- context manager ---

    async def __aenter__(self):
        if self._transport == "stdio":
            await self._stdio_connect()
        else:
            self._http = httpx.AsyncClient(timeout=self._timeout)
            await self._http_call("initialize", _INIT_PARAMS)
            await self._http_notify("notifications/initialized")
        return self

    async def __aexit__(self, *exc):
        if self._transport == "stdio":
            await self._stdio_close()
        elif self._http:
            await self._http.aclose()
            self._http = None

    # --- stdio internals ---

    async def _stdio_connect(self):
        env = {**os.environ, **self._env} if self._env else None
        self._process = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._reader_task = asyncio.create_task(self._stdio_reader())
        await self._stdio_call("initialize", _INIT_PARAMS)
        # notify the server that the client is ready to send requests
        await self._stdio_notify("notifications/initialized")

    async def _stdio_reader(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("stdio reader started before subprocess connected")
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                try:
                    data = json.loads(line.decode())
                except json.JSONDecodeError:
                    continue
                req_id = data.get("id")
                if req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if not fut.done():
                        if "error" in data:
                            fut.set_exception(
                                RuntimeError(data["error"].get("message", "MCP error"))
                            )
                        else:
                            fut.set_result(data.get("result", {}))
        except Exception as e:
            log.warning("MCP stdio reader stopped: %s", e)

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    async def _stdio_notify(self, method: str, params: dict | None = None) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("stdio notify before subprocess connected")
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}})
        process.stdin.write((msg + "\n").encode())
        await process.stdin.drain()

    async def _stdio_call(self, method: str, params: dict) -> dict:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("stdio call before subprocess connected")
        req_id = self._next_id()
        fut = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        msg = json.dumps(
            {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        )
        process.stdin.write((msg + "\n").encode())
        await process.stdin.drain()
        return await fut

    async def _stdio_close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        if self._process:
            self._process.terminate()
            await self._process.wait()

    # --- http internals ---

    async def _http_call(self, method: str, params: dict) -> dict:
        req_id = self._next_id()
        msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        async with self._http.stream("POST", self._url, json=msg) as resp:  # type: ignore[union-attr]
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "error" in data:
                        raise RuntimeError(data["error"].get("message", "MCP error"))
                    return data.get("result", {})
        raise RuntimeError(f"No response received from server for '{method}'")

    async def _http_notify(self, method: str, params: dict | None = None):
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        try:
            # drain the response stream so the server can finalize the response
            async with self._http.stream("POST", self._url, json=msg) as resp:  # type: ignore[union-attr]
                async for _ in resp.aiter_bytes():
                    pass
        except Exception as e:
            log.debug("MCP notification delivery failed (non-fatal): %s", e)

    # --- tool access ---

    async def get_tools(self):
        if self._transport == "stdio":
            result = await self._stdio_call("tools/list", {})
        else:
            result = await self._http_call("tools/list", {})
        return [self._make_proxy(t) for t in result.get("tools", [])]

    def _make_proxy(self, mcp_tool: dict):
        # factory isolates closure scope so each proxy captures its own tool reference
        transport = self._transport
        client = self
        call_timeout = self._timeout
        on_mcp = self.on_mcp
        server_id = self._command or self._url

        async def proxy_fn(**kwargs):
            args_payload = {"name": mcp_tool["name"], "arguments": kwargs}
            start = time.perf_counter()
            err: str | None = None
            try:
                if transport == "stdio":
                    coro = client._stdio_call("tools/call", args_payload)
                    result = (
                        await asyncio.wait_for(coro, timeout=call_timeout)
                        if call_timeout is not None
                        else await coro
                    )
                else:
                    result = await client._http_call("tools/call", args_payload)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                raise
            finally:
                latency_ms = (time.perf_counter() - start) * 1000.0
                if on_mcp:
                    await on_mcp(
                        MCPEvent(
                            server=server_id or "unknown",
                            tool=mcp_tool["name"],
                            args=args_payload,
                            latency_ms=latency_ms,
                            error=err,
                        )
                    )
            parts = [
                c["text"]
                for c in result.get("content", [])
                if c.get("type") == "text" and c.get("text")
            ]
            return "\n".join(parts) or "(no output)"

        proxy_fn.__name__ = mcp_tool["name"]

        desc_lines, param_descs, in_args = [], {}, False
        for line in inspect.cleandoc(mcp_tool.get("description") or "").splitlines():
            s = line.strip()
            if s.lower() in ("args:", "arguments:"):
                in_args = True
            elif in_args and ":" in s:
                pname, _, rest = s.partition(":")
                param_descs[pname.strip()] = rest.strip()
            elif not in_args:
                desc_lines.append(line)

        clean_desc = "\n".join(desc_lines).strip()
        # deep copy so mutations in one proxy don't bleed into another
        schema = copy.deepcopy(
            mcp_tool.get("inputSchema") or {"type": "object", "properties": {}}
        )
        for pname, pdesc in param_descs.items():
            if pname in schema.get("properties", {}):
                schema["properties"][pname]["description"] = pdesc

        # schema grafted so the agent sees a uniform tool interface
        proxy_fn.schema = {  # type: ignore[attr-defined]
            "type": "function",
            "function": {
                "name": mcp_tool["name"],
                # Some LLM providers reject empty description strings with 400;
                # fall back to the tool name so the schema is always valid.
                "description": clean_desc or mcp_tool["name"],
                "parameters": schema,
            },
        }
        proxy_fn.name = mcp_tool["name"]  # type: ignore[attr-defined]
        proxy_fn.description = clean_desc  # type: ignore[attr-defined]
        return proxy_fn


class MCPManager:
    "Aggregates multiple MCPClient connections and exposes a unified tool list."

    def __init__(self, clients: list):
        """
        Args:
            clients: List of `MCPClient` instances to aggregate. All are entered as a single context manager.
        """
        self.clients = clients
        self._exit_stack = contextlib.AsyncExitStack()

    async def __aenter__(self):
        # single stack tears down all clients together on exit
        for client in self.clients:
            await self._exit_stack.enter_async_context(client)
        return self

    async def __aexit__(self, *exc):
        await self._exit_stack.aclose()

    async def get_all_tools(self):
        # parallel fetch avoids sequential round-trips when multiple servers are connected
        tasks = [client.get_tools() for client in self.clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        tools = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.warning("MCP client %d failed to load tools: %s", i, result)
            else:
                tools.extend(result)
        return tools
