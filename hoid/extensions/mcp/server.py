import asyncio
import dataclasses
import inspect
import json
import sys
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from hoid._optional import require as _require
from hoid.core.tools import build_schema

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
except ImportError:
    FastAPI = None  # type: ignore[assignment,misc]
    Request = None  # type: ignore[assignment,misc]
    StreamingResponse = None  # type: ignore[assignment,misc]

_PROTO_VERSION = "2025-03-26"


@dataclasses.dataclass
class MCPContext:
    "Per-request context injected into MCP tool handlers."

    lifespan: dict


def _mcp_schema(func: Callable) -> dict:
    schema = build_schema(func)
    params = schema["function"]["parameters"]
    # strip ctx — injected at call time, not part of the wire schema
    props = {k: v for k, v in params.get("properties", {}).items() if k != "ctx"}
    required = [r for r in params.get("required", []) if r != "ctx"]
    schema["function"]["parameters"] = {
        **params,
        "properties": props,
        "required": required,
    }
    return schema


class MCPServer:
    "Minimal MCP server supporting stdio and streamable-HTTP transports."

    def __init__(
        self,
        name: str,
        lifespan: Callable[["MCPServer"], AbstractAsyncContextManager[dict]] | None = None,
    ):
        """
        Args:
            name: Server name reported in the initialize handshake.
            lifespan: Optional asynccontextmanager factory `(server) -> AsyncIterator[dict]`.
        """
        self.name = name
        self._lifespan_factory = lifespan
        self._tools: dict[str, dict] = {}

    def tool(self):
        "Decorator that registers an async function as an MCP tool."

        def decorator(func: Callable):
            schema = _mcp_schema(func)
            self._tools[func.__name__] = {
                "name": func.__name__,
                "description": schema["function"]["description"],
                "inputSchema": schema["function"]["parameters"],
                "handler": func,
            }
            return func

        return decorator

    # --- JSON-RPC dispatch ---

    async def _handle(self, request: dict, lifespan_data: dict) -> dict | None:
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        # notifications have no id and must not receive a response
        if req_id is None:
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": _PROTO_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self.name, "version": "1.0.0"},
                },
            }

        if method == "tools/list":
            tools_out = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "inputSchema": t["inputSchema"],
                }
                for t in self._tools.values()
            ]
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools_out}}

        if method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            if tool_name not in self._tools:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Tool '{tool_name}' not found",
                    },
                }
            handler = self._tools[tool_name]["handler"]
            call_kwargs = dict(args)
            if "ctx" in inspect.signature(handler).parameters:
                call_kwargs["ctx"] = MCPContext(lifespan=lifespan_data)
            try:
                result = await handler(**call_kwargs)
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(e)},
                }
            if not isinstance(result, str):
                result = json.dumps(result)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": result}]},
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    # --- stdio transport ---

    async def run_stdio(self):
        "Run the server reading JSON-RPC from stdin and writing responses to stdout."
        if self._lifespan_factory:
            async with self._lifespan_factory(self) as lifespan_data:
                await self._stdio_loop(lifespan_data or {})
        else:
            await self._stdio_loop({})

    async def _stdio_loop(self, lifespan_data: dict):
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                response = await self._handle(request, lifespan_data)
            except Exception as e:
                req_id = request.get("id")
                response = (
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32603, "message": str(e)},
                    }
                    if req_id is not None
                    else None
                )
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

    def run(self):
        "Start the server in stdio mode (blocking)."
        asyncio.run(self.run_stdio())

    # --- HTTP transport ---

    def http_app(self):
        "Return a FastAPI ASGI app serving the MCP endpoint at POST /mcp. Requires [mcp] extra."
        _require("fastapi", FastAPI)

        server = self

        @asynccontextmanager
        async def _lifespan(app: FastAPI):
            if server._lifespan_factory:
                async with server._lifespan_factory(server) as data:
                    app.state.lifespan_data = data or {}
                    yield
            else:
                app.state.lifespan_data = {}
                yield

        app = FastAPI(lifespan=_lifespan)

        @app.post("/mcp")
        async def mcp_endpoint(request: Request):
            payload = await request.json()
            ld = request.app.state.lifespan_data

            async def _stream():
                response = await server._handle(payload, ld)
                if response is not None:
                    yield f"data: {json.dumps(response)}\n\n"

            return StreamingResponse(_stream(), media_type="text/event-stream")

        return app
