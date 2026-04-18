"""
Lexy AI - Async MCP Client.

Implements the Model Context Protocol (MCP) JSON-RPC 2.0 transport layer
directly over stdio (subprocess) and SSE (HTTP). No external ``mcp`` package
needed -- we speak the wire protocol ourselves.

Protocol reference: https://spec.modelcontextprotocol.io/
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx

from lexy_core.utils.logging import get_logger

log = get_logger(module="mcp_client")


class MCPError(Exception):
    """Error returned by an MCP server."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"MCP error {code}: {message}")


class MCPClient:
    """
    MCP client implementing JSON-RPC 2.0 over stdio or SSE.

    Usage::

        client = MCPClient("filesystem")
        await client.connect_stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
        tools = client.tools
        result = await client.call_tool("read_file", {"path": "/tmp/hello.txt"})
        await client.disconnect()
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._transport: str = ""  # "stdio" or "sse"
        self._process: asyncio.subprocess.Process | None = None
        self._sse_client: httpx.AsyncClient | None = None
        self._sse_url: str = ""
        self._sse_post_url: str = ""
        self._request_id: int = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._connected: bool = False
        self._server_capabilities: dict[str, Any] = {}
        self._server_info: dict[str, Any] = {}
        self._tools: list[dict[str, Any]] = []
        self._resources: list[dict[str, Any]] = []

    # ─── Properties ──────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    @property
    def resources(self) -> list[dict[str, Any]]:
        return list(self._resources)

    @property
    def server_capabilities(self) -> dict[str, Any]:
        return dict(self._server_capabilities)

    @property
    def server_info(self) -> dict[str, Any]:
        return dict(self._server_info)

    # ─── Connect ─────────────────────────────────────────────────

    async def connect_stdio(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Start an MCP server as subprocess, communicate via stdin/stdout JSON-RPC."""
        full_env = {**os.environ, **(env or {})}
        resolved_args = args or []

        log.info(
            "mcp.connecting",
            name=self.name,
            transport="stdio",
            command=command,
            args=resolved_args,
        )

        self._process = await asyncio.create_subprocess_exec(
            command,
            *resolved_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
        )
        self._transport = "stdio"
        self._reader_task = asyncio.create_task(
            self._read_stdio_loop(),
            name=f"mcp.{self.name}.reader",
        )
        # Hintergrund-Task fuer stderr-Logging
        asyncio.create_task(
            self._read_stderr_loop(),
            name=f"mcp.{self.name}.stderr",
        )

        await self._initialize()
        self._connected = True
        log.info(
            "mcp.connected",
            name=self.name,
            transport="stdio",
            tools=len(self._tools),
            resources=len(self._resources),
        )

    async def connect_sse(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Connect to an SSE-based MCP server via HTTP POST for requests."""
        self._sse_url = url
        # MCP SSE: the /sse endpoint streams events, /message receives requests
        if url.endswith("/sse"):
            self._sse_post_url = url[: -len("/sse")] + "/message"
        else:
            self._sse_post_url = url.rstrip("/") + "/message"

        log.info(
            "mcp.connecting",
            name=self.name,
            transport="sse",
            url=url,
        )

        self._sse_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            headers=headers or {},
        )
        self._transport = "sse"

        await self._initialize()
        self._connected = True
        log.info(
            "mcp.connected",
            name=self.name,
            transport="sse",
            tools=len(self._tools),
            resources=len(self._resources),
        )

    # ─── Public API ──────────────────────────────────────────────

    async def list_tools(self) -> list[dict[str, Any]]:
        """Fetch the current tool list from the server."""
        if not self._connected:
            return []
        result = await self._send_request("tools/list", {})
        self._tools = result.get("tools", [])
        return list(self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool on the MCP server and return the result."""
        log.debug("mcp.call_tool", server=self.name, tool=name)
        result = await self._send_request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        return result

    async def list_resources(self) -> list[dict[str, Any]]:
        """Fetch the current resource list from the server."""
        if not self._connected:
            return []
        result = await self._send_request("resources/list", {})
        self._resources = result.get("resources", [])
        return list(self._resources)

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """Read a resource by URI."""
        result = await self._send_request("resources/read", {"uri": uri})
        return result

    # ─── Disconnect ──────────────────────────────────────────────

    async def disconnect(self) -> None:
        """Disconnect from the server, clean up all resources."""
        self._connected = False

        # Reader-Task stoppen
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None

        # Subprocess beenden
        if self._process is not None:
            try:
                if self._process.stdin is not None:
                    self._process.stdin.close()
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            except Exception as exc:
                log.warning("mcp.disconnect_error", name=self.name, error=str(exc))
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

        # SSE Client schliessen
        if self._sse_client is not None:
            await self._sse_client.aclose()
            self._sse_client = None

        # Alle wartenden Requests abbrechen
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ConnectionError("MCP client disconnected"))
        self._pending.clear()

        self._tools.clear()
        self._resources.clear()
        log.info("mcp.disconnected", name=self.name)

    # ─── Protocol Internals ──────────────────────────────────────

    async def _initialize(self) -> dict[str, Any]:
        """Perform the MCP initialize/initialized handshake."""
        result = await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"roots": {"listChanged": False}},
                "clientInfo": {"name": "lexy-mcp-bridge", "version": "1.0.0"},
            },
        )
        self._server_capabilities = result.get("capabilities", {})
        self._server_info = result.get("serverInfo", {})

        # Initialized-Notification (kein Response erwartet)
        await self._send_notification("notifications/initialized", {})

        # Tools holen, falls der Server welche anbietet
        if self._server_capabilities.get("tools"):
            try:
                tools_result = await self._send_request("tools/list", {})
                self._tools = tools_result.get("tools", [])
            except Exception as exc:
                log.warning("mcp.tools_list_failed", name=self.name, error=str(exc))

        # Resources holen, falls vorhanden
        if self._server_capabilities.get("resources"):
            try:
                res_result = await self._send_request("resources/list", {})
                self._resources = res_result.get("resources", [])
            except Exception as exc:
                log.warning("mcp.resources_list_failed", name=self.name, error=str(exc))

        return result

    async def _send_request(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a JSON-RPC 2.0 request and await the response."""
        self._request_id += 1
        req_id = self._request_id

        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[req_id] = future

        try:
            await self._send_message(message)
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise TimeoutError(
                f"MCP request '{method}' timed out after 30s (server={self.name})"
            ) from None
        finally:
            self._pending.pop(req_id, None)

    async def _send_notification(
        self,
        method: str,
        params: dict[str, Any],
    ) -> None:
        """Send a JSON-RPC 2.0 notification (no id, no response expected)."""
        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await self._send_message(message)

    async def _send_message(self, message: dict[str, Any]) -> None:
        """Serialize and send a JSON-RPC message via the active transport."""
        data = json.dumps(message, separators=(",", ":"))

        if self._transport == "stdio":
            if self._process is None or self._process.stdin is None:
                raise ConnectionError("stdio process not available")
            self._process.stdin.write((data + "\n").encode("utf-8"))
            await self._process.stdin.drain()

        elif self._transport == "sse":
            if self._sse_client is None:
                raise ConnectionError("SSE client not available")
            resp = await self._sse_client.post(
                self._sse_post_url,
                content=data,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                # SSE POST returns the JSON-RPC response directly
                try:
                    response = resp.json()
                    self._handle_response(response)
                except json.JSONDecodeError:
                    pass
            elif resp.status_code == 202:
                # Accepted: response kommt via SSE stream
                pass
            else:
                log.warning(
                    "mcp.sse_post_error",
                    name=self.name,
                    status=resp.status_code,
                    body=resp.text[:200],
                )
        else:
            raise ConnectionError(f"Unknown transport: {self._transport}")

    # ─── Reader Loops ────────────────────────────────────────────

    async def _read_stdio_loop(self) -> None:
        """Continuously read JSON-RPC responses from stdout."""
        try:
            while self._process is not None and self._process.stdout is not None:
                line = await self._process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    response = json.loads(text)
                    self._handle_response(response)
                except json.JSONDecodeError:
                    # Server kann auch Nicht-JSON-Zeilen schreiben (z.B. Logs)
                    log.debug("mcp.non_json_stdout", name=self.name, line=text[:200])
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("mcp.reader_error", name=self.name, error=str(exc))
            self._connected = False

    async def _read_stderr_loop(self) -> None:
        """Log stderr output from the subprocess (diagnostics only)."""
        try:
            while self._process is not None and self._process.stderr is not None:
                line = await self._process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    log.debug("mcp.server_stderr", name=self.name, line=text[:300])
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    # ─── Response Handling ───────────────────────────────────────

    def _handle_response(self, response: dict[str, Any]) -> None:
        """Dispatch a JSON-RPC response to its pending future."""
        # Notifications vom Server (kein "id")
        if "id" not in response and "method" in response:
            log.debug(
                "mcp.notification",
                name=self.name,
                method=response["method"],
            )
            return

        req_id = response.get("id")
        if req_id is None:
            return

        future = self._pending.get(req_id)
        if future is None or future.done():
            return

        error = response.get("error")
        if error is not None:
            code = error.get("code", -1)
            message = error.get("message", "Unknown MCP error")
            data = error.get("data")
            future.set_exception(MCPError(code=code, message=message, data=data))
        else:
            future.set_result(response.get("result", {}))
