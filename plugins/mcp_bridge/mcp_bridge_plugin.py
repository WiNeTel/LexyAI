"""
Lexy AI - MCP Bridge Plugin.

Connects to external MCP (Model Context Protocol) servers and exposes
their tools as native Lexy tools that the LLM agent can call.

Supports:
  - stdio transport (subprocess): ``npx -y @server/name`` etc.
  - SSE transport (HTTP): ``http://host:port/sse``
  - Auto-connect on enable from config
  - Management tools: list/connect/disconnect servers
  - WebSocket handlers for frontend control
  - Dynamic tool bridging: MCP tools appear as Lexy tools

Config (plugins.yaml or plugin.yaml defaults):

.. code-block:: yaml

   mcp_bridge:
     auto_connect: true
     tool_prefix: "mcp_"
     reconnect_interval: 30
     servers:
       - name: filesystem
         transport: stdio
         command: npx
         args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
       - name: remote_api
         transport: sse
         url: "http://localhost:3001/sse"
"""

from __future__ import annotations

import asyncio
from typing import Any

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

from .mcp_registry import MCPRegistry
from .mcp_tool_bridge import MCPToolBridge

log = get_logger(module="mcp_bridge_plugin")

# ─── Tool Schemas ────────────────────────────────────────────────

SCHEMA_LIST_SERVERS: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
}

SCHEMA_CONNECT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Unique name for the MCP server",
        },
        "transport": {
            "type": "string",
            "enum": ["stdio", "sse"],
            "description": "Transport type (default: stdio)",
        },
        "command": {
            "type": "string",
            "description": "For stdio: executable to run (e.g. 'npx', 'python')",
        },
        "args": {
            "type": "array",
            "items": {"type": "string"},
            "description": "For stdio: command-line arguments",
        },
        "url": {
            "type": "string",
            "description": "For SSE: the server endpoint URL",
        },
        "env": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "For stdio: extra environment variables",
        },
        "headers": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "For SSE: extra HTTP headers",
        },
    },
    "required": ["name"],
}

SCHEMA_DISCONNECT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Name of the MCP server to disconnect",
        },
    },
    "required": ["name"],
}

SCHEMA_SERVER_TOOLS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Name of the MCP server",
        },
    },
    "required": ["name"],
}

SCHEMA_SERVER_RESOURCES: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Name of the MCP server",
        },
    },
    "required": ["name"],
}

SCHEMA_READ_RESOURCE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "server": {
            "type": "string",
            "description": "Name of the MCP server",
        },
        "uri": {
            "type": "string",
            "description": "Resource URI to read",
        },
    },
    "required": ["server", "uri"],
}


class MCPBridgePlugin(BasePlugin):
    """
    MCP Bridge — connect external MCP servers and use their tools in Lexy.

    Lifecycle:
      on_load  → read config, create registry + bridge
      on_enable → register management tools + WS handlers, auto-connect
      on_disable → disconnect all servers, unbridge tools
    """

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._registry: MCPRegistry | None = None
        self._bridge: MCPToolBridge | None = None
        self._auto_connect: bool = True
        self._tool_prefix: str = "mcp_"
        self._reconnect_interval: int = 30
        self._server_configs: list[dict[str, Any]] = []
        self._reconnect_task: asyncio.Task[None] | None = None

    # ─── Lifecycle ───────────────────────────────────────────────

    async def on_load(self) -> None:
        """Read config, create MCPRegistry and MCPToolBridge."""
        config = self.api.get_config()

        self._auto_connect = bool(config.get("auto_connect", True))
        self._tool_prefix = str(config.get("tool_prefix", "mcp_"))
        self._reconnect_interval = int(config.get("reconnect_interval", 30))
        self._server_configs = list(config.get("servers", []))

        self._registry = MCPRegistry()
        self._bridge = MCPToolBridge(api=self.api, prefix=self._tool_prefix)

        log.info(
            "mcp_bridge.loaded",
            auto_connect=self._auto_connect,
            prefix=self._tool_prefix,
            servers_configured=len(self._server_configs),
        )

    async def on_enable(self) -> None:
        """Register management tools, WS handlers, and auto-connect servers."""
        assert self._registry is not None
        assert self._bridge is not None

        # -- Management Tools registrieren --
        self.api.register_tool(
            name="mcp_list_servers",
            handler=self._tool_list_servers,
            description=(
                "List all connected MCP servers with their status, "
                "transport type, and tool/resource counts."
            ),
            schema=SCHEMA_LIST_SERVERS,
        )

        self.api.register_tool(
            name="mcp_connect",
            handler=self._tool_connect,
            description=(
                "Connect to a new MCP server. Supports stdio (subprocess) "
                "and SSE (HTTP) transports. Tools are automatically bridged."
            ),
            schema=SCHEMA_CONNECT,
        )

        self.api.register_tool(
            name="mcp_disconnect",
            handler=self._tool_disconnect,
            description="Disconnect from an MCP server and remove its tools.",
            schema=SCHEMA_DISCONNECT,
        )

        self.api.register_tool(
            name="mcp_server_tools",
            handler=self._tool_server_tools,
            description="List all tools provided by a specific MCP server.",
            schema=SCHEMA_SERVER_TOOLS,
        )

        self.api.register_tool(
            name="mcp_server_resources",
            handler=self._tool_server_resources,
            description="List all resources provided by a specific MCP server.",
            schema=SCHEMA_SERVER_RESOURCES,
        )

        self.api.register_tool(
            name="mcp_read_resource",
            handler=self._tool_read_resource,
            description="Read a resource from an MCP server by its URI.",
            schema=SCHEMA_READ_RESOURCE,
        )

        # -- WebSocket Handler registrieren --
        self.api.register_ws_handler("mcp_list_servers", self._ws_list_servers)
        self.api.register_ws_handler("mcp_connect_server", self._ws_connect_server)
        self.api.register_ws_handler("mcp_disconnect_server", self._ws_disconnect_server)

        # -- Auto-Connect --
        if self._auto_connect and self._server_configs:
            await self._auto_connect_servers()

        # -- Reconnect-Task starten --
        if self._reconnect_interval > 0:
            self._reconnect_task = asyncio.create_task(
                self._reconnect_loop(),
                name="mcp_bridge.reconnect",
            )

        await self.api.emit("mcp_bridge.enabled", {
            "servers": self._registry.list_servers(),
        })

        log.info("mcp_bridge.enabled")

    async def on_disable(self) -> None:
        """Disconnect all servers and clean up."""
        # Reconnect-Task stoppen
        if self._reconnect_task is not None and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reconnect_task = None

        # Alle Tools entfernen
        if self._bridge is not None:
            self._bridge.unbridge_all()

        # Alle Server trennen
        if self._registry is not None:
            await self._registry.disconnect_all()

        await self.api.emit("mcp_bridge.disabled", {})
        log.info("mcp_bridge.disabled")

    # ─── Auto-Connect ────────────────────────────────────────────

    async def _auto_connect_servers(self) -> None:
        """Connect all servers from config on startup."""
        assert self._registry is not None
        assert self._bridge is not None

        for server_cfg in self._server_configs:
            name = server_cfg.get("name", "unknown")
            try:
                entry = await self._registry.connect_server(server_cfg)
                if entry.status == "connected" and entry.client is not None:
                    count = await self._bridge.bridge_server_tools(
                        entry.name, entry.client,
                    )
                    log.info(
                        "mcp.auto_connected",
                        server=entry.name,
                        tools=count,
                    )
                else:
                    log.warning(
                        "mcp.auto_connect_failed",
                        server=name,
                        status=entry.status,
                        error=entry.error,
                    )
            except Exception as exc:
                log.error(
                    "mcp.auto_connect_error",
                    server=name,
                    error=str(exc),
                )

    # ─── Reconnect Loop ─────────────────────────────────────────

    async def _reconnect_loop(self) -> None:
        """Periodically try to reconnect servers in error state."""
        assert self._registry is not None
        assert self._bridge is not None

        try:
            while True:
                await asyncio.sleep(self._reconnect_interval)
                for info in self._registry.list_servers():
                    if info["status"] != "error":
                        continue

                    name = info["name"]
                    log.info("mcp.reconnecting", server=name)
                    success = await self._registry.reconnect_server(name)
                    if success:
                        entry = self._registry.get_server(name)
                        if entry is not None and entry.client is not None:
                            count = await self._bridge.bridge_server_tools(
                                name, entry.client,
                            )
                            log.info(
                                "mcp.reconnected",
                                server=name,
                                tools=count,
                            )
                            await self._broadcast_server_update()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("mcp.reconnect_loop_error", error=str(exc))

    # ─── Tool Handlers ───────────────────────────────────────────

    async def _tool_list_servers(self) -> dict[str, Any]:
        """List all connected MCP servers."""
        assert self._registry is not None
        servers = self._registry.list_servers()
        bridged = (
            self._bridge.get_all_bridged_tools()
            if self._bridge is not None
            else {}
        )
        return {
            "servers": servers,
            "total": len(servers),
            "bridged_tools": {k: len(v) for k, v in bridged.items()},
        }

    async def _tool_connect(
        self,
        name: str,
        transport: str = "stdio",
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        env: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Connect to a new MCP server and bridge its tools."""
        assert self._registry is not None
        assert self._bridge is not None

        config: dict[str, Any] = {"name": name, "transport": transport}
        if command is not None:
            config["command"] = command
        if args is not None:
            config["args"] = args
        if url is not None:
            config["url"] = url
        if env is not None:
            config["env"] = env
        if headers is not None:
            config["headers"] = headers

        entry = await self._registry.connect_server(config)

        if entry.status == "connected" and entry.client is not None:
            count = await self._bridge.bridge_server_tools(name, entry.client)
            await self._broadcast_server_update()
            return {
                "status": "connected",
                "server": name,
                "tools_bridged": count,
                "resources": entry.resources_count,
            }

        return {
            "status": entry.status,
            "server": name,
            "error": entry.error,
        }

    async def _tool_disconnect(self, name: str) -> dict[str, Any]:
        """Disconnect from an MCP server."""
        assert self._registry is not None
        assert self._bridge is not None

        # Tools zuerst entfernen
        unbridged = self._bridge.unbridge_server_tools(name)
        disconnected = await self._registry.disconnect_server(name)

        await self._broadcast_server_update()
        return {
            "disconnected": disconnected,
            "server": name,
            "tools_removed": unbridged,
        }

    async def _tool_server_tools(self, name: str) -> dict[str, Any]:
        """List tools of a specific MCP server."""
        assert self._registry is not None

        entry = self._registry.get_server(name)
        if entry is None or entry.client is None:
            return {"error": f"Server '{name}' not connected", "server": name}

        tools = entry.client.tools
        bridged = (
            self._bridge.get_bridged_tools(name)
            if self._bridge is not None
            else []
        )

        return {
            "server": name,
            "tools": [
                {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "lexy_name": next(
                        (b for b in bridged if b.endswith(f"_{t.get('name', '')}"))
                        , None,
                    ),
                }
                for t in tools
            ],
            "count": len(tools),
        }

    async def _tool_server_resources(self, name: str) -> dict[str, Any]:
        """List resources of a specific MCP server."""
        assert self._registry is not None

        entry = self._registry.get_server(name)
        if entry is None or entry.client is None:
            return {"error": f"Server '{name}' not connected", "server": name}

        resources = entry.client.resources
        return {
            "server": name,
            "resources": [
                {
                    "uri": r.get("uri", ""),
                    "name": r.get("name", ""),
                    "description": r.get("description", ""),
                    "mimeType": r.get("mimeType", ""),
                }
                for r in resources
            ],
            "count": len(resources),
        }

    async def _tool_read_resource(
        self,
        server: str,
        uri: str,
    ) -> dict[str, Any]:
        """Read a resource from an MCP server."""
        assert self._registry is not None

        entry = self._registry.get_server(server)
        if entry is None or entry.client is None:
            return {"error": f"Server '{server}' not connected", "server": server}

        try:
            result = await entry.client.read_resource(uri)
            contents = result.get("contents", [])
            texts: list[str] = []
            for item in contents:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text is not None:
                        texts.append(str(text))
                    blob = item.get("blob")
                    if blob is not None:
                        texts.append(f"[Binary data: {len(str(blob))} chars]")
            return {
                "server": server,
                "uri": uri,
                "content": "\n".join(texts) if texts else str(result),
            }
        except Exception as exc:
            return {"error": str(exc), "server": server, "uri": uri}

    # ─── WebSocket Handlers ──────────────────────────────────────

    async def _ws_list_servers(
        self,
        data: dict[str, Any],
        client_id: str,
    ) -> dict[str, Any]:
        """WS: Return the current server list."""
        assert self._registry is not None
        return {
            "type": "mcp_servers",
            "servers": self._registry.list_servers(),
        }

    async def _ws_connect_server(
        self,
        data: dict[str, Any],
        client_id: str,
    ) -> dict[str, Any]:
        """WS: Connect a server and bridge its tools."""
        config = data.get("config", {})
        if not config.get("name"):
            return {"type": "mcp_error", "error": "Missing 'name' in server config"}

        result = await self._tool_connect(**config)
        return {"type": "mcp_connect_result", **result}

    async def _ws_disconnect_server(
        self,
        data: dict[str, Any],
        client_id: str,
    ) -> dict[str, Any]:
        """WS: Disconnect a server."""
        name = data.get("name", "")
        if not name:
            return {"type": "mcp_error", "error": "Missing 'name'"}

        result = await self._tool_disconnect(name=name)
        return {"type": "mcp_disconnect_result", **result}

    # ─── Helpers ─────────────────────────────────────────────────

    async def _broadcast_server_update(self) -> None:
        """Broadcast the current server list to all WS clients."""
        assert self._registry is not None
        try:
            await self.api.ws_broadcast({
                "type": "mcp_servers_updated",
                "servers": self._registry.list_servers(),
            })
        except Exception as exc:
            log.debug("mcp.broadcast_failed", error=str(exc))
