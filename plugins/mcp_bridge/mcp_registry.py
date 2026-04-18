"""
Lexy AI - MCP Server Registry.

Manages the lifecycle of connected MCP servers: connect, disconnect,
reconnect, and status queries. Each server is tracked as an
``MCPServerEntry`` with its client instance and connection metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from lexy_core.utils.logging import get_logger

from .mcp_client import MCPClient

log = get_logger(module="mcp_registry")


@dataclass
class MCPServerEntry:
    """Tracks a single MCP server connection."""

    name: str
    config: dict[str, Any]
    client: MCPClient | None = None
    status: str = "disconnected"  # disconnected | connecting | connected | error
    tools_count: int = 0
    resources_count: int = 0
    error: str = ""
    connected_at: float | None = None
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for WS/API responses."""
        return {
            "name": self.name,
            "status": self.status,
            "transport": self.config.get("transport", "stdio"),
            "tools_count": self.tools_count,
            "resources_count": self.resources_count,
            "error": self.error,
            "connected_at": self.connected_at,
            "retry_count": self.retry_count,
        }


class MCPRegistry:
    """
    Registry of MCP server connections.

    Handles connecting, disconnecting, reconnecting, and listing servers.
    Thread-safe: all mutations are async and non-reentrant.
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerEntry] = {}
        self._connect_lock = __import__("asyncio").Lock()

    # ─── Connect ─────────────────────────────────────────────────

    async def connect_server(self, config: dict[str, Any]) -> MCPServerEntry:
        """
        Connect to an MCP server based on its config dict.

        Config keys:
          - name (str, required): unique identifier
          - transport (str): "stdio" (default) or "sse"
          - command (str): for stdio, executable to run
          - args (list[str]): for stdio, command-line arguments
          - env (dict[str, str]): for stdio, extra environment vars
          - url (str): for sse, the SSE endpoint URL
          - headers (dict[str, str]): for sse, extra HTTP headers
        """
        name = config["name"]
        transport = config.get("transport", "stdio")

        async with self._connect_lock:
            # Falls bereits verbunden, erst trennen
            existing = self._servers.get(name)
            if existing is not None and existing.client is not None:
                if existing.status == "connected":
                    log.info("mcp.already_connected", name=name)
                    return existing
                # Halb-verbunden → aufräumen
                await self._safe_disconnect_client(existing.client)

            entry = MCPServerEntry(name=name, config=config, status="connecting")
            self._servers[name] = entry

            client = MCPClient(name)
            try:
                if transport == "stdio":
                    command = config.get("command", "")
                    if not command:
                        raise ValueError("stdio transport requires 'command' in config")
                    await client.connect_stdio(
                        command=command,
                        args=config.get("args", []),
                        env=config.get("env"),
                    )
                elif transport == "sse":
                    url = config.get("url", "")
                    if not url:
                        raise ValueError("sse transport requires 'url' in config")
                    await client.connect_sse(
                        url=url,
                        headers=config.get("headers"),
                    )
                else:
                    raise ValueError(f"Unknown transport: {transport!r}")

                entry.client = client
                entry.status = "connected"
                entry.tools_count = len(client.tools)
                entry.resources_count = len(client.resources)
                entry.connected_at = time.time()
                entry.error = ""
                entry.retry_count = 0

                log.info(
                    "mcp.server_connected",
                    name=name,
                    transport=transport,
                    tools=entry.tools_count,
                    resources=entry.resources_count,
                )

            except Exception as exc:
                entry.status = "error"
                entry.error = str(exc)
                entry.client = None
                entry.retry_count += 1
                await self._safe_disconnect_client(client)
                log.error(
                    "mcp.connect_failed",
                    name=name,
                    transport=transport,
                    error=str(exc),
                )

            return entry

    # ─── Disconnect ──────────────────────────────────────────────

    async def disconnect_server(self, name: str) -> bool:
        """Disconnect a single server by name. Returns True if it was connected."""
        entry = self._servers.get(name)
        if entry is None:
            log.warning("mcp.disconnect_unknown", name=name)
            return False

        if entry.client is not None:
            await self._safe_disconnect_client(entry.client)
            entry.client = None

        entry.status = "disconnected"
        entry.tools_count = 0
        entry.resources_count = 0
        entry.connected_at = None
        entry.error = ""
        log.info("mcp.server_disconnected", name=name)
        return True

    async def disconnect_all(self) -> None:
        """Disconnect every registered server."""
        names = list(self._servers.keys())
        for name in names:
            await self.disconnect_server(name)
        log.info("mcp.all_disconnected", count=len(names))

    # ─── Reconnect ───────────────────────────────────────────────

    async def reconnect_server(self, name: str) -> bool:
        """Disconnect and reconnect a server. Returns True on success."""
        entry = self._servers.get(name)
        if entry is None:
            log.warning("mcp.reconnect_unknown", name=name)
            return False

        config = entry.config
        await self.disconnect_server(name)
        new_entry = await self.connect_server(config)
        return new_entry.status == "connected"

    # ─── Queries ─────────────────────────────────────────────────

    def get_server(self, name: str) -> MCPServerEntry | None:
        """Get a server entry by name (or None)."""
        return self._servers.get(name)

    def list_servers(self) -> list[dict[str, Any]]:
        """Return serialized info for all registered servers."""
        return [entry.to_dict() for entry in self._servers.values()]

    def get_connected_servers(self) -> list[MCPServerEntry]:
        """Return only entries with status 'connected'."""
        return [
            entry
            for entry in self._servers.values()
            if entry.status == "connected" and entry.client is not None
        ]

    def remove_server(self, name: str) -> bool:
        """Remove a server entry entirely (must be disconnected first)."""
        entry = self._servers.get(name)
        if entry is None:
            return False
        if entry.status == "connected":
            log.warning("mcp.remove_still_connected", name=name)
            return False
        del self._servers[name]
        return True

    # ─── Internal ────────────────────────────────────────────────

    @staticmethod
    async def _safe_disconnect_client(client: MCPClient) -> None:
        """Disconnect a client, swallowing any errors."""
        try:
            await client.disconnect()
        except Exception as exc:
            log.warning("mcp.disconnect_error", name=client.name, error=str(exc))
