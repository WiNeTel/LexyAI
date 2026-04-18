"""
Lexy AI - MCP Tool Bridge.

Bridges tools exposed by MCP servers into Lexy's ToolRegistry so the
LLM agent can discover and call them transparently.

Each MCP tool is registered as ``{prefix}{server}_{tool}`` (e.g.
``mcp_filesystem_read_file``).  When the LLM invokes that tool, the
bridge forwards the call to the MCP server via its ``MCPClient``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lexy_core.utils.logging import get_logger

if TYPE_CHECKING:
    from lexy_core.plugin_system.plugin_api import PluginAPI
    from .mcp_client import MCPClient

log = get_logger(module="mcp_tool_bridge")


class MCPToolBridge:
    """
    Translates MCP server tools into Lexy tool registrations.

    Keeps track of which tools were registered per server so they can be
    cleanly removed when a server disconnects.
    """

    def __init__(self, api: "PluginAPI", prefix: str = "mcp_") -> None:
        self._api = api
        self._prefix = prefix
        # server_name -> [lexy_tool_name, ...]
        self._bridged_tools: dict[str, list[str]] = {}

    # ─── Bridge / Unbridge ───────────────────────────────────────

    async def bridge_server_tools(
        self,
        server_name: str,
        client: "MCPClient",
    ) -> int:
        """
        Register all tools from an MCP server as Lexy tools.

        Returns the number of tools successfully registered.
        """
        tools = client.tools
        if not tools:
            log.info("mcp.no_tools", server=server_name)
            return 0

        registered: list[str] = []

        for tool_def in tools:
            mcp_tool_name: str = tool_def.get("name", "")
            if not mcp_tool_name:
                continue

            lexy_name = f"{self._prefix}{server_name}_{mcp_tool_name}"
            description = tool_def.get("description", "")
            input_schema = tool_def.get("inputSchema", {
                "type": "object",
                "properties": {},
            })

            # Handler-Closure: wir binden client + mcp_tool_name fest ein
            handler = self._make_handler(client, mcp_tool_name, server_name)

            self._api.register_tool(
                name=lexy_name,
                handler=handler,
                description=f"[MCP:{server_name}] {description}",
                schema=input_schema,
            )
            registered.append(lexy_name)

        self._bridged_tools[server_name] = registered
        log.info(
            "mcp.tools_bridged",
            server=server_name,
            count=len(registered),
            tools=[t.removeprefix(self._prefix) for t in registered],
        )
        return len(registered)

    def unbridge_server_tools(self, server_name: str) -> int:
        """
        Unregister all tools that were bridged from an MCP server.

        Returns the number of tools successfully removed.
        """
        tool_names = self._bridged_tools.pop(server_name, [])
        if not tool_names:
            return 0

        registry = self._api.get_tool_registry()
        count = 0
        if registry is not None:
            for name in tool_names:
                try:
                    if registry.unregister(name):
                        count += 1
                except Exception as exc:
                    log.warning(
                        "mcp.unregister_failed",
                        tool=name,
                        error=str(exc),
                    )
        else:
            log.warning("mcp.no_registry", server=server_name)

        log.info("mcp.tools_unbridged", server=server_name, count=count)
        return count

    def unbridge_all(self) -> int:
        """Unbridge tools from every server. Returns total count."""
        total = 0
        server_names = list(self._bridged_tools.keys())
        for name in server_names:
            total += self.unbridge_server_tools(name)
        return total

    def get_bridged_tools(self, server_name: str) -> list[str]:
        """Return the Lexy tool names bridged from a server."""
        return list(self._bridged_tools.get(server_name, []))

    def get_all_bridged_tools(self) -> dict[str, list[str]]:
        """Return all bridged tools grouped by server."""
        return {k: list(v) for k, v in self._bridged_tools.items()}

    # ─── Handler Factory ─────────────────────────────────────────

    @staticmethod
    def _make_handler(
        client: "MCPClient",
        mcp_tool_name: str,
        server_name: str,
    ) -> Any:
        """
        Create an async handler closure for a single MCP tool.

        Der Handler nimmt beliebige kwargs entgegen (vom LLM), leitet
        sie an den MCP Server weiter und formatiert die Antwort.
        """

        async def handler(**kwargs: Any) -> dict[str, Any]:
            try:
                result = await client.call_tool(mcp_tool_name, kwargs)
                return _format_mcp_result(result, server_name, mcp_tool_name)
            except Exception as exc:
                log.error(
                    "mcp.tool_call_failed",
                    server=server_name,
                    tool=mcp_tool_name,
                    error=str(exc),
                )
                return {
                    "error": str(exc),
                    "server": server_name,
                    "tool": mcp_tool_name,
                }

        # Name und Docstring fuer Introspection
        handler.__name__ = f"mcp_{server_name}_{mcp_tool_name}"
        handler.__doc__ = f"MCP bridge handler for {server_name}/{mcp_tool_name}"
        return handler


def _format_mcp_result(
    result: dict[str, Any],
    server_name: str,
    tool_name: str,
) -> dict[str, Any]:
    """
    Convert an MCP tool/call result to a flat Lexy-friendly dict.

    MCP returns ``{"content": [{"type": "text", "text": "..."}, ...]}``
    plus optional ``isError`` flag. We flatten the content list into a
    single result string.
    """
    is_error = result.get("isError", False)
    content = result.get("content", [])

    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            item_type = item.get("type", "")
            if item_type == "text":
                parts.append(item.get("text", ""))
            elif item_type == "image":
                mime = item.get("mimeType", "unknown")
                parts.append(f"[Image: {mime}]")
            elif item_type == "resource":
                uri = item.get("resource", {}).get("uri", "unknown")
                parts.append(f"[Resource: {uri}]")
            else:
                # Unbekannter Content-Type: als JSON-String
                parts.append(str(item))
        elif isinstance(item, str):
            parts.append(item)

    text_result = "\n".join(parts) if parts else str(result)

    output: dict[str, Any] = {"result": text_result}
    if is_error:
        output["error"] = text_result
        output["is_error"] = True
    output["server"] = server_name
    output["tool"] = tool_name
    return output
