"""Tests for the MCP Bridge plugin -- client, registry, tool bridge.

Covers:
* MCPClient: JSON-RPC message construction, properties, request ID
* MCPError exception
* MCPRegistry: add/remove/list servers, MCPServerEntry
* MCPToolBridge: format_result, tool name prefixing, bridge/unbridge
* Handler factory closure
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.mcp_bridge.mcp_client import MCPClient, MCPError
from plugins.mcp_bridge.mcp_registry import MCPRegistry, MCPServerEntry
from plugins.mcp_bridge.mcp_tool_bridge import MCPToolBridge, _format_mcp_result


# ─── MCPError ─────────────────────────────────────────────────────────────


class TestMCPError:
    def test_stores_code_and_message(self) -> None:
        err = MCPError(code=-32600, message="Invalid Request")
        assert err.code == -32600
        assert err.message == "Invalid Request"
        assert err.data is None

    def test_stores_data(self) -> None:
        err = MCPError(code=-32601, message="Method not found", data={"detail": "x"})
        assert err.data == {"detail": "x"}

    def test_str_representation(self) -> None:
        err = MCPError(code=42, message="Custom error")
        assert "42" in str(err)
        assert "Custom error" in str(err)


# ─── MCPClient ────────────────────────────────────────────────────────────


class TestMCPClient:
    def test_init_defaults(self) -> None:
        client = MCPClient("test-server")
        assert client.name == "test-server"
        assert client.connected is False
        assert client.tools == []
        assert client.resources == []
        assert client.server_capabilities == {}
        assert client.server_info == {}

    def test_transport_not_set_initially(self) -> None:
        client = MCPClient("srv")
        assert client._transport == ""

    def test_request_id_starts_at_zero(self) -> None:
        client = MCPClient("srv")
        assert client._request_id == 0

    def test_tools_returns_copy(self) -> None:
        client = MCPClient("srv")
        client._tools = [{"name": "tool1"}]
        tools = client.tools
        tools.append({"name": "tool2"})
        assert len(client._tools) == 1  # Original unchanged

    def test_resources_returns_copy(self) -> None:
        client = MCPClient("srv")
        client._resources = [{"uri": "file:///a"}]
        resources = client.resources
        resources.clear()
        assert len(client._resources) == 1

    def test_server_capabilities_returns_copy(self) -> None:
        client = MCPClient("srv")
        client._server_capabilities = {"tools": True}
        caps = client.server_capabilities
        caps["extra"] = True
        assert "extra" not in client._server_capabilities

    def test_handle_response_resolves_future(self) -> None:
        client = MCPClient("srv")
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        client._pending[1] = future

        client._handle_response({"id": 1, "result": {"data": "ok"}})
        assert future.result() == {"data": "ok"}
        loop.close()

    def test_handle_response_error_sets_exception(self) -> None:
        client = MCPClient("srv")
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        client._pending[2] = future

        client._handle_response({
            "id": 2,
            "error": {"code": -32600, "message": "Bad request"},
        })
        with pytest.raises(MCPError) as exc_info:
            future.result()
        assert exc_info.value.code == -32600
        loop.close()

    def test_handle_response_notification_ignored(self) -> None:
        client = MCPClient("srv")
        # Notifications have no "id" field -- should not raise
        client._handle_response({"method": "notifications/progress", "params": {}})

    def test_handle_response_unknown_id_ignored(self) -> None:
        client = MCPClient("srv")
        # No pending future for id=999 -- should not raise
        client._handle_response({"id": 999, "result": {}})

    @pytest.mark.asyncio
    async def test_list_tools_when_disconnected(self) -> None:
        client = MCPClient("srv")
        # Not connected -> returns empty list
        tools = await client.list_tools()
        assert tools == []

    @pytest.mark.asyncio
    async def test_list_resources_when_disconnected(self) -> None:
        client = MCPClient("srv")
        resources = await client.list_resources()
        assert resources == []


# ─── MCPServerEntry ───────────────────────────────────────────────────────


class TestMCPServerEntry:
    def test_defaults(self) -> None:
        entry = MCPServerEntry(name="test", config={"transport": "stdio"})
        assert entry.status == "disconnected"
        assert entry.tools_count == 0
        assert entry.resources_count == 0
        assert entry.client is None
        assert entry.error == ""
        assert entry.connected_at is None
        assert entry.retry_count == 0

    def test_to_dict(self) -> None:
        entry = MCPServerEntry(
            name="fs",
            config={"transport": "stdio", "command": "node"},
            status="connected",
            tools_count=5,
            resources_count=2,
            connected_at=1000.0,
        )
        d = entry.to_dict()
        assert d["name"] == "fs"
        assert d["status"] == "connected"
        assert d["transport"] == "stdio"
        assert d["tools_count"] == 5
        assert d["resources_count"] == 2
        assert d["connected_at"] == 1000.0

    def test_to_dict_sse_transport(self) -> None:
        entry = MCPServerEntry(
            name="remote",
            config={"transport": "sse", "url": "http://localhost:3000/sse"},
        )
        d = entry.to_dict()
        assert d["transport"] == "sse"


# ─── MCPRegistry ──────────────────────────────────────────────────────────


class TestMCPRegistry:
    def test_init_empty(self) -> None:
        registry = MCPRegistry()
        assert registry.list_servers() == []

    def test_get_nonexistent_server(self) -> None:
        registry = MCPRegistry()
        assert registry.get_server("nonexistent") is None

    def test_get_connected_servers_empty(self) -> None:
        registry = MCPRegistry()
        assert registry.get_connected_servers() == []

    def test_remove_server_nonexistent(self) -> None:
        registry = MCPRegistry()
        assert registry.remove_server("nope") is False

    def test_remove_server_connected_fails(self) -> None:
        registry = MCPRegistry()
        entry = MCPServerEntry(name="srv", config={}, status="connected")
        registry._servers["srv"] = entry
        # Should refuse to remove a connected server
        assert registry.remove_server("srv") is False

    def test_remove_server_disconnected_succeeds(self) -> None:
        registry = MCPRegistry()
        entry = MCPServerEntry(name="srv", config={}, status="disconnected")
        registry._servers["srv"] = entry
        assert registry.remove_server("srv") is True
        assert registry.get_server("srv") is None

    def test_list_servers_returns_dicts(self) -> None:
        registry = MCPRegistry()
        entry = MCPServerEntry(name="s1", config={"transport": "stdio"})
        registry._servers["s1"] = entry
        servers = registry.list_servers()
        assert len(servers) == 1
        assert servers[0]["name"] == "s1"
        assert isinstance(servers[0], dict)

    def test_get_connected_servers_filters(self) -> None:
        registry = MCPRegistry()
        connected = MCPServerEntry(
            name="c1", config={}, status="connected",
            client=MagicMock(),
        )
        disconnected = MCPServerEntry(name="d1", config={}, status="disconnected")
        registry._servers["c1"] = connected
        registry._servers["d1"] = disconnected

        result = registry.get_connected_servers()
        assert len(result) == 1
        assert result[0].name == "c1"


# ─── _format_mcp_result ──────────────────────────────────────────────────


class TestFormatMcpResult:
    def test_single_text_content(self) -> None:
        result = _format_mcp_result(
            {"content": [{"type": "text", "text": "Hello World"}]},
            "srv", "tool",
        )
        assert result["result"] == "Hello World"
        assert result["server"] == "srv"
        assert result["tool"] == "tool"

    def test_empty_content(self) -> None:
        result = _format_mcp_result({"content": []}, "srv", "tool")
        assert "result" in result

    def test_multiple_text_items(self) -> None:
        result = _format_mcp_result({
            "content": [
                {"type": "text", "text": "Line 1"},
                {"type": "text", "text": "Line 2"},
            ]
        }, "srv", "tool")
        assert "Line 1" in result["result"]
        assert "Line 2" in result["result"]

    def test_image_content(self) -> None:
        result = _format_mcp_result({
            "content": [{"type": "image", "mimeType": "image/png"}]
        }, "srv", "tool")
        assert "Image" in result["result"]
        assert "image/png" in result["result"]

    def test_resource_content(self) -> None:
        result = _format_mcp_result({
            "content": [{"type": "resource", "resource": {"uri": "file:///a.txt"}}]
        }, "srv", "tool")
        assert "Resource" in result["result"]
        assert "file:///a.txt" in result["result"]

    def test_error_flag(self) -> None:
        result = _format_mcp_result(
            {"isError": True, "content": [{"type": "text", "text": "Something failed"}]},
            "srv", "tool",
        )
        assert result["is_error"] is True
        assert "error" in result
        assert "Something failed" in result["error"]

    def test_string_content_items(self) -> None:
        result = _format_mcp_result(
            {"content": ["plain string"]},
            "srv", "tool",
        )
        assert "plain string" in result["result"]

    def test_unknown_content_type(self) -> None:
        result = _format_mcp_result(
            {"content": [{"type": "binary", "data": "AAAA"}]},
            "srv", "tool",
        )
        # Unknown types get stringified
        assert "binary" in result["result"]


# ─── MCPToolBridge ────────────────────────────────────────────────────────


class TestMCPToolBridge:
    def test_init_stores_prefix(self) -> None:
        bridge = MCPToolBridge(api=MagicMock(), prefix="mcp_")
        assert bridge._prefix == "mcp_"

    def test_custom_prefix(self) -> None:
        bridge = MCPToolBridge(api=MagicMock(), prefix="ext_")
        assert bridge._prefix == "ext_"

    def test_get_bridged_tools_empty(self) -> None:
        bridge = MCPToolBridge(api=MagicMock())
        assert bridge.get_bridged_tools("nonexistent") == []

    def test_get_all_bridged_tools_empty(self) -> None:
        bridge = MCPToolBridge(api=MagicMock())
        assert bridge.get_all_bridged_tools() == {}

    def test_unbridge_nonexistent_server(self) -> None:
        bridge = MCPToolBridge(api=MagicMock())
        count = bridge.unbridge_server_tools("nonexistent")
        assert count == 0

    def test_unbridge_all_empty(self) -> None:
        bridge = MCPToolBridge(api=MagicMock())
        total = bridge.unbridge_all()
        assert total == 0

    @pytest.mark.asyncio
    async def test_bridge_server_tools_no_tools(self) -> None:
        api = MagicMock()
        bridge = MCPToolBridge(api=api)
        client = MagicMock()
        client.tools = []

        count = await bridge.bridge_server_tools("empty_server", client)
        assert count == 0

    @pytest.mark.asyncio
    async def test_bridge_server_tools_registers_tools(self) -> None:
        api = MagicMock()
        bridge = MCPToolBridge(api=api, prefix="mcp_")
        client = MagicMock()
        client.tools = [
            {"name": "read_file", "description": "Read a file", "inputSchema": {"type": "object"}},
            {"name": "write_file", "description": "Write a file"},
        ]

        count = await bridge.bridge_server_tools("filesystem", client)
        assert count == 2

        # Verify tool names
        bridged = bridge.get_bridged_tools("filesystem")
        assert "mcp_filesystem_read_file" in bridged
        assert "mcp_filesystem_write_file" in bridged

        # Verify api.register_tool was called
        assert api.register_tool.call_count == 2

    @pytest.mark.asyncio
    async def test_bridge_skips_nameless_tools(self) -> None:
        api = MagicMock()
        bridge = MCPToolBridge(api=api, prefix="mcp_")
        client = MagicMock()
        client.tools = [
            {"name": "valid_tool", "description": "d"},
            {"name": "", "description": "nameless"},
        ]

        count = await bridge.bridge_server_tools("srv", client)
        assert count == 1

    @pytest.mark.asyncio
    async def test_make_handler_calls_client(self) -> None:
        """The handler closure should call client.call_tool and format result."""
        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": "file contents here"}],
        })

        handler = MCPToolBridge._make_handler(mock_client, "read_file", "fs")
        result = await handler(path="/tmp/test.txt")

        mock_client.call_tool.assert_awaited_once_with(
            "read_file", {"path": "/tmp/test.txt"}
        )
        assert result["result"] == "file contents here"
        assert result["server"] == "fs"
        assert result["tool"] == "read_file"

    @pytest.mark.asyncio
    async def test_make_handler_error(self) -> None:
        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(side_effect=RuntimeError("Connection lost"))

        handler = MCPToolBridge._make_handler(mock_client, "broken_tool", "srv")
        result = await handler()

        assert "error" in result
        assert "Connection lost" in result["error"]
        assert result["server"] == "srv"
        assert result["tool"] == "broken_tool"
