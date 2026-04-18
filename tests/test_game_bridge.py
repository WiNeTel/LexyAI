"""Tests for the game_bridge plugin -- RCON client and plugin helpers."""

from __future__ import annotations

import asyncio
import struct
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.game_bridge.rcon import AsyncRCON
from plugins.game_bridge.game_bridge_plugin import GameBridgePlugin


# ── RCON packet helpers ─────────────────────────────────────────


class TestRCONPackPacker:
    """Verify the static _pack_packet helper produces correct wire format."""

    def test_login_packet(self) -> None:
        raw = AsyncRCON._pack_packet(1, 3, "mypassword")
        # Parse it back
        size = struct.unpack("<i", raw[0:4])[0]
        assert size == len(raw) - 4
        req_id = struct.unpack("<i", raw[4:8])[0]
        ptype = struct.unpack("<i", raw[8:12])[0]
        assert req_id == 1
        assert ptype == 3
        # Body ist null-terminiert + padding null
        body_and_pad = raw[12:]
        assert body_and_pad == b"mypassword\x00\x00"

    def test_command_packet(self) -> None:
        raw = AsyncRCON._pack_packet(42, 2, "/list")
        size = struct.unpack("<i", raw[0:4])[0]
        req_id = struct.unpack("<i", raw[4:8])[0]
        ptype = struct.unpack("<i", raw[8:12])[0]
        assert req_id == 42
        assert ptype == 2
        body_and_pad = raw[12:]
        assert body_and_pad == b"/list\x00\x00"

    def test_empty_body(self) -> None:
        raw = AsyncRCON._pack_packet(1, 2, "")
        body_and_pad = raw[12:]
        assert body_and_pad == b"\x00\x00"

    def test_size_field_excludes_itself(self) -> None:
        raw = AsyncRCON._pack_packet(1, 2, "x")
        declared_size = struct.unpack("<i", raw[0:4])[0]
        actual_payload = raw[4:]
        assert declared_size == len(actual_payload)


# ── RCON client integration (mocked TCP) ─────────────────────────


def _build_response_bytes(req_id: int, resp_type: int, body: str) -> bytes:
    """Build a fake RCON response as raw bytes (size-prefixed)."""
    body_bytes = body.encode("utf-8") + b"\x00\x00"
    payload = struct.pack("<ii", req_id, resp_type) + body_bytes
    return struct.pack("<i", len(payload)) + payload


class TestAsyncRCON:
    """Test the AsyncRCON client with mocked TCP streams."""

    @pytest.mark.asyncio
    async def test_connect_success(self) -> None:
        rcon = AsyncRCON("127.0.0.1", 25575, "secret")
        resp_bytes = _build_response_bytes(1, 2, "")

        mock_reader = AsyncMock(spec=asyncio.StreamReader)
        mock_reader.readexactly = AsyncMock(side_effect=[
            resp_bytes[0:4],   # size
            resp_bytes[4:],    # payload
        ])
        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch("plugins.game_bridge.rcon.asyncio.open_connection",
                    return_value=(mock_reader, mock_writer)):
            result = await rcon.connect()

        assert result is True
        assert rcon.connected is True
        await rcon.disconnect()
        assert rcon.connected is False

    @pytest.mark.asyncio
    async def test_connect_auth_failed(self) -> None:
        """Server returns request_id -1 on bad password."""
        rcon = AsyncRCON("127.0.0.1", 25575, "wrong")
        resp_bytes = _build_response_bytes(-1, 2, "")

        mock_reader = AsyncMock(spec=asyncio.StreamReader)
        mock_reader.readexactly = AsyncMock(side_effect=[
            resp_bytes[0:4],
            resp_bytes[4:],
        ])
        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch("plugins.game_bridge.rcon.asyncio.open_connection",
                    return_value=(mock_reader, mock_writer)):
            result = await rcon.connect()

        assert result is False
        assert rcon.connected is False

    @pytest.mark.asyncio
    async def test_connect_refused(self) -> None:
        rcon = AsyncRCON("127.0.0.1", 25575, "secret")

        with patch("plugins.game_bridge.rcon.asyncio.open_connection",
                    side_effect=ConnectionRefusedError("refused")):
            result = await rcon.connect()

        assert result is False
        assert rcon.connected is False

    @pytest.mark.asyncio
    async def test_command_returns_body(self) -> None:
        rcon = AsyncRCON("127.0.0.1", 25575, "secret")

        # Simuliere bereits verbundenen Client
        login_resp = _build_response_bytes(1, 2, "")
        cmd_resp = _build_response_bytes(2, 0, "There are 3 players online: Alice, Bob, Charlie")

        all_reads = [
            login_resp[0:4], login_resp[4:],
            cmd_resp[0:4], cmd_resp[4:],
        ]

        mock_reader = AsyncMock(spec=asyncio.StreamReader)
        mock_reader.readexactly = AsyncMock(side_effect=all_reads)
        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch("plugins.game_bridge.rcon.asyncio.open_connection",
                    return_value=(mock_reader, mock_writer)):
            await rcon.connect()
            result = await rcon.command("/list")

        assert "Alice" in result
        assert "Bob" in result
        await rcon.disconnect()

    @pytest.mark.asyncio
    async def test_command_when_not_connected(self) -> None:
        rcon = AsyncRCON("127.0.0.1", 25575, "secret")
        with pytest.raises(RuntimeError, match="not connected"):
            await rcon.command("/list")


# ── Plugin helper tests ──────────────────────────────────────────


class TestGameBridgeHelpers:
    """Test static/class methods of the plugin without full lifecycle."""

    def test_strip_minecraft_colors(self) -> None:
        colored = "\u00a7aThere are \u00a7e3\u00a7a of max \u00a7e20\u00a7a players"
        result = GameBridgePlugin._strip_minecraft_colors(colored)
        assert "\u00a7" not in result
        assert "There are 3 of max 20 players" == result

    def test_parse_minecraft_player_list_with_players(self) -> None:
        response = "There are 3 of a max of 20 players online: Alice, Bob, Charlie"
        players = GameBridgePlugin._parse_minecraft_player_list(response)
        assert players == ["Alice", "Bob", "Charlie"]

    def test_parse_minecraft_player_list_empty(self) -> None:
        response = "There are 0 of a max of 20 players online:"
        players = GameBridgePlugin._parse_minecraft_player_list(response)
        assert players == []

    def test_parse_minecraft_player_list_no_colon(self) -> None:
        response = "Some unexpected format"
        players = GameBridgePlugin._parse_minecraft_player_list(response)
        assert players == []

    def test_parse_factorio_player_list(self) -> None:
        response = "Online players (2):\n  Alice (online)\n  Bob (online)\n"
        players = GameBridgePlugin._parse_factorio_player_list(response)
        assert players == ["Alice", "Bob"]

    def test_parse_factorio_player_list_empty(self) -> None:
        response = "Online players (0):\n"
        players = GameBridgePlugin._parse_factorio_player_list(response)
        assert players == []

    def test_parse_factorio_player_list_no_online_suffix(self) -> None:
        response = "Players online:\n  Alice\n  Bob\n"
        players = GameBridgePlugin._parse_factorio_player_list(response)
        assert players == ["Alice", "Bob"]
