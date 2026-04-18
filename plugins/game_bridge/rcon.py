"""
Lexy AI - Async RCON Client.

Pure-Python implementation of the Source RCON protocol over TCP.
No external dependencies -- uses only asyncio and struct from stdlib.

Packet format (little-endian):
    4 bytes  - packet size (excludes these 4 bytes)
    4 bytes  - request id (int32)
    4 bytes  - type (int32): 3=login, 2=command, 0=response
    N bytes  - payload (null-terminated ASCII/UTF-8)
    1 byte   - padding null

Flow: connect -> login (type 3) -> auth response -> command (type 2) -> response (type 0)
"""

from __future__ import annotations

import asyncio
import struct
from typing import Any

from lexy_core.utils.logging import get_logger

log = get_logger(module="rcon")

# RCON packet types
_TYPE_LOGIN: int = 3
_TYPE_COMMAND: int = 2
_TYPE_RESPONSE: int = 0

# Login response shares type 2 with command, but the request_id distinguishes it.
_TYPE_LOGIN_RESPONSE: int = 2


class AsyncRCON:
    """Async RCON client for Minecraft / Factorio / Source-engine servers."""

    def __init__(
        self,
        host: str,
        port: int,
        password: str,
        timeout: float = 10.0,
    ) -> None:
        self._host = host
        self._port = port
        self._password = password
        self._timeout = timeout

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._request_id: int = 0
        self._connected: bool = False
        self._authenticated: bool = False

    # ── Public properties ────────────────────────────────────────

    @property
    def connected(self) -> bool:
        """True when the TCP socket is open AND authentication succeeded."""
        return self._connected and self._authenticated

    # ── Lifecycle ────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Open the TCP connection and authenticate via RCON login.

        Returns True on success, False on any failure (connection refused,
        timeout, bad password, etc.).
        """
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port),
                timeout=self._timeout,
            )
            self._connected = True
            log.debug("rcon.tcp_connected", host=self._host, port=self._port)

            # Sende Login-Paket
            response = await self._send_packet(_TYPE_LOGIN, self._password)
            if response is None:
                log.warning("rcon.login_no_response", host=self._host)
                await self.disconnect()
                return False

            req_id, _resp_type, _body = response
            if req_id == -1:
                # Server sendet request_id -1 bei fehlgeschlagener Authentifizierung
                log.warning("rcon.auth_failed", host=self._host)
                await self.disconnect()
                return False

            self._authenticated = True
            log.info("rcon.authenticated", host=self._host, port=self._port)
            return True

        except (ConnectionRefusedError, OSError, asyncio.TimeoutError) as exc:
            log.warning(
                "rcon.connect_failed",
                host=self._host,
                port=self._port,
                error=str(exc),
            )
            self._connected = False
            self._authenticated = False
            return False

    async def disconnect(self) -> None:
        """Close the TCP connection gracefully."""
        self._connected = False
        self._authenticated = False
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass
            self._writer = None
            self._reader = None
        log.debug("rcon.disconnected", host=self._host, port=self._port)

    async def command(self, cmd: str) -> str:
        """Send an RCON command and return the response body.

        Raises RuntimeError if not connected.  Returns an empty string when
        the server sends no body.
        """
        if not self.connected:
            raise RuntimeError(
                f"RCON not connected to {self._host}:{self._port}"
            )

        try:
            response = await self._send_packet(_TYPE_COMMAND, cmd)
            if response is None:
                # Verbindung verloren waehrend des Commands
                await self.disconnect()
                return ""
            _req_id, _resp_type, body = response
            return body
        except Exception as exc:
            log.error("rcon.command_failed", command=cmd, error=str(exc))
            await self.disconnect()
            return ""

    # ── Packet helpers ───────────────────────────────────────────

    @staticmethod
    def _pack_packet(request_id: int, packet_type: int, body: str) -> bytes:
        """Build a raw RCON packet ready to send over the wire.

        Layout:
            [4 bytes size][4 bytes request_id][4 bytes type][body\\x00][\\x00]
        """
        body_bytes = body.encode("utf-8") + b"\x00"  # null-terminated payload
        padding = b"\x00"  # trailing padding byte
        payload = struct.pack("<ii", request_id, packet_type) + body_bytes + padding
        return struct.pack("<i", len(payload)) + payload

    async def _read_packet(self) -> tuple[int, int, str] | None:
        """Read a single RCON response packet from the stream.

        Returns ``(request_id, response_type, body)`` or ``None`` on error.
        """
        if self._reader is None:
            return None

        try:
            # Lese Paketlaenge (4 Bytes)
            size_data = await asyncio.wait_for(
                self._reader.readexactly(4),
                timeout=self._timeout,
            )
            size = struct.unpack("<i", size_data)[0]

            if size < 10 or size > 4096:
                # Minimale Paketgroesse: 4 (id) + 4 (type) + 1 (body null) + 1 (padding) = 10
                # Maximale vernuenftige Groesse: 4096
                log.warning("rcon.invalid_packet_size", size=size)
                return None

            # Lese den Rest des Pakets
            body_data = await asyncio.wait_for(
                self._reader.readexactly(size),
                timeout=self._timeout,
            )

            resp_id = struct.unpack("<i", body_data[0:4])[0]
            resp_type = struct.unpack("<i", body_data[4:8])[0]
            # Body ist alles nach dem Header bis auf die letzten 2 Null-Bytes
            resp_body = body_data[8:-2].decode("utf-8", errors="replace")

            return (resp_id, resp_type, resp_body)

        except (
            asyncio.TimeoutError,
            asyncio.IncompleteReadError,
            ConnectionResetError,
            struct.error,
        ) as exc:
            log.warning("rcon.read_failed", error=str(exc))
            await self.disconnect()
            return None

    async def _send_packet(
        self, packet_type: int, body: str
    ) -> tuple[int, int, str] | None:
        """Send a packet and read the response.

        Returns ``(request_id, response_type, body)`` or ``None`` on failure.
        """
        if self._writer is None or self._reader is None:
            return None

        self._request_id += 1
        req_id = self._request_id

        packet = self._pack_packet(req_id, packet_type, body)

        try:
            self._writer.write(packet)
            await self._writer.drain()
        except (OSError, ConnectionResetError) as exc:
            log.warning("rcon.write_failed", error=str(exc))
            await self.disconnect()
            return None

        return await self._read_packet()
