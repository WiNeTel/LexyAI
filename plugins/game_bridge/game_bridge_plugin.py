"""
Lexy AI - Game Bridge Plugin.

Provides RCON-based communication with Minecraft and Factorio game servers.
Registers four LLM-callable tools: game_command, game_status, game_chat,
game_list_players.
"""

from __future__ import annotations

import re
from typing import Any

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger
from plugins.game_bridge.rcon import AsyncRCON

log = get_logger(module="game_bridge")

# Minecraft-Farbcodes entfernen (section-sign + hex/format char)
_MC_COLOR_RE = re.compile(r"\u00a7[0-9a-fk-or]", re.IGNORECASE)

# Valide Spiel-IDs
_VALID_GAMES = frozenset({"minecraft", "factorio"})

# ── Tool schemas ─────────────────────────────────────────────────

GAME_COMMAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "game": {
            "type": "string",
            "enum": ["minecraft", "factorio"],
            "description": "Target game server",
        },
        "command": {
            "type": "string",
            "description": "The command to execute on the server (e.g. '/time set day')",
        },
    },
    "required": ["game", "command"],
}

GAME_STATUS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "game": {
            "type": "string",
            "enum": ["minecraft", "factorio"],
            "description": "Target game server",
        },
    },
    "required": ["game"],
}

GAME_CHAT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "game": {
            "type": "string",
            "enum": ["minecraft", "factorio"],
            "description": "Target game server",
        },
        "message": {
            "type": "string",
            "description": "Chat message to send in-game",
        },
    },
    "required": ["game", "message"],
}

GAME_LIST_PLAYERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "game": {
            "type": "string",
            "enum": ["minecraft", "factorio"],
            "description": "Target game server",
        },
    },
    "required": ["game"],
}


class GameBridgePlugin(BasePlugin):
    """Minecraft and Factorio game bridge via RCON."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._clients: dict[str, AsyncRCON] = {}
        self._game_configs: dict[str, dict[str, Any]] = {}

    # ── Lifecycle ────────────────────────────────────────────────

    async def on_load(self) -> None:
        """Read game server configurations from the plugin config."""
        config = self.api.get_config()
        games_cfg: dict[str, Any] = config.get("games", {})

        for game_name in ("minecraft", "factorio"):
            game_cfg = games_cfg.get(game_name, {})
            if not isinstance(game_cfg, dict):
                continue
            self._game_configs[game_name] = game_cfg

        log.info(
            "game_bridge.loaded",
            configured_games=list(self._game_configs.keys()),
        )

    async def on_enable(self) -> None:
        """Register tools and attempt best-effort connections to enabled servers."""
        # Registriere die 4 Tools
        self.api.register_tool(
            name="game_command",
            handler=self._tool_game_command,
            description=(
                "Send an RCON command to a game server (Minecraft or Factorio). "
                "Returns the server response."
            ),
            schema=GAME_COMMAND_SCHEMA,
        )
        self.api.register_tool(
            name="game_status",
            handler=self._tool_game_status,
            description=(
                "Check game server connectivity and basic status info."
            ),
            schema=GAME_STATUS_SCHEMA,
        )
        self.api.register_tool(
            name="game_chat",
            handler=self._tool_game_chat,
            description=(
                "Send a chat message visible to all players in the game. "
                "For Minecraft uses /say, for Factorio sends a server message."
            ),
            schema=GAME_CHAT_SCHEMA,
        )
        self.api.register_tool(
            name="game_list_players",
            handler=self._tool_game_list_players,
            description=(
                "List all currently online players on the game server."
            ),
            schema=GAME_LIST_PLAYERS_SCHEMA,
        )

        # Best-effort Verbindung zu aktivierten Servern
        for game_name, game_cfg in self._game_configs.items():
            if not game_cfg.get("enabled", False):
                log.debug("game_bridge.skipped_disabled", game=game_name)
                continue

            host = str(game_cfg.get("host", "127.0.0.1"))
            port = int(game_cfg.get("port", 25575 if game_name == "minecraft" else 27015))
            password = str(game_cfg.get("password", ""))

            client = AsyncRCON(host=host, port=port, password=password)
            self._clients[game_name] = client

            success = await client.connect()
            if success:
                log.info("game_bridge.connected", game=game_name, host=host, port=port)
            else:
                log.warning(
                    "game_bridge.connect_failed_on_startup",
                    game=game_name,
                    host=host,
                    port=port,
                )

    async def on_disable(self) -> None:
        """Disconnect all RCON clients."""
        for game_name, client in self._clients.items():
            if client.connected:
                await client.disconnect()
                log.info("game_bridge.disconnected", game=game_name)
        self._clients.clear()

    # ── Helpers ──────────────────────────────────────────────────

    def _get_client(self, game: str) -> AsyncRCON | None:
        """Look up the RCON client for a game, or None if not configured."""
        return self._clients.get(game.lower())

    async def _ensure_connected(self, game: str) -> tuple[AsyncRCON | None, str | None]:
        """Return a connected client or an error string.

        Attempts reconnection once if the client exists but is disconnected.
        """
        normalized = game.lower()
        if normalized not in _VALID_GAMES:
            return None, f"Unknown game '{game}'. Valid: minecraft, factorio"

        client = self._get_client(normalized)
        if client is None:
            available = list(self._clients.keys()) or ["(none configured)"]
            return None, (
                f"Game '{game}' is not configured. "
                f"Available: {', '.join(available)}"
            )

        if not client.connected:
            # Einmaliger Reconnect-Versuch
            success = await client.connect()
            if not success:
                return None, f"Could not connect to {game} server"

        return client, None

    @staticmethod
    def _strip_minecraft_colors(text: str) -> str:
        """Remove Minecraft color/formatting codes from text."""
        return _MC_COLOR_RE.sub("", text)

    @staticmethod
    def _parse_minecraft_player_list(response: str) -> list[str]:
        """Parse the /list response into a list of player names.

        Typical format: 'There are X of a max of Y players online: player1, player2'
        """
        match = re.search(r":\s*(.+)$", response)
        if match:
            names_str = match.group(1).strip()
            if names_str:
                return [n.strip() for n in names_str.split(",") if n.strip()]
        return []

    @staticmethod
    def _parse_factorio_player_list(response: str) -> list[str]:
        """Parse the /players online response into a list of player names.

        Factorio gibt Spieler zeilenweise aus, mit optionalem '(online)' Suffix.
        """
        players: list[str] = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # Ueberspringe Header-Zeilen wie 'Online players (2):'
            if not line or line.startswith("Online players") or line.startswith("Players online"):
                continue
            # Entferne '(online)' Suffix
            name = re.sub(r"\s*\(online\)\s*", "", line).strip()
            if name:
                players.append(name)
        return players

    # ── Tool handlers ────────────────────────────────────────────

    async def _tool_game_command(self, game: str, command: str) -> dict[str, Any]:
        """Send an arbitrary RCON command to a game server."""
        client, error = await self._ensure_connected(game)
        if error is not None:
            return {"error": error}

        assert client is not None  # fuer mypy

        response = await client.command(command)

        # Minecraft-Farbcodes entfernen
        if game.lower() == "minecraft":
            response = self._strip_minecraft_colors(response)

        return {
            "game": game.lower(),
            "command": command,
            "response": response,
        }

    async def _tool_game_status(self, game: str) -> dict[str, Any]:
        """Check server connectivity and return basic info."""
        normalized = game.lower()
        if normalized not in _VALID_GAMES:
            return {"error": f"Unknown game '{game}'. Valid: minecraft, factorio"}

        client = self._get_client(normalized)
        if client is None:
            return {
                "game": normalized,
                "connected": False,
                "info": "Game not configured in plugin settings",
            }

        if not client.connected:
            # Versuche einmalig zu verbinden
            success = await client.connect()
            if not success:
                return {
                    "game": normalized,
                    "connected": False,
                    "info": "Server not reachable",
                }

        # Server ist verbunden -- hole erweiterte Infos
        info_parts: list[str] = ["Server connected"]

        if normalized == "minecraft":
            raw = await client.command("/list")
            cleaned = self._strip_minecraft_colors(raw)
            if cleaned:
                info_parts.append(cleaned)
        elif normalized == "factorio":
            raw = await client.command("/players online")
            if raw:
                info_parts.append(raw.strip())

        return {
            "game": normalized,
            "connected": True,
            "info": " | ".join(info_parts),
        }

    async def _tool_game_chat(self, game: str, message: str) -> dict[str, Any]:
        """Send a chat message visible to all players."""
        client, error = await self._ensure_connected(game)
        if error is not None:
            return {"error": error}

        assert client is not None

        normalized = game.lower()
        if normalized == "minecraft":
            # Minecraft: /say sendet als [Server]
            response = await client.command(f"/say {message}")
            response = self._strip_minecraft_colors(response)
        elif normalized == "factorio":
            # Factorio: direkt die Nachricht als Server-Broadcast senden
            # (Factorio RCON behandelt jeden Text als Server-Nachricht)
            response = await client.command(message)
        else:
            return {"error": f"Unknown game '{game}'"}

        return {
            "game": normalized,
            "message": message,
            "sent": True,
            "response": response,
        }

    async def _tool_game_list_players(self, game: str) -> dict[str, Any]:
        """List all online players on the game server."""
        client, error = await self._ensure_connected(game)
        if error is not None:
            return {"error": error}

        assert client is not None

        normalized = game.lower()
        players: list[str] = []

        if normalized == "minecraft":
            raw = await client.command("/list")
            cleaned = self._strip_minecraft_colors(raw)
            players = self._parse_minecraft_player_list(cleaned)
        elif normalized == "factorio":
            raw = await client.command("/players online")
            players = self._parse_factorio_player_list(raw)

        return {
            "game": normalized,
            "players": players,
            "count": len(players),
        }
