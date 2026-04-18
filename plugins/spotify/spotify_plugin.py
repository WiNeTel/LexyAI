"""
Lexy AI - Spotify Plugin.

Provides LLM tools for Spotify playback control, search, queue management,
and playlist operations. Authentication uses OAuth 2.0 Authorization Code flow
with tokens persisted in plugin-private SQLite.

Tools registered:
    spotify_auth, spotify_play, spotify_pause, spotify_skip, spotify_volume,
    spotify_search, spotify_queue, spotify_now_playing, spotify_devices,
    spotify_playlist_create, spotify_playlist_add
"""

from __future__ import annotations

from typing import Any

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger
from plugins.spotify.spotify_auth import SpotifyAuth
from plugins.spotify.spotify_client import SpotifyClient

log = get_logger(module="spotify_plugin")


# ====================================================================== #
#  Tool schemas
# ====================================================================== #

_SCHEMA_AUTH: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["url", "code"],
            "description": (
                "'url' = generate the Spotify login link. "
                "'code' = exchange an authorization code for tokens."
            ),
        },
        "code": {
            "type": "string",
            "description": "Authorization code returned by Spotify (required when action='code').",
        },
    },
    "required": ["action"],
}

_SCHEMA_PLAY: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Free-text search query (e.g. 'Bohemian Rhapsody', 'Beethoven Moonlight'). "
                "If given, the plugin searches Spotify and plays the top result."
            ),
        },
        "uri": {
            "type": "string",
            "description": (
                "Spotify URI (e.g. 'spotify:track:...', 'spotify:album:...'). "
                "If given, plays this URI directly."
            ),
        },
        "device_id": {
            "type": "string",
            "description": "Target device ID (optional, uses active device if omitted).",
        },
    },
}

_SCHEMA_PAUSE: dict[str, Any] = {
    "type": "object",
    "properties": {},
}

_SCHEMA_SKIP: dict[str, Any] = {
    "type": "object",
    "properties": {
        "direction": {
            "type": "string",
            "enum": ["next", "previous"],
            "description": "Skip direction (default 'next').",
        },
    },
}

_SCHEMA_VOLUME: dict[str, Any] = {
    "type": "object",
    "properties": {
        "volume": {
            "type": "integer",
            "description": "Volume level 0-100.",
            "minimum": 0,
            "maximum": 100,
        },
    },
    "required": ["volume"],
}

_SCHEMA_SEARCH: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query.",
        },
        "type": {
            "type": "string",
            "description": "Comma-separated types: track, album, artist, playlist (default 'track').",
        },
        "limit": {
            "type": "integer",
            "description": "Max results per type (default 5, max 50).",
        },
    },
    "required": ["query"],
}

_SCHEMA_QUEUE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "uri": {
            "type": "string",
            "description": "Spotify URI to add to queue (e.g. 'spotify:track:...').",
        },
    },
    "required": ["uri"],
}

_SCHEMA_NOW_PLAYING: dict[str, Any] = {
    "type": "object",
    "properties": {},
}

_SCHEMA_DEVICES: dict[str, Any] = {
    "type": "object",
    "properties": {},
}

_SCHEMA_PLAYLIST_CREATE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Playlist name.",
        },
        "description": {
            "type": "string",
            "description": "Playlist description (optional).",
        },
        "public": {
            "type": "boolean",
            "description": "Whether the playlist is public (default false).",
        },
    },
    "required": ["name"],
}

_SCHEMA_PLAYLIST_ADD: dict[str, Any] = {
    "type": "object",
    "properties": {
        "playlist_id": {
            "type": "string",
            "description": "Spotify playlist ID.",
        },
        "uris": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of Spotify URIs to add.",
        },
    },
    "required": ["playlist_id", "uris"],
}


# ====================================================================== #
#  Helper: Nicht-authentifiziert-Antwort
# ====================================================================== #

def _not_authed_response(auth: SpotifyAuth) -> dict[str, Any]:
    """Standard error dict when the user hasn't authenticated yet."""
    url = auth.get_auth_url()
    return {
        "error": "Not authenticated with Spotify.",
        "instructions": (
            "Visit the URL below to authorize Lexy, then call "
            "spotify_auth with action='code' and the returned code."
        ),
        "auth_url": url,
    }


def _ms_to_mmss(ms: int) -> str:
    """Convert milliseconds to 'mm:ss' string."""
    total_s = ms // 1000
    minutes = total_s // 60
    seconds = total_s % 60
    return f"{minutes}:{seconds:02d}"


# ====================================================================== #
#  Plugin
# ====================================================================== #

class SpotifyPlugin(BasePlugin):
    """Spotify playback control, search, and playlist management."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._auth: SpotifyAuth | None = None
        self._spotify: SpotifyClient | None = None
        self._market: str = "DE"

    # ─── Lifecycle ──────────────────────────────────────────────── #

    async def on_load(self) -> None:
        config = self.api.get_config()
        client_id = str(config.get("client_id", ""))
        client_secret = str(config.get("client_secret", ""))
        redirect_uri = str(config.get("redirect_uri", "http://localhost:8765/callback/spotify"))
        scopes = str(config.get("scopes", ""))
        self._market = str(config.get("market", "DE"))

        if not client_id or not client_secret:
            log.warning(
                "spotify.missing_credentials",
                hint="Set client_id and client_secret in plugins.yaml",
            )

        db = await self.api.get_db()
        await db.execute(
            """CREATE TABLE IF NOT EXISTS tokens (
                user_id TEXT PRIMARY KEY DEFAULT 'default',
                access_token TEXT NOT NULL,
                refresh_token TEXT NOT NULL,
                expires_at REAL NOT NULL,
                scopes TEXT
            )"""
        )
        await db.commit()

        self._auth = SpotifyAuth(client_id, client_secret, redirect_uri, scopes, db)
        self._spotify = SpotifyClient(self._auth)
        log.info("spotify.loaded")

    async def on_enable(self) -> None:
        self.api.register_tool(
            name="spotify_auth",
            handler=self._tool_auth,
            description=(
                "Authenticate with Spotify. action='url' returns the login URL. "
                "action='code' exchanges an authorization code for tokens."
            ),
            schema=_SCHEMA_AUTH,
        )
        self.api.register_tool(
            name="spotify_play",
            handler=self._tool_play,
            description=(
                "Play a song, album, or playlist on Spotify. Provide 'query' to "
                "search and auto-play the top result, or 'uri' to play a specific "
                "Spotify URI. Omit both to resume paused playback."
            ),
            schema=_SCHEMA_PLAY,
        )
        self.api.register_tool(
            name="spotify_pause",
            handler=self._tool_pause,
            description="Pause Spotify playback.",
            schema=_SCHEMA_PAUSE,
        )
        self.api.register_tool(
            name="spotify_skip",
            handler=self._tool_skip,
            description="Skip to the next or previous track on Spotify.",
            schema=_SCHEMA_SKIP,
        )
        self.api.register_tool(
            name="spotify_volume",
            handler=self._tool_volume,
            description="Set Spotify playback volume (0-100).",
            schema=_SCHEMA_VOLUME,
        )
        self.api.register_tool(
            name="spotify_search",
            handler=self._tool_search,
            description=(
                "Search the Spotify catalog for tracks, albums, artists, or playlists."
            ),
            schema=_SCHEMA_SEARCH,
        )
        self.api.register_tool(
            name="spotify_queue",
            handler=self._tool_queue,
            description="Add a track to the Spotify playback queue.",
            schema=_SCHEMA_QUEUE,
        )
        self.api.register_tool(
            name="spotify_now_playing",
            handler=self._tool_now_playing,
            description="Show the currently playing track on Spotify.",
            schema=_SCHEMA_NOW_PLAYING,
        )
        self.api.register_tool(
            name="spotify_devices",
            handler=self._tool_devices,
            description="List available Spotify playback devices.",
            schema=_SCHEMA_DEVICES,
        )
        self.api.register_tool(
            name="spotify_playlist_create",
            handler=self._tool_playlist_create,
            description="Create a new Spotify playlist.",
            schema=_SCHEMA_PLAYLIST_CREATE,
        )
        self.api.register_tool(
            name="spotify_playlist_add",
            handler=self._tool_playlist_add,
            description="Add tracks to a Spotify playlist.",
            schema=_SCHEMA_PLAYLIST_ADD,
        )

        # WS-Handler: Frontend kann Spotify-Status abfragen
        self.api.register_ws_handler("spotify_status", self._ws_status)
        self.api.register_ws_handler("spotify_command", self._ws_command)

        log.info("spotify.enabled", tools=11, ws_handlers=2)

    async def on_disable(self) -> None:
        if self._spotify is not None:
            await self._spotify.close()
            self._spotify = None
        if self._auth is not None:
            await self._auth.close()
            self._auth = None
        log.info("spotify.disabled")

    # ─── Auth check helper ──────────────────────────────────────── #

    async def _require_auth(self) -> dict[str, Any] | None:
        """Return error dict if not authenticated, else None."""
        assert self._auth is not None
        if not await self._auth.is_authenticated():
            return _not_authed_response(self._auth)
        return None

    # ─── Tool: Auth ─────────────────────────────────────────────── #

    async def _tool_auth(
        self,
        action: str,
        code: str | None = None,
    ) -> dict[str, Any]:
        assert self._auth is not None

        if action == "url":
            url = self._auth.get_auth_url()
            return {
                "auth_url": url,
                "instructions": (
                    "Open this URL in a browser, authorize Lexy, then call "
                    "spotify_auth with action='code' and the code from the redirect URL."
                ),
            }

        if action == "code":
            if not code:
                return {"error": "Parameter 'code' is required when action='code'."}
            result = await self._auth.exchange_code(code)
            if "error" in result:
                return result
            return {"status": "authenticated", "message": "Spotify connected successfully!"}

        return {"error": f"Unknown action '{action}'. Use 'url' or 'code'."}

    # ─── Tool: Play ─────────────────────────────────────────────── #

    async def _tool_play(
        self,
        query: str | None = None,
        uri: str | None = None,
        device_id: str | None = None,
    ) -> dict[str, Any]:
        assert self._spotify is not None
        auth_err = await self._require_auth()
        if auth_err is not None:
            return auth_err

        # Wenn eine Suchanfrage da ist, erst suchen
        if query and not uri:
            search_result = await self._spotify.search(
                query=query, types="track", limit=1, market=self._market,
            )
            if "error" in search_result:
                return search_result

            tracks = search_result.get("tracks", {})
            items: list[dict[str, Any]] = tracks.get("items", []) if isinstance(tracks, dict) else []
            if not items:
                return {"error": f"No tracks found for query: {query}"}

            track = items[0]
            uri = str(track.get("uri", ""))
            track_name = str(track.get("name", "Unknown"))
            artists = ", ".join(
                str(a.get("name", "")) for a in track.get("artists", [])
            )
            album_name = str(track.get("album", {}).get("name", ""))
            duration_ms = int(track.get("duration_ms", 0))

            play_result = await self._spotify.play(uri=uri, device_id=device_id)
            if "error" in play_result:
                return play_result

            return {
                "status": "playing",
                "track": track_name,
                "artist": artists,
                "album": album_name,
                "duration": _ms_to_mmss(duration_ms),
                "uri": uri,
            }

        # Direkt URI abspielen oder einfach Resume
        play_result = await self._spotify.play(uri=uri, device_id=device_id)
        if "error" in play_result:
            return play_result

        if uri:
            return {"status": "playing", "uri": uri}
        return {"status": "resumed"}

    # ─── Tool: Pause ────────────────────────────────────────────── #

    async def _tool_pause(self) -> dict[str, Any]:
        assert self._spotify is not None
        auth_err = await self._require_auth()
        if auth_err is not None:
            return auth_err
        result = await self._spotify.pause()
        if "error" in result:
            return result
        return {"status": "paused"}

    # ─── Tool: Skip ─────────────────────────────────────────────── #

    async def _tool_skip(
        self,
        direction: str | None = None,
    ) -> dict[str, Any]:
        assert self._spotify is not None
        auth_err = await self._require_auth()
        if auth_err is not None:
            return auth_err

        direction = direction or "next"
        if direction == "previous":
            result = await self._spotify.previous_track()
        else:
            result = await self._spotify.next_track()

        if "error" in result:
            return result
        return {"status": "skipped", "direction": direction}

    # ─── Tool: Volume ───────────────────────────────────────────── #

    async def _tool_volume(self, volume: int) -> dict[str, Any]:
        assert self._spotify is not None
        auth_err = await self._require_auth()
        if auth_err is not None:
            return auth_err

        result = await self._spotify.set_volume(volume)
        if "error" in result:
            return result
        return {"status": "volume_set", "volume": max(0, min(100, volume))}

    # ─── Tool: Search ───────────────────────────────────────────── #

    async def _tool_search(
        self,
        query: str,
        type: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        assert self._spotify is not None
        auth_err = await self._require_auth()
        if auth_err is not None:
            return auth_err

        search_type = type or "track"
        search_limit = max(1, min(50, limit or 5))

        raw = await self._spotify.search(
            query=query,
            types=search_type,
            limit=search_limit,
            market=self._market,
        )
        if "error" in raw:
            return raw

        # Ergebnisse fuer den LLM aufbereiten
        formatted: dict[str, Any] = {"query": query}

        if "tracks" in raw:
            tracks_data = raw["tracks"]
            items = tracks_data.get("items", []) if isinstance(tracks_data, dict) else []
            formatted["tracks"] = [
                {
                    "name": str(t.get("name", "")),
                    "artist": ", ".join(str(a.get("name", "")) for a in t.get("artists", [])),
                    "album": str(t.get("album", {}).get("name", "")),
                    "uri": str(t.get("uri", "")),
                    "duration": _ms_to_mmss(int(t.get("duration_ms", 0))),
                }
                for t in items
            ]

        if "albums" in raw:
            albums_data = raw["albums"]
            items = albums_data.get("items", []) if isinstance(albums_data, dict) else []
            formatted["albums"] = [
                {
                    "name": str(a.get("name", "")),
                    "artist": ", ".join(str(ar.get("name", "")) for ar in a.get("artists", [])),
                    "uri": str(a.get("uri", "")),
                    "total_tracks": int(a.get("total_tracks", 0)),
                }
                for a in items
            ]

        if "artists" in raw:
            artists_data = raw["artists"]
            items = artists_data.get("items", []) if isinstance(artists_data, dict) else []
            formatted["artists"] = [
                {
                    "name": str(a.get("name", "")),
                    "uri": str(a.get("uri", "")),
                    "genres": list(a.get("genres", [])),
                }
                for a in items
            ]

        if "playlists" in raw:
            playlists_data = raw["playlists"]
            items = playlists_data.get("items", []) if isinstance(playlists_data, dict) else []
            formatted["playlists"] = [
                {
                    "name": str(p.get("name", "")),
                    "uri": str(p.get("uri", "")),
                    "owner": str(p.get("owner", {}).get("display_name", "")),
                    "total_tracks": int(p.get("tracks", {}).get("total", 0)),
                }
                for p in items
            ]

        return formatted

    # ─── Tool: Queue ────────────────────────────────────────────── #

    async def _tool_queue(self, uri: str) -> dict[str, Any]:
        assert self._spotify is not None
        auth_err = await self._require_auth()
        if auth_err is not None:
            return auth_err

        result = await self._spotify.add_to_queue(uri)
        if "error" in result:
            return result
        return {"status": "queued", "uri": uri}

    # ─── Tool: Now Playing ──────────────────────────────────────── #

    async def _tool_now_playing(self) -> dict[str, Any]:
        assert self._spotify is not None
        auth_err = await self._require_auth()
        if auth_err is not None:
            return auth_err

        playback = await self._spotify.get_playback()
        if "error" in playback:
            return playback

        if not playback:
            return {"status": "nothing_playing", "message": "No active playback found."}

        is_playing = bool(playback.get("is_playing", False))
        item = playback.get("item") or {}
        device = playback.get("device") or {}

        track_name = str(item.get("name", "Unknown"))
        artists = ", ".join(
            str(a.get("name", "")) for a in item.get("artists", [])
        )
        album_name = str(item.get("album", {}).get("name", ""))
        progress_ms = int(playback.get("progress_ms", 0))
        duration_ms = int(item.get("duration_ms", 0))
        device_name = str(device.get("name", "Unknown"))
        volume = device.get("volume_percent")

        return {
            "status": "playing" if is_playing else "paused",
            "track": track_name,
            "artist": artists,
            "album": album_name,
            "progress": _ms_to_mmss(progress_ms),
            "duration": _ms_to_mmss(duration_ms),
            "device": device_name,
            "volume": volume,
            "uri": str(item.get("uri", "")),
        }

    # ─── Tool: Devices ──────────────────────────────────────────── #

    async def _tool_devices(self) -> dict[str, Any]:
        assert self._spotify is not None
        auth_err = await self._require_auth()
        if auth_err is not None:
            return auth_err

        raw = await self._spotify.get_devices()
        if "error" in raw:
            return raw

        devices_list: list[dict[str, Any]] = raw.get("devices", [])
        if not devices_list:
            return {"devices": [], "message": "No active Spotify devices found."}

        formatted = [
            {
                "id": str(d.get("id", "")),
                "name": str(d.get("name", "")),
                "type": str(d.get("type", "")),
                "is_active": bool(d.get("is_active", False)),
                "volume": d.get("volume_percent"),
            }
            for d in devices_list
        ]
        return {"devices": formatted}

    # ─── Tool: Playlist Create ──────────────────────────────────── #

    async def _tool_playlist_create(
        self,
        name: str,
        description: str | None = None,
        public: bool | None = None,
    ) -> dict[str, Any]:
        assert self._spotify is not None
        auth_err = await self._require_auth()
        if auth_err is not None:
            return auth_err

        # User-ID vom aktuellen Account holen
        user = await self._spotify.get_current_user()
        if "error" in user:
            return user

        user_id = str(user.get("id", ""))
        if not user_id:
            return {"error": "Could not determine Spotify user ID."}

        result = await self._spotify.create_playlist(
            user_id=user_id,
            name=name,
            description=description or "",
            public=public if public is not None else False,
        )
        if "error" in result:
            return result

        return {
            "status": "created",
            "playlist_id": str(result.get("id", "")),
            "name": str(result.get("name", "")),
            "uri": str(result.get("uri", "")),
            "url": str(result.get("external_urls", {}).get("spotify", "")),
        }

    # ─── Tool: Playlist Add ─────────────────────────────────────── #

    async def _tool_playlist_add(
        self,
        playlist_id: str,
        uris: list[str],
    ) -> dict[str, Any]:
        assert self._spotify is not None
        auth_err = await self._require_auth()
        if auth_err is not None:
            return auth_err

        result = await self._spotify.add_to_playlist(playlist_id=playlist_id, uris=uris)
        if "error" in result:
            return result

        return {
            "status": "added",
            "playlist_id": playlist_id,
            "tracks_added": len(uris),
        }

    # ─── WebSocket handlers ─────────────────────────────────────── #

    async def _ws_status(self, data: dict[str, Any]) -> dict[str, Any]:
        """Handle 'spotify_status' WS messages from the frontend."""
        assert self._spotify is not None
        assert self._auth is not None

        if not await self._auth.is_authenticated():
            return {"type": "spotify_status", "authenticated": False}

        playback = await self._spotify.get_playback()
        if "error" in playback or not playback:
            return {
                "type": "spotify_status",
                "authenticated": True,
                "playing": False,
            }

        item = playback.get("item") or {}
        return {
            "type": "spotify_status",
            "authenticated": True,
            "playing": bool(playback.get("is_playing", False)),
            "track": str(item.get("name", "")),
            "artist": ", ".join(
                str(a.get("name", "")) for a in item.get("artists", [])
            ),
            "album": str(item.get("album", {}).get("name", "")),
            "progress_ms": int(playback.get("progress_ms", 0)),
            "duration_ms": int(item.get("duration_ms", 0)),
        }

    async def _ws_command(self, data: dict[str, Any]) -> dict[str, Any]:
        """Handle 'spotify_command' WS messages for frontend controls."""
        command = str(data.get("command", ""))
        handler_map: dict[str, Any] = {
            "play": self._tool_play,
            "pause": self._tool_pause,
            "skip": self._tool_skip,
            "volume": self._tool_volume,
            "now_playing": self._tool_now_playing,
        }
        handler = handler_map.get(command)
        if handler is None:
            return {"type": "spotify_command", "error": f"Unknown command: {command}"}

        # Weiterleitung der Parameter an den Tool-Handler
        params = dict(data.get("params", {}))
        result = await handler(**params)
        return {"type": "spotify_command", "command": command, **result}
