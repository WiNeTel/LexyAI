"""
Lexy AI - Spotify Web API Client.

Async httpx wrapper for the Spotify REST API. All methods auto-attach
a valid Bearer token and transparently retry once on 401 (expired token).
"""

from __future__ import annotations

from typing import Any

import httpx

from lexy_core.utils.logging import get_logger
from plugins.spotify.spotify_auth import SpotifyAuth

log = get_logger(module="spotify_client")


class SpotifyClient:
    """Async Spotify Web API client."""

    BASE_URL = "https://api.spotify.com/v1"

    def __init__(self, auth: SpotifyAuth) -> None:
        self._auth = auth
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=10.0,
        )

    # ------------------------------------------------------------------ #
    #  Internal request helper
    # ------------------------------------------------------------------ #

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make authenticated request. Auto-refresh token on 401."""
        token = await self._auth.get_valid_token()
        if not token:
            return {"error": "Not authenticated. Use spotify_auth to connect."}

        headers = {"Authorization": f"Bearer {token}"}

        try:
            resp = await self._client.request(
                method, path, headers=headers, **kwargs,
            )
        except httpx.HTTPError as exc:
            log.error("spotify_client.request_failed", method=method, path=path, error=str(exc))
            return {"error": f"Request failed: {exc}"}

        # 204 No Content ist normal fuer Playback-Befehle
        if resp.status_code == 204:
            return {}

        # 401 Unauthorized -> einmal Token refresh und Retry
        if resp.status_code == 401:
            log.debug("spotify_client.token_expired_retrying", path=path)
            new_token = await self._auth.refresh_access_token()
            if not new_token:
                return {"error": "Token refresh failed. Re-authenticate with spotify_auth."}
            headers["Authorization"] = f"Bearer {new_token}"
            try:
                resp = await self._client.request(
                    method, path, headers=headers, **kwargs,
                )
            except httpx.HTTPError as exc:
                log.error("spotify_client.retry_failed", method=method, path=path, error=str(exc))
                return {"error": f"Retry failed: {exc}"}
            if resp.status_code == 204:
                return {}

        if resp.status_code >= 400:
            detail = resp.text[:200] if resp.text else ""
            log.warning(
                "spotify_client.api_error",
                status=resp.status_code,
                path=path,
                detail=detail,
            )
            return {"error": f"Spotify API error {resp.status_code}", "detail": detail}

        if not resp.content:
            return {}

        return dict(resp.json())

    # ------------------------------------------------------------------ #
    #  Playback
    # ------------------------------------------------------------------ #

    async def get_playback(self) -> dict[str, Any]:
        """Get current playback state."""
        return await self._request("GET", "/me/player")

    async def play(
        self,
        uri: str | None = None,
        device_id: str | None = None,
    ) -> dict[str, Any]:
        """Start/resume playback. Optionally play a specific URI."""
        params: dict[str, str] = {}
        if device_id:
            params["device_id"] = device_id

        body: dict[str, Any] = {}
        if uri:
            if uri.startswith("spotify:track:"):
                body["uris"] = [uri]
            else:
                # Album, Playlist, Artist URI -> context_uri
                body["context_uri"] = uri

        return await self._request(
            "PUT", "/me/player/play",
            params=params,
            json=body if body else None,
        )

    async def pause(self) -> dict[str, Any]:
        """Pause playback."""
        return await self._request("PUT", "/me/player/pause")

    async def next_track(self) -> dict[str, Any]:
        """Skip to next track."""
        return await self._request("POST", "/me/player/next")

    async def previous_track(self) -> dict[str, Any]:
        """Skip to previous track."""
        return await self._request("POST", "/me/player/previous")

    async def set_volume(self, percent: int) -> dict[str, Any]:
        """Set volume (0-100)."""
        clamped = max(0, min(100, percent))
        return await self._request(
            "PUT", "/me/player/volume",
            params={"volume_percent": clamped},
        )

    # ------------------------------------------------------------------ #
    #  Search
    # ------------------------------------------------------------------ #

    async def search(
        self,
        query: str,
        types: str = "track",
        limit: int = 5,
        market: str = "DE",
    ) -> dict[str, Any]:
        """Search Spotify catalog."""
        return await self._request(
            "GET", "/search",
            params={"q": query, "type": types, "limit": limit, "market": market},
        )

    # ------------------------------------------------------------------ #
    #  Queue
    # ------------------------------------------------------------------ #

    async def add_to_queue(self, uri: str) -> dict[str, Any]:
        """Add a track to the playback queue."""
        return await self._request(
            "POST", "/me/player/queue",
            params={"uri": uri},
        )

    # ------------------------------------------------------------------ #
    #  Devices
    # ------------------------------------------------------------------ #

    async def get_devices(self) -> dict[str, Any]:
        """List available playback devices."""
        return await self._request("GET", "/me/player/devices")

    # ------------------------------------------------------------------ #
    #  User
    # ------------------------------------------------------------------ #

    async def get_current_user(self) -> dict[str, Any]:
        """Get current user profile."""
        return await self._request("GET", "/me")

    # ------------------------------------------------------------------ #
    #  Playlists
    # ------------------------------------------------------------------ #

    async def create_playlist(
        self,
        user_id: str,
        name: str,
        description: str = "",
        public: bool = False,
    ) -> dict[str, Any]:
        """Create a new playlist for the given user."""
        return await self._request(
            "POST", f"/users/{user_id}/playlists",
            json={"name": name, "description": description, "public": public},
        )

    async def add_to_playlist(
        self,
        playlist_id: str,
        uris: list[str],
    ) -> dict[str, Any]:
        """Add tracks to a playlist."""
        return await self._request(
            "POST", f"/playlists/{playlist_id}/tracks",
            json={"uris": uris},
        )

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #

    async def close(self) -> None:
        """Close the internal HTTP client."""
        await self._client.aclose()
