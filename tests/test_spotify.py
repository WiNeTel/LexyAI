"""Tests for the Spotify plugin -- auth URL generation, client methods with mocked httpx.

Covers:
* SpotifyAuth.get_auth_url() generates valid URL with all params
* SpotifyAuth.is_authenticated with mocked tokens
* SpotifyClient methods with mocked httpx responses
* Search result formatting
* Now-playing (get_playback) response format
* Error handling (not authenticated, HTTP errors)
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse, parse_qs

import httpx
import pytest

from plugins.spotify.spotify_auth import SpotifyAuth
from plugins.spotify.spotify_client import SpotifyClient


# ─── Helper: create SpotifyAuth without __init__ side effects ─────────────


def _make_auth(
    client_id: str = "test_id",
    client_secret: str = "test_secret",
    redirect_uri: str = "http://localhost:8765/callback/spotify",
    scopes: str = "user-read-playback-state user-modify-playback-state",
) -> SpotifyAuth:
    """Create a SpotifyAuth bypassing __init__ (no db/httpx needed)."""
    auth = SpotifyAuth.__new__(SpotifyAuth)
    auth._client_id = client_id
    auth._client_secret = client_secret
    auth._redirect_uri = redirect_uri
    auth._scopes = scopes
    auth._db = MagicMock()
    auth._access_token = None
    auth._refresh_token = None
    auth._expires_at = 0.0
    auth._http = MagicMock()
    return auth


# ─── SpotifyAuth ──────────────────────────────────────────────────────────


class TestSpotifyAuthUrl:
    def test_url_contains_authorize_endpoint(self) -> None:
        auth = _make_auth()
        url = auth.get_auth_url()
        assert "https://accounts.spotify.com/authorize" in url

    def test_url_contains_client_id(self) -> None:
        auth = _make_auth(client_id="my_client_123")
        url = auth.get_auth_url()
        assert "client_id=my_client_123" in url

    def test_url_contains_response_type_code(self) -> None:
        auth = _make_auth()
        url = auth.get_auth_url()
        assert "response_type=code" in url

    def test_url_contains_redirect_uri(self) -> None:
        auth = _make_auth(redirect_uri="http://example.com/cb")
        url = auth.get_auth_url()
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        assert "redirect_uri" in params
        assert "http://example.com/cb" in params["redirect_uri"]

    def test_url_contains_scopes(self) -> None:
        auth = _make_auth(scopes="user-read-playback-state")
        url = auth.get_auth_url()
        assert "scope=" in url

    def test_url_contains_state_param(self) -> None:
        auth = _make_auth()
        url = auth.get_auth_url()
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        assert "state" in params
        assert len(params["state"][0]) > 10  # random token


class TestSpotifyAuthTokens:
    @pytest.mark.asyncio
    async def test_is_authenticated_no_tokens(self) -> None:
        auth = _make_auth()
        auth._access_token = None
        auth._expires_at = 0.0
        # Mock load_tokens to return None (no stored tokens)
        auth.load_tokens = AsyncMock(return_value=None)
        result = await auth.is_authenticated()
        assert result is False

    @pytest.mark.asyncio
    async def test_is_authenticated_with_valid_token(self) -> None:
        auth = _make_auth()
        auth._access_token = "valid_token"
        auth._expires_at = time.time() + 3600  # Expires in an hour
        result = await auth.is_authenticated()
        assert result is True

    @pytest.mark.asyncio
    async def test_get_valid_token_from_cache(self) -> None:
        auth = _make_auth()
        auth._access_token = "cached_token"
        auth._expires_at = time.time() + 3600
        token = await auth.get_valid_token()
        assert token == "cached_token"

    @pytest.mark.asyncio
    async def test_get_valid_token_expired_triggers_refresh(self) -> None:
        auth = _make_auth()
        auth._access_token = "expired"
        auth._expires_at = time.time() - 100  # Already expired
        auth.load_tokens = AsyncMock(return_value={
            "access_token": "expired",
            "refresh_token": "refresh_tok",
            "expires_at": time.time() - 100,
            "scopes": "user-read-playback-state",
        })
        auth.refresh_access_token = AsyncMock(return_value="new_token")
        token = await auth.get_valid_token()
        assert token == "new_token"


# ─── SpotifyClient ────────────────────────────────────────────────────────


class TestSpotifyClient:
    def _make_client(self) -> tuple[SpotifyClient, SpotifyAuth]:
        auth = _make_auth()
        auth.get_valid_token = AsyncMock(return_value="test_token")
        auth.refresh_access_token = AsyncMock(return_value="refreshed_token")
        client = SpotifyClient(auth)
        return client, auth

    @pytest.mark.asyncio
    async def test_search_success(self) -> None:
        client, auth = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"tracks": {"items": []}}'
        mock_response.json.return_value = {"tracks": {"items": []}}

        client._client = MagicMock()
        client._client.request = AsyncMock(return_value=mock_response)

        result = await client.search("test query")
        assert "tracks" in result

    @pytest.mark.asyncio
    async def test_search_not_authenticated(self) -> None:
        client, auth = self._make_client()
        auth.get_valid_token = AsyncMock(return_value=None)

        result = await client.search("test")
        assert "error" in result
        assert "Not authenticated" in result["error"]

    @pytest.mark.asyncio
    async def test_get_playback_success(self) -> None:
        client, auth = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"is_playing": true}'
        mock_response.json.return_value = {"is_playing": True, "item": {"name": "Song"}}

        client._client = MagicMock()
        client._client.request = AsyncMock(return_value=mock_response)

        result = await client.get_playback()
        assert "is_playing" in result

    @pytest.mark.asyncio
    async def test_pause_returns_empty_on_204(self) -> None:
        client, auth = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 204

        client._client = MagicMock()
        client._client.request = AsyncMock(return_value=mock_response)

        result = await client.pause()
        assert result == {}

    @pytest.mark.asyncio
    async def test_http_error_handling(self) -> None:
        client, auth = self._make_client()

        client._client = MagicMock()
        client._client.request = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        result = await client.search("test")
        assert "error" in result
        assert "Request failed" in result["error"]

    @pytest.mark.asyncio
    async def test_set_volume_clamps_range(self) -> None:
        client, auth = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 204

        client._client = MagicMock()
        client._client.request = AsyncMock(return_value=mock_response)

        # Volume above 100 should be clamped
        result = await client.set_volume(150)
        assert result == {}
        call_args = client._client.request.call_args
        assert call_args[1]["params"]["volume_percent"] == 100

    @pytest.mark.asyncio
    async def test_api_error_response(self) -> None:
        client, auth = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"

        client._client = MagicMock()
        client._client.request = AsyncMock(return_value=mock_response)

        result = await client.get_devices()
        assert "error" in result
        assert "403" in result["error"]

    def test_base_url(self) -> None:
        assert SpotifyClient.BASE_URL == "https://api.spotify.com/v1"

    def test_client_has_all_methods(self) -> None:
        methods = [
            "get_playback", "play", "pause", "next_track", "previous_track",
            "set_volume", "search", "add_to_queue", "get_devices",
            "get_current_user", "create_playlist", "add_to_playlist", "close",
        ]
        for m in methods:
            assert hasattr(SpotifyClient, m), f"Missing method: {m}"
