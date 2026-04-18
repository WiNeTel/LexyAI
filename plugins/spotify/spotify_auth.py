"""
Lexy AI - Spotify OAuth 2.0 Authentication.

Handles the Authorization Code flow:
1. Generate an auth URL for the user to visit.
2. Exchange the returned code for access + refresh tokens.
3. Persist tokens in SQLite for cross-session reuse.
4. Auto-refresh expired access tokens.
"""

from __future__ import annotations

import secrets
import time
from typing import Any
from urllib.parse import urlencode

import aiosqlite
import httpx

from lexy_core.utils.logging import get_logger

log = get_logger(module="spotify_auth")


class SpotifyAuth:
    """Handles Spotify OAuth 2.0 authentication."""

    TOKEN_URL = "https://accounts.spotify.com/api/token"
    AUTH_URL = "https://accounts.spotify.com/authorize"

    # Refresh 60 s before actual expiry to avoid races
    _EXPIRY_BUFFER_SECONDS: int = 60

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scopes: str,
        db: aiosqlite.Connection,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri
        self._scopes = scopes
        self._db = db

        # In-memory cache to avoid DB reads on every request
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._expires_at: float = 0.0

        self._http = httpx.AsyncClient(timeout=10.0)

    # ------------------------------------------------------------------ #
    #  Auth URL
    # ------------------------------------------------------------------ #

    def get_auth_url(self) -> str:
        """Generate Spotify authorization URL for the user to visit."""
        state = secrets.token_urlsafe(16)
        params = {
            "client_id": self._client_id,
            "response_type": "code",
            "redirect_uri": self._redirect_uri,
            "scope": self._scopes,
            "state": state,
        }
        url = f"{self.AUTH_URL}?{urlencode(params)}"
        log.info("spotify_auth.url_generated", redirect_uri=self._redirect_uri)
        return url

    # ------------------------------------------------------------------ #
    #  Code exchange
    # ------------------------------------------------------------------ #

    async def exchange_code(self, code: str) -> dict[str, Any]:
        """Exchange authorization code for access + refresh tokens."""
        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self._redirect_uri,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        try:
            resp = await self._http.post(self.TOKEN_URL, data=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.error("spotify_auth.exchange_failed", error=str(exc))
            return {"error": f"Token exchange failed: {exc}"}

        data = resp.json()
        access_token = str(data.get("access_token", ""))
        refresh_token = str(data.get("refresh_token", ""))
        expires_in = int(data.get("expires_in", 3600))

        if not access_token:
            log.error("spotify_auth.exchange_no_token", response=data)
            return {"error": "No access_token in response"}

        await self.save_tokens(access_token, refresh_token, expires_in)
        log.info("spotify_auth.tokens_saved", expires_in=expires_in)
        return {"ok": True, "expires_in": expires_in}

    # ------------------------------------------------------------------ #
    #  Token refresh
    # ------------------------------------------------------------------ #

    async def refresh_access_token(self) -> str | None:
        """Refresh the access token using stored refresh token."""
        if not self._refresh_token:
            tokens = await self.load_tokens()
            if not tokens:
                log.warning("spotify_auth.refresh_no_token")
                return None
            self._refresh_token = tokens["refresh_token"]

        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        try:
            resp = await self._http.post(self.TOKEN_URL, data=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.error("spotify_auth.refresh_failed", error=str(exc))
            return None

        data = resp.json()
        access_token = str(data.get("access_token", ""))
        # Spotify may or may not return a new refresh token
        refresh_token = str(data.get("refresh_token", "")) or self._refresh_token
        expires_in = int(data.get("expires_in", 3600))

        if not access_token:
            log.error("spotify_auth.refresh_no_token", response=data)
            return None

        await self.save_tokens(access_token, refresh_token, expires_in)
        log.info("spotify_auth.token_refreshed", expires_in=expires_in)
        return access_token

    # ------------------------------------------------------------------ #
    #  Token access
    # ------------------------------------------------------------------ #

    async def get_valid_token(self) -> str | None:
        """Get a valid access token, refreshing if needed."""
        # Versuche erst den In-Memory-Cache
        if self._access_token and time.time() < self._expires_at:
            return self._access_token

        # Lade aus der DB
        tokens = await self.load_tokens()
        if not tokens:
            return None

        self._access_token = tokens["access_token"]
        self._refresh_token = tokens["refresh_token"]
        self._expires_at = tokens["expires_at"]

        # Noch gueltig?
        if time.time() < (self._expires_at - self._EXPIRY_BUFFER_SECONDS):
            return self._access_token

        # Muss refreshed werden
        log.debug("spotify_auth.token_expired_refreshing")
        refreshed = await self.refresh_access_token()
        return refreshed

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    async def save_tokens(
        self,
        access_token: str,
        refresh_token: str,
        expires_in: int,
    ) -> None:
        """Persist tokens to SQLite."""
        expires_at = time.time() + expires_in
        await self._db.execute(
            """INSERT INTO tokens (user_id, access_token, refresh_token, expires_at, scopes)
               VALUES ('default', ?, ?, ?, ?)
               ON CONFLICT(user_id)
               DO UPDATE SET access_token=excluded.access_token,
                             refresh_token=excluded.refresh_token,
                             expires_at=excluded.expires_at,
                             scopes=excluded.scopes""",
            (access_token, refresh_token, expires_at, self._scopes),
        )
        await self._db.commit()

        # Aktualisiere In-Memory-Cache
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._expires_at = expires_at

    async def load_tokens(self) -> dict[str, Any] | None:
        """Load stored tokens from SQLite."""
        cursor = await self._db.execute(
            "SELECT access_token, refresh_token, expires_at, scopes "
            "FROM tokens WHERE user_id = 'default'"
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "access_token": str(row[0]),
            "refresh_token": str(row[1]),
            "expires_at": float(row[2]),
            "scopes": str(row[3]),
        }

    # ------------------------------------------------------------------ #
    #  Status
    # ------------------------------------------------------------------ #

    async def is_authenticated(self) -> bool:
        """Check if we have valid (or refreshable) tokens."""
        token = await self.get_valid_token()
        return token is not None

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #

    async def close(self) -> None:
        """Close the internal HTTP client."""
        await self._http.aclose()
