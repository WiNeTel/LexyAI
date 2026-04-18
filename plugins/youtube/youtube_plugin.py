"""
Lexy AI - YouTube Plugin.

Provides YouTube search, video info, playback control, trending videos,
and channel browsing via YouTube Data API v3. Playback is routed to the
frontend through WebSocket broadcasts.
"""

from __future__ import annotations

from typing import Any

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

from .youtube_client import YouTubeClient

log = get_logger(module="youtube_plugin")

# Returned when no API key is set — keeps the message consistent across tools.
_NO_KEY_ERROR: dict[str, str] = {
    "error": (
        "YouTube API key not configured. "
        "Set it in config/plugins.yaml under youtube.api_key"
    ),
}

# ─── Tool schemas ────────────────────────────────────────────────

SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query (e.g. 'Python tutorial', 'Lo-Fi beats')",
        },
        "max_results": {
            "type": "integer",
            "description": "Number of results (1-25, default from config)",
        },
        "type": {
            "type": "string",
            "enum": ["video", "playlist", "channel"],
            "description": "Result type filter (default: video)",
        },
    },
    "required": ["query"],
}

VIDEO_INFO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "video_id": {
            "type": "string",
            "description": "YouTube video ID (e.g. 'dQw4w9WgXcQ')",
        },
    },
    "required": ["video_id"],
}

PLAY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "video_id": {
            "type": "string",
            "description": "YouTube video ID to play in the frontend",
        },
    },
    "required": ["video_id"],
}

TRENDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "description": (
                "Video category ID (0=all, 10=music, 20=gaming, 24=entertainment, "
                "25=news, 28=science)"
            ),
        },
        "region": {
            "type": "string",
            "description": "ISO 3166-1 alpha-2 region code (default: DE)",
        },
    },
    "required": [],
}

CHANNEL_VIDEOS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "channel_id": {
            "type": "string",
            "description": "YouTube channel ID (e.g. 'UC...')",
        },
        "max_results": {
            "type": "integer",
            "description": "Number of results (1-25, default 10)",
        },
    },
    "required": ["channel_id"],
}


class YouTubePlugin(BasePlugin):
    """YouTube search, info, and frontend playback control."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._client: YouTubeClient | None = None
        self._api_key: str = ""
        self._max_results: int = 5
        self._region_code: str = "DE"
        self._safe_search: str = "moderate"

    # ─── Lifecycle ───────────────────────────────────────────────

    async def on_load(self) -> None:
        """Read config and create the YouTube API client."""
        config = self.api.get_config()
        self._api_key = str(config.get("api_key", ""))
        self._max_results = int(config.get("max_results", 5))
        self._region_code = str(config.get("region_code", "DE"))
        self._safe_search = str(config.get("safe_search", "moderate"))

        if self._api_key:
            self._client = YouTubeClient(self._api_key)
            log.info("youtube.client_created", region=self._region_code)
        else:
            log.warning("youtube.no_api_key", hint="Set youtube.api_key in plugins.yaml")

    async def on_enable(self) -> None:
        """Register tools and WebSocket handlers."""
        # --- Tools ---
        self.api.register_tool(
            name="youtube_search",
            handler=self._tool_search,
            description=(
                "Search YouTube for videos, playlists, or channels. "
                "Returns titles, thumbnails, and URLs."
            ),
            schema=SEARCH_SCHEMA,
        )
        self.api.register_tool(
            name="youtube_video_info",
            handler=self._tool_video_info,
            description=(
                "Get detailed information about a YouTube video: "
                "title, duration, views, likes, description."
            ),
            schema=VIDEO_INFO_SCHEMA,
        )
        self.api.register_tool(
            name="youtube_play",
            handler=self._tool_play,
            description=(
                "Play a YouTube video in the frontend player. "
                "Sends an embed command over WebSocket."
            ),
            schema=PLAY_SCHEMA,
        )
        self.api.register_tool(
            name="youtube_trending",
            handler=self._tool_trending,
            description="Get currently trending YouTube videos for a region.",
            schema=TRENDING_SCHEMA,
        )
        self.api.register_tool(
            name="youtube_channel_videos",
            handler=self._tool_channel_videos,
            description="Get the most recent videos from a YouTube channel.",
            schema=CHANNEL_VIDEOS_SCHEMA,
        )

        # --- WebSocket handlers ---
        self.api.register_ws_handler("youtube_search_request", self._ws_search)
        self.api.register_ws_handler("youtube_play_request", self._ws_play)

        log.info("youtube.enabled", tools=5, ws_handlers=2)

    async def on_disable(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
        log.info("youtube.disabled")

    # ─── Tool handlers ───────────────────────────────────────────

    async def _tool_search(
        self,
        query: str,
        max_results: int | None = None,
        type: str | None = None,
    ) -> dict[str, Any]:
        """Tool: youtube_search."""
        if not self._api_key or self._client is None:
            return dict(_NO_KEY_ERROR)

        count = max_results if max_results is not None else self._max_results
        count = max(1, min(25, count))
        search_type = type if type in ("video", "playlist", "channel") else "video"

        return await self._client.search(
            query=query,
            max_results=count,
            search_type=search_type,
            region_code=self._region_code,
            safe_search=self._safe_search,
        )

    async def _tool_video_info(self, video_id: str) -> dict[str, Any]:
        """Tool: youtube_video_info."""
        if not self._api_key or self._client is None:
            return dict(_NO_KEY_ERROR)

        return await self._client.video_info(video_id)

    async def _tool_play(self, video_id: str) -> dict[str, Any]:
        """Tool: youtube_play. Fetches info then broadcasts to frontend."""
        if not self._api_key or self._client is None:
            return dict(_NO_KEY_ERROR)

        info = await self._client.video_info(video_id)
        if "error" in info:
            return info

        # Broadcast play command to all connected frontend clients
        await self.api.ws_broadcast({
            "type": "youtube_play",
            "video_id": video_id,
            "embed_url": info["embed_url"],
            "title": info["title"],
            "channel": info["channel"],
            "thumbnail": info["thumbnail"],
            "duration": info.get("duration", ""),
        })

        log.info("youtube.play_broadcast", video_id=video_id, title=info["title"])
        return {
            "status": "playing",
            "video_id": video_id,
            "title": info["title"],
            "embed_url": info["embed_url"],
        }

    async def _tool_trending(
        self,
        category: str | None = None,
        region: str | None = None,
    ) -> dict[str, Any]:
        """Tool: youtube_trending."""
        if not self._api_key or self._client is None:
            return dict(_NO_KEY_ERROR)

        return await self._client.trending(
            region_code=region or self._region_code,
            category_id=category or "0",
            max_results=self._max_results,
        )

    async def _tool_channel_videos(
        self,
        channel_id: str,
        max_results: int | None = None,
    ) -> dict[str, Any]:
        """Tool: youtube_channel_videos."""
        if not self._api_key or self._client is None:
            return dict(_NO_KEY_ERROR)

        count = max_results if max_results is not None else 10
        count = max(1, min(25, count))

        return await self._client.channel_videos(
            channel_id=channel_id,
            max_results=count,
        )

    # ─── WebSocket handlers ──────────────────────────────────────

    async def _ws_search(self, data: dict[str, Any]) -> None:
        """Handle frontend search requests via WebSocket."""
        query = str(data.get("query", ""))
        if not query:
            await self.api.ws_broadcast({
                "type": "youtube_search_response",
                "error": "Empty search query",
            })
            return

        max_results = data.get("max_results", self._max_results)
        search_type = data.get("search_type", "video")

        result = await self._tool_search(
            query=query,
            max_results=max_results,
            type=search_type,
        )
        await self.api.ws_broadcast({
            "type": "youtube_search_response",
            **result,
        })

    async def _ws_play(self, data: dict[str, Any]) -> None:
        """Handle frontend play requests via WebSocket."""
        video_id = str(data.get("video_id", ""))
        if not video_id:
            await self.api.ws_broadcast({
                "type": "youtube_play_error",
                "error": "No video_id provided",
            })
            return

        result = await self._tool_play(video_id=video_id)
        if "error" in result:
            await self.api.ws_broadcast({
                "type": "youtube_play_error",
                **result,
            })
