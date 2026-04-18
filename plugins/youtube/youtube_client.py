"""
Lexy AI - YouTube Data API v3 async client.

Wraps search, video info, trending, and channel queries via httpx.
"""

from __future__ import annotations

from typing import Any

import httpx

from lexy_core.utils.logging import get_logger

log = get_logger(module="youtube_client")


class YouTubeClient:
    """Async YouTube Data API v3 client."""

    BASE_URL = "https://www.googleapis.com/youtube/v3"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = httpx.AsyncClient(base_url=self.BASE_URL, timeout=10.0)

    # ─── Search ──────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_type: str = "video",
        region_code: str = "DE",
        safe_search: str = "moderate",
    ) -> dict[str, Any]:
        """Search YouTube videos/playlists/channels."""
        params: dict[str, Any] = {
            "part": "snippet",
            "q": query,
            "maxResults": max_results,
            "type": search_type,
            "regionCode": region_code,
            "safeSearch": safe_search,
            "key": self._api_key,
        }
        try:
            resp = await self._client.get("/search", params=params)
        except httpx.HTTPError as exc:
            log.error("youtube.search_failed", error=str(exc), query=query)
            return {"error": f"HTTP error: {exc}", "query": query}

        if resp.status_code != 200:
            log.warning(
                "youtube.search_api_error",
                status=resp.status_code,
                query=query,
            )
            return {
                "error": f"YouTube API error {resp.status_code}",
                "detail": resp.text[:200],
            }

        data = resp.json()
        results = _parse_search_items(data.get("items", []))
        total = data.get("pageInfo", {}).get("totalResults", 0)

        log.info("youtube.search_ok", query=query, count=len(results))
        return {"query": query, "results": results, "total_results": total}

    # ─── Video info ──────────────────────────────────────────────

    async def video_info(self, video_id: str) -> dict[str, Any]:
        """Get detailed info for a single video."""
        params: dict[str, Any] = {
            "part": "snippet,contentDetails,statistics",
            "id": video_id,
            "key": self._api_key,
        }
        try:
            resp = await self._client.get("/videos", params=params)
        except httpx.HTTPError as exc:
            log.error("youtube.video_info_failed", error=str(exc), video_id=video_id)
            return {"error": f"HTTP error: {exc}"}

        if resp.status_code != 200:
            return {"error": f"YouTube API error {resp.status_code}"}

        items = resp.json().get("items", [])
        if not items:
            return {"error": "Video not found", "video_id": video_id}

        item = items[0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        details = item.get("contentDetails", {})

        return {
            "video_id": video_id,
            "title": snippet.get("title", ""),
            "description": snippet.get("description", "")[:500],
            "channel": snippet.get("channelTitle", ""),
            "published_at": snippet.get("publishedAt", ""),
            "duration": details.get("duration", ""),
            "views": _safe_int(stats.get("viewCount", 0)),
            "likes": _safe_int(stats.get("likeCount", 0)),
            "comments": _safe_int(stats.get("commentCount", 0)),
            "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "embed_url": f"https://www.youtube.com/embed/{video_id}",
        }

    # ─── Trending ────────────────────────────────────────────────

    async def trending(
        self,
        region_code: str = "DE",
        category_id: str = "0",
        max_results: int = 10,
    ) -> dict[str, Any]:
        """Get trending videos for a region."""
        params: dict[str, Any] = {
            "part": "snippet,statistics",
            "chart": "mostPopular",
            "regionCode": region_code,
            "videoCategoryId": category_id,
            "maxResults": max_results,
            "key": self._api_key,
        }
        try:
            resp = await self._client.get("/videos", params=params)
        except httpx.HTTPError as exc:
            log.error("youtube.trending_failed", error=str(exc))
            return {"error": f"HTTP error: {exc}"}

        if resp.status_code != 200:
            return {"error": f"YouTube API error {resp.status_code}"}

        data = resp.json()
        results: list[dict[str, Any]] = []
        for item in data.get("items", []):
            vid_id = item.get("id", "")
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            results.append({
                "video_id": vid_id,
                "title": snippet.get("title", ""),
                "channel": snippet.get("channelTitle", ""),
                "views": _safe_int(stats.get("viewCount", 0)),
                "thumbnail": (
                    snippet.get("thumbnails", {}).get("medium", {}).get("url", "")
                ),
                "url": f"https://www.youtube.com/watch?v={vid_id}",
            })

        log.info("youtube.trending_ok", region=region_code, count=len(results))
        return {"region": region_code, "results": results}

    # ─── Channel videos ──────────────────────────────────────────

    async def channel_videos(
        self,
        channel_id: str,
        max_results: int = 10,
    ) -> dict[str, Any]:
        """Get recent videos from a channel."""
        params: dict[str, Any] = {
            "part": "snippet",
            "channelId": channel_id,
            "maxResults": max_results,
            "order": "date",
            "type": "video",
            "key": self._api_key,
        }
        try:
            resp = await self._client.get("/search", params=params)
        except httpx.HTTPError as exc:
            log.error(
                "youtube.channel_videos_failed",
                error=str(exc),
                channel_id=channel_id,
            )
            return {"error": f"HTTP error: {exc}"}

        if resp.status_code != 200:
            return {"error": f"YouTube API error {resp.status_code}"}

        data = resp.json()
        results: list[dict[str, Any]] = []
        for item in data.get("items", []):
            vid_id = item.get("id", {}).get("videoId", "")
            snippet = item.get("snippet", {})
            results.append({
                "video_id": vid_id,
                "title": snippet.get("title", ""),
                "published_at": snippet.get("publishedAt", ""),
                "thumbnail": (
                    snippet.get("thumbnails", {}).get("medium", {}).get("url", "")
                ),
                "url": f"https://www.youtube.com/watch?v={vid_id}",
            })

        log.info(
            "youtube.channel_videos_ok",
            channel_id=channel_id,
            count=len(results),
        )
        return {"channel_id": channel_id, "results": results}

    # ─── Lifecycle ───────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()
        log.debug("youtube.client_closed")


# ─── Helpers ─────────────────────────────────────────────────────


def _safe_int(value: Any) -> int:
    """Convert to int without raising on empty/None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_search_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transform raw YouTube search items into a clean list."""
    results: list[dict[str, Any]] = []
    for item in items:
        snippet = item.get("snippet", {})
        vid_id = item.get("id", {})
        result: dict[str, Any] = {
            "title": snippet.get("title", ""),
            "description": snippet.get("description", "")[:200],
            "channel": snippet.get("channelTitle", ""),
            "published_at": snippet.get("publishedAt", ""),
            "thumbnail": (
                snippet.get("thumbnails", {}).get("medium", {}).get("url", "")
            ),
        }
        kind = vid_id.get("kind", "")
        if kind == "youtube#video":
            result["video_id"] = vid_id.get("videoId", "")
            result["url"] = f"https://www.youtube.com/watch?v={result['video_id']}"
            result["type"] = "video"
        elif kind == "youtube#playlist":
            result["playlist_id"] = vid_id.get("playlistId", "")
            result["url"] = (
                f"https://www.youtube.com/playlist?list={result['playlist_id']}"
            )
            result["type"] = "playlist"
        elif kind == "youtube#channel":
            result["channel_id"] = vid_id.get("channelId", "")
            result["url"] = (
                f"https://www.youtube.com/channel/{result['channel_id']}"
            )
            result["type"] = "channel"
        results.append(result)
    return results
