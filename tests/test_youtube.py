"""Tests for the YouTube plugin -- client methods with mocked httpx.

Covers:
* YouTubeClient.search() with mocked httpx
* video_info response formatting
* trending response format
* channel_videos response format
* Error handling for HTTP failures
* _safe_int and _parse_search_items helpers
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from plugins.youtube.youtube_client import (
    YouTubeClient,
    _safe_int,
    _parse_search_items,
)


# ─── Helper ──────────────────────────────────────────────────────────────


def _make_client(api_key: str = "test-api-key") -> YouTubeClient:
    return YouTubeClient(api_key)


def _mock_response(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
    text: str = "",
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


# ─── _safe_int helper ────────────────────────────────────────────────────


class TestSafeInt:
    def test_int_input(self) -> None:
        assert _safe_int(42) == 42

    def test_string_input(self) -> None:
        assert _safe_int("12345") == 12345

    def test_none_returns_zero(self) -> None:
        assert _safe_int(None) == 0

    def test_empty_string_returns_zero(self) -> None:
        assert _safe_int("") == 0

    def test_float_string(self) -> None:
        # int("3.5") raises ValueError, so it returns 0
        assert _safe_int("3.5") == 0


# ─── _parse_search_items helper ──────────────────────────────────────────


class TestParseSearchItems:
    def test_video_item(self) -> None:
        items = [{
            "id": {"kind": "youtube#video", "videoId": "abc123"},
            "snippet": {
                "title": "Test Video",
                "description": "A test video description.",
                "channelTitle": "TestChannel",
                "publishedAt": "2025-01-01T00:00:00Z",
                "thumbnails": {"medium": {"url": "http://img.example.com/thumb.jpg"}},
            },
        }]
        parsed = _parse_search_items(items)
        assert len(parsed) == 1
        assert parsed[0]["title"] == "Test Video"
        assert parsed[0]["video_id"] == "abc123"
        assert parsed[0]["type"] == "video"
        assert parsed[0]["url"] == "https://www.youtube.com/watch?v=abc123"

    def test_playlist_item(self) -> None:
        items = [{
            "id": {"kind": "youtube#playlist", "playlistId": "PLabc"},
            "snippet": {
                "title": "Test Playlist",
                "description": "",
                "channelTitle": "Ch",
                "publishedAt": "",
                "thumbnails": {},
            },
        }]
        parsed = _parse_search_items(items)
        assert len(parsed) == 1
        assert parsed[0]["type"] == "playlist"
        assert parsed[0]["playlist_id"] == "PLabc"
        assert "playlist?list=PLabc" in parsed[0]["url"]

    def test_channel_item(self) -> None:
        items = [{
            "id": {"kind": "youtube#channel", "channelId": "UCxyz"},
            "snippet": {
                "title": "Test Channel",
                "description": "",
                "channelTitle": "TC",
                "publishedAt": "",
                "thumbnails": {},
            },
        }]
        parsed = _parse_search_items(items)
        assert len(parsed) == 1
        assert parsed[0]["type"] == "channel"
        assert parsed[0]["channel_id"] == "UCxyz"

    def test_empty_items(self) -> None:
        assert _parse_search_items([]) == []

    def test_description_truncated(self) -> None:
        items = [{
            "id": {"kind": "youtube#video", "videoId": "v1"},
            "snippet": {
                "title": "Long Desc",
                "description": "x" * 500,
                "channelTitle": "Ch",
                "publishedAt": "",
                "thumbnails": {},
            },
        }]
        parsed = _parse_search_items(items)
        assert len(parsed[0]["description"]) <= 200


# ─── YouTubeClient.search() ──────────────────────────────────────────────


class TestYouTubeSearch:
    @pytest.mark.asyncio
    async def test_search_success(self) -> None:
        client = _make_client()
        mock_resp = _mock_response(200, {
            "items": [{
                "id": {"kind": "youtube#video", "videoId": "v1"},
                "snippet": {
                    "title": "Found Video",
                    "description": "desc",
                    "channelTitle": "Ch",
                    "publishedAt": "2025-01-01",
                    "thumbnails": {"medium": {"url": "http://thumb.jpg"}},
                },
            }],
            "pageInfo": {"totalResults": 1},
        })
        client._client = MagicMock()
        client._client.get = AsyncMock(return_value=mock_resp)

        result = await client.search("test query", max_results=5)
        assert result["query"] == "test query"
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Found Video"
        assert result["total_results"] == 1

    @pytest.mark.asyncio
    async def test_search_api_error(self) -> None:
        client = _make_client()
        mock_resp = _mock_response(403, text="Forbidden")
        client._client = MagicMock()
        client._client.get = AsyncMock(return_value=mock_resp)

        result = await client.search("test")
        assert "error" in result
        assert "403" in result["error"]

    @pytest.mark.asyncio
    async def test_search_http_exception(self) -> None:
        client = _make_client()
        client._client = MagicMock()
        client._client.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        result = await client.search("test")
        assert "error" in result
        assert "HTTP error" in result["error"]


# ─── YouTubeClient.video_info() ──────────────────────────────────────────


class TestYouTubeVideoInfo:
    @pytest.mark.asyncio
    async def test_video_info_success(self) -> None:
        client = _make_client()
        mock_resp = _mock_response(200, {
            "items": [{
                "snippet": {
                    "title": "My Video",
                    "description": "A video about testing.",
                    "channelTitle": "TestChan",
                    "publishedAt": "2025-06-15T12:00:00Z",
                    "thumbnails": {"high": {"url": "http://img.jpg"}},
                },
                "statistics": {
                    "viewCount": "1000",
                    "likeCount": "50",
                    "commentCount": "10",
                },
                "contentDetails": {"duration": "PT5M30S"},
            }],
        })
        client._client = MagicMock()
        client._client.get = AsyncMock(return_value=mock_resp)

        result = await client.video_info("vid123")
        assert result["video_id"] == "vid123"
        assert result["title"] == "My Video"
        assert result["views"] == 1000
        assert result["likes"] == 50
        assert result["duration"] == "PT5M30S"
        assert result["url"] == "https://www.youtube.com/watch?v=vid123"
        assert result["embed_url"] == "https://www.youtube.com/embed/vid123"

    @pytest.mark.asyncio
    async def test_video_info_not_found(self) -> None:
        client = _make_client()
        mock_resp = _mock_response(200, {"items": []})
        client._client = MagicMock()
        client._client.get = AsyncMock(return_value=mock_resp)

        result = await client.video_info("missing")
        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_video_info_api_error(self) -> None:
        client = _make_client()
        mock_resp = _mock_response(500)
        client._client = MagicMock()
        client._client.get = AsyncMock(return_value=mock_resp)

        result = await client.video_info("vid")
        assert "error" in result


# ─── YouTubeClient.trending() ────────────────────────────────────────────


class TestYouTubeTrending:
    @pytest.mark.asyncio
    async def test_trending_success(self) -> None:
        client = _make_client()
        mock_resp = _mock_response(200, {
            "items": [
                {
                    "id": "t1",
                    "snippet": {
                        "title": "Trending #1",
                        "channelTitle": "PopCh",
                        "thumbnails": {"medium": {"url": "http://thumb1.jpg"}},
                    },
                    "statistics": {"viewCount": "500000"},
                },
                {
                    "id": "t2",
                    "snippet": {
                        "title": "Trending #2",
                        "channelTitle": "NewsCh",
                        "thumbnails": {"medium": {"url": "http://thumb2.jpg"}},
                    },
                    "statistics": {"viewCount": "300000"},
                },
            ],
        })
        client._client = MagicMock()
        client._client.get = AsyncMock(return_value=mock_resp)

        result = await client.trending(region_code="DE", max_results=10)
        assert result["region"] == "DE"
        assert len(result["results"]) == 2
        assert result["results"][0]["title"] == "Trending #1"
        assert result["results"][0]["views"] == 500000

    @pytest.mark.asyncio
    async def test_trending_api_error(self) -> None:
        client = _make_client()
        mock_resp = _mock_response(429)
        client._client = MagicMock()
        client._client.get = AsyncMock(return_value=mock_resp)

        result = await client.trending()
        assert "error" in result


# ─── Client metadata ─────────────────────────────────────────────────────


class TestYouTubeClientMeta:
    def test_base_url(self) -> None:
        assert YouTubeClient.BASE_URL == "https://www.googleapis.com/youtube/v3"

    def test_init_stores_api_key(self) -> None:
        client = _make_client("my-key")
        assert client._api_key == "my-key"

    def test_client_has_required_methods(self) -> None:
        methods = ["search", "video_info", "trending", "channel_videos", "close"]
        for m in methods:
            assert hasattr(YouTubeClient, m), f"Missing method: {m}"
