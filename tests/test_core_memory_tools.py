"""
Phase 12.A — Tests for the core memory tools.

Mike's question: "Wenn ich Lexy sage 'merke dir das', sollte das nicht
unter Facts gespeichert werden?" — pre-Phase-12 the answer was no
because ``memory_store`` was only available to plugins, not as an LLM
tool in the main chat. These tests pin the new contract: after app
startup, both ``memory_store`` and ``memory_recall`` are in the
tool_registry, the handlers persist + retrieve via
``MemoryManager``, and bad inputs are rejected cleanly.

We boot a real :class:`LexyApp` (via the existing
``test_gateway`` pattern) so the tool wiring goes through the same
startup path production uses.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient

from lexy_core.app import LexyApp


@pytest.fixture(scope="module")
def lexy_client() -> TestClient:
    app = LexyApp("config/config.yaml")
    asyncio.get_event_loop().run_until_complete(app.startup())
    client = TestClient(app.fastapi)
    yield client
    asyncio.get_event_loop().run_until_complete(app.shutdown())


def _app(client: TestClient) -> LexyApp:
    return client.app.state.lexy  # type: ignore[attr-defined]


# ─── Tool registration ──────────────────────────────────────────────


def test_memory_store_registered(lexy_client: TestClient) -> None:
    """``memory_store`` lives in the tool_registry after startup."""
    app = _app(lexy_client)
    assert app.tool_registry is not None
    tool = app.tool_registry.get_tool("memory_store")
    assert tool is not None
    assert tool.source == "core"
    # Schema check — required field 'text' present.
    assert "text" in tool.schema.get("properties", {})
    assert "text" in tool.schema.get("required", [])


def test_memory_recall_registered(lexy_client: TestClient) -> None:
    """``memory_recall`` lives in the tool_registry after startup."""
    app = _app(lexy_client)
    assert app.tool_registry is not None
    tool = app.tool_registry.get_tool("memory_recall")
    assert tool is not None
    assert tool.source == "core"
    assert "query" in tool.schema.get("properties", {})


# ─── Handler behaviour ──────────────────────────────────────────────


def test_memory_store_persists_to_facts_collection(
    lexy_client: TestClient,
) -> None:
    """Calling the tool with a fact lands a row in the facts collection."""
    app = _app(lexy_client)
    assert app.memory is not None
    fact = f"Test fact {uuid.uuid4().hex[:8]} — Mike wohnt am Nordpol"

    result = asyncio.get_event_loop().run_until_complete(
        app._tool_memory_store(text=fact, collection="facts")
    )
    assert result["ok"] is True
    assert result["collection"] == "facts"
    assert result["id"]


def test_memory_store_rejects_empty_text(
    lexy_client: TestClient,
) -> None:
    app = _app(lexy_client)
    result = asyncio.get_event_loop().run_until_complete(
        app._tool_memory_store(text="")
    )
    assert result["ok"] is False
    assert "required" in result["error"].lower()


def test_memory_store_coerces_invalid_collection(
    lexy_client: TestClient,
) -> None:
    """An LLM that hallucinates a collection name shouldn't blow up —
    we silently fall back to 'facts'."""
    app = _app(lexy_client)
    result = asyncio.get_event_loop().run_until_complete(
        app._tool_memory_store(
            text="Coerce-test fact", collection="galactic"
        )
    )
    assert result["ok"] is True
    assert result["collection"] == "facts"


def test_memory_recall_finds_stored_fact(
    lexy_client: TestClient,
) -> None:
    """Round-trip: store a fact, then recall by a related query."""
    app = _app(lexy_client)
    marker = uuid.uuid4().hex[:8]
    fact = f"Round-trip {marker} — Mikes Lieblingscafé heisst Cafe Roma"
    asyncio.get_event_loop().run_until_complete(
        app._tool_memory_store(text=fact, collection="facts")
    )

    result = asyncio.get_event_loop().run_until_complete(
        app._tool_memory_recall(
            query=f"Lieblingscafé {marker}",
            collection="facts",
            limit=5,
        )
    )
    assert result["ok"] is True
    assert result["count"] >= 1
    # The fact we just stored is in the hits.
    assert any(marker in h.get("text", "") for h in result["hits"])


def test_memory_recall_rejects_empty_query(
    lexy_client: TestClient,
) -> None:
    app = _app(lexy_client)
    result = asyncio.get_event_loop().run_until_complete(
        app._tool_memory_recall(query="")
    )
    assert result["ok"] is False


def test_memory_recall_clamps_limit(lexy_client: TestClient) -> None:
    """Out-of-range limit values are clamped to [1, 20]."""
    app = _app(lexy_client)
    # Sanity: a stored item exists so recall has something to clamp.
    asyncio.get_event_loop().run_until_complete(
        app._tool_memory_store(
            text="Clamp-limit-fact", collection="facts"
        )
    )
    # Too-large limit: clamps to 20, no error.
    big = asyncio.get_event_loop().run_until_complete(
        app._tool_memory_recall(query="Clamp-limit", limit=999)
    )
    assert big["ok"] is True
    # Negative limit: clamps to 1, no error.
    small = asyncio.get_event_loop().run_until_complete(
        app._tool_memory_recall(query="Clamp-limit", limit=-5)
    )
    assert small["ok"] is True


def test_memory_store_tags_optional(lexy_client: TestClient) -> None:
    """``tags`` is optional and lands as a comma-separated string in metadata."""
    app = _app(lexy_client)
    result = asyncio.get_event_loop().run_until_complete(
        app._tool_memory_store(
            text="Fact with tags",
            collection="facts",
            tags="user_info,address",
        )
    )
    assert result["ok"] is True
