"""
Phase 11 hotfix — REST tests for the per-session character_turns endpoint.

Mike reported that resuming an RP session showed only user messages, no
character bubbles. Cause: ``/sessions/{id}/history`` reads the agent's
session_store, but character bubbles persist in a separate table
(``data/plugins/character_chat/character_chat.db`` →
``character_turns``). The new REST endpoint
``GET /api/v1/plugins/character_chat/sessions/{session_id}/turns``
exposes that table so the frontend can interleave on resume.

These tests insert rows directly into the plugin's DB connection, then
hit the REST endpoint and verify the JSON shape + chronological order.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from lexy_core.app import LexyApp


def _uniq(prefix: str) -> str:
    """Tests run against the real persistent DB, so re-runs collide on
    primary keys without per-run uniqueness. Use UUID suffixes to keep
    each run's rows distinct without polluting the DB schema with extra
    columns."""
    return f"{prefix}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def lexy_client() -> TestClient:
    app = LexyApp("config/config.yaml")
    asyncio.get_event_loop().run_until_complete(app.startup())
    client = TestClient(app.fastapi)
    yield client
    asyncio.get_event_loop().run_until_complete(app.shutdown())


async def _insert_turn(
    plugin,
    *,
    turn_id: str,
    session_id: str,
    character_id: str,
    character_name: str,
    round_id: str,
    order_num: int,
    content: str,
    trigger_kind: str = "user",
    trigger_text: str = "",
    reasoning: str = "",
    created_at: float | None = None,
) -> None:
    db = await plugin.api.get_db()
    await db.execute(
        "INSERT INTO character_turns (id, session_id, character_id, "
        "character_name, round_id, order_num, content, skipped, "
        "trigger_kind, trigger_text, reasoning, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)",
        (
            turn_id, session_id, character_id, character_name, round_id,
            order_num, content, trigger_kind, trigger_text, reasoning,
            created_at if created_at is not None else time.time(),
        ),
    )
    await db.commit()


def _get_plugin(lexy_client: TestClient):
    # FastAPI app instance is on the TestClient; pull the plugin via
    # the same path the gateway uses internally — ``request.app.state.lexy``.
    app = lexy_client.app.state.lexy  # type: ignore[attr-defined]
    return app.plugin_loader.get_plugin("character_chat")


# ─── Empty-result path ──────────────────────────────────────────────


def test_unknown_session_returns_empty_turns(lexy_client: TestClient) -> None:
    """A session with no rows yields ``{turns: []}`` (not 404).

    The frontend's resume flow can't distinguish "no RP session yet"
    from "RP session with no rounds yet" — and neither should error.
    """
    resp = lexy_client.get(
        "/api/v1/plugins/character_chat/sessions/__never-existed__/turns"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "__never-existed__"
    assert body["turns"] == []


# ─── Happy path with mixed rounds ───────────────────────────────────


def test_turns_returned_chronologically_with_full_metadata(
    lexy_client: TestClient,
) -> None:
    """Insert two rounds across one session, verify JSON shape."""
    sid = _uniq("test-rp-history-happy")
    plugin = _get_plugin(lexy_client)
    assert plugin is not None

    base = time.time()
    t1 = _uniq("t")
    t2 = _uniq("t")
    t3 = _uniq("t")
    asyncio.get_event_loop().run_until_complete(_insert_turn(
        plugin,
        turn_id=t1, session_id=sid,
        character_id="c1", character_name="Alpha",
        round_id="r1", order_num=0,
        content="Alpha sagt hi", trigger_text="Hallo zusammen",
        reasoning="Alpha denkt kurz nach",
        created_at=base,
    ))
    asyncio.get_event_loop().run_until_complete(_insert_turn(
        plugin,
        turn_id=t2, session_id=sid,
        character_id="c2", character_name="Beta",
        round_id="r1", order_num=1,
        content="Beta sagt auch hi", trigger_text="Hallo zusammen",
        created_at=base + 0.1,
    ))
    asyncio.get_event_loop().run_until_complete(_insert_turn(
        plugin,
        turn_id=t3, session_id=sid,
        character_id="c1", character_name="Alpha",
        round_id="r2", order_num=0,
        content="Alpha antwortet", trigger_text="Wie geht's?",
        created_at=base + 1.0,
    ))

    resp = lexy_client.get(
        f"/api/v1/plugins/character_chat/sessions/{sid}/turns"
    )
    assert resp.status_code == 200
    turns = resp.json()["turns"]
    assert len(turns) == 3

    # Chronological order — earliest created_at first.
    assert turns[0]["turn_id"] == t1
    assert turns[1]["turn_id"] == t2
    assert turns[2]["turn_id"] == t3

    # Schema check: every field the frontend interleaver consumes.
    for t in turns:
        for key in (
            "turn_id", "character_id", "character_name", "round_id",
            "order", "content", "skipped", "trigger_kind",
            "trigger_text", "reasoning", "created_at",
        ):
            assert key in t, f"missing {key} in {t}"
    assert turns[0]["character_name"] == "Alpha"
    assert turns[2]["round_id"] == "r2"
    # Display-only reasoning round-trips through persistence + the endpoint.
    assert turns[0]["reasoning"] == "Alpha denkt kurz nach"
    assert turns[1]["reasoning"] == ""  # absent → empty default


def test_limit_param_caps_returned_rows(lexy_client: TestClient) -> None:
    """``?limit=N`` clamps the SELECT and we surface oldest-first."""
    sid = _uniq("test-rp-history-limit")
    plugin = _get_plugin(lexy_client)
    base = time.time()
    for i in range(5):
        asyncio.get_event_loop().run_until_complete(_insert_turn(
            plugin,
            turn_id=_uniq(f"t-lim-{i}"), session_id=sid,
            character_id="c1", character_name="Alpha",
            round_id=f"r-{i}", order_num=0,
            content=f"turn {i}", trigger_text=f"msg {i}",
            created_at=base + i,
        ))

    resp = lexy_client.get(
        f"/api/v1/plugins/character_chat/sessions/{sid}/turns?limit=2"
    )
    assert resp.status_code == 200
    turns = resp.json()["turns"]
    assert len(turns) == 2
    # The oldest two come first (chronological ASC).
    assert turns[0]["content"] == "turn 0"
    assert turns[1]["content"] == "turn 1"


def test_turns_isolated_per_session(lexy_client: TestClient) -> None:
    """Inserting into session A must not leak into session B's response."""
    sid_a = _uniq("test-rp-iso-a")
    sid_b = _uniq("test-rp-iso-b")
    plugin = _get_plugin(lexy_client)
    base = time.time()
    t_a = _uniq("t-iso-a")
    t_b = _uniq("t-iso-b")
    asyncio.get_event_loop().run_until_complete(_insert_turn(
        plugin,
        turn_id=t_a, session_id=sid_a,
        character_id="c1", character_name="Alpha",
        round_id="r-a", order_num=0,
        content="A only", created_at=base,
    ))
    asyncio.get_event_loop().run_until_complete(_insert_turn(
        plugin,
        turn_id=t_b, session_id=sid_b,
        character_id="c1", character_name="Alpha",
        round_id="r-b", order_num=0,
        content="B only", created_at=base + 1,
    ))

    resp_a = lexy_client.get(
        f"/api/v1/plugins/character_chat/sessions/{sid_a}/turns"
    )
    resp_b = lexy_client.get(
        f"/api/v1/plugins/character_chat/sessions/{sid_b}/turns"
    )
    a_turns = [t["turn_id"] for t in resp_a.json()["turns"]]
    b_turns = [t["turn_id"] for t in resp_b.json()["turns"]]
    assert t_a in a_turns and t_b not in a_turns
    assert t_b in b_turns and t_a not in b_turns
