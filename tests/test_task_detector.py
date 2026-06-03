"""
Tests for Phase-P4 autonomous skill learning.

Three pure-ish layers:

1. ``TaskDetector`` — when is a turn skill-worthy? Complex single turn,
   repeated sequence, RP guard, idempotency.
2. ``_parse_skill_draft`` — tolerant JSON parsing of the author brain's draft.
3. The plugin's ``after_response_send`` listener gating — fires
   ``_auto_create_skill`` only for real signals, never for RP / when disabled.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from plugins.skill_writer.skill_writer_plugin import (
    SkillWriterPlugin,
    _parse_skill_draft,
)
from plugins.skill_writer.task_detector import (
    TaskDetector,
    is_rp_tool,
    signature_of,
    tool_name,
)


# ─── Detector ───────────────────────────────────────────────────────────────


def test_complex_turn_signals() -> None:
    det = TaskDetector(complex_threshold=4, repeat_threshold=3)
    signal = det.record(["a", "b", "c", "d"])
    assert signal is not None
    assert signal.reason == "complex"
    assert signal.tools == ["a", "b", "c", "d"]


def test_below_complex_needs_repetition() -> None:
    det = TaskDetector(complex_threshold=8, repeat_threshold=3)
    assert det.record(["x", "y"]) is None
    assert det.record(["x", "y"]) is None
    signal = det.record(["x", "y"])  # 3rd time
    assert signal is not None
    assert signal.reason == "repeated"


def test_signature_is_order_sensitive() -> None:
    det = TaskDetector(complex_threshold=8, repeat_threshold=2)
    assert det.record(["a", "b"]) is None
    assert det.record(["b", "a"]) is None  # different signature → separate count
    assert det.record(["a", "b"]) is not None  # ab seen twice


def test_rp_tools_are_never_learned() -> None:
    det = TaskDetector(complex_threshold=2, repeat_threshold=1)
    assert det.record(["run_character_round", "narrate_scene"]) is None
    assert det.record(["fetch", "run_character_round"]) is None


def test_empty_turn_no_signal() -> None:
    det = TaskDetector()
    assert det.record([]) is None
    assert det.record([{}, {"foo": "bar"}]) is None


def test_signature_fires_only_once() -> None:
    det = TaskDetector(complex_threshold=2, repeat_threshold=1)
    assert det.record(["a", "b"]) is not None
    assert det.record(["a", "b"]) is None  # already signalled


def test_tool_name_extraction() -> None:
    assert tool_name("run_skill") == "run_skill"
    assert tool_name({"tool": "read_pdf"}) == "read_pdf"
    assert tool_name({"name": "fetch"}) == "fetch"
    assert tool_name({"nope": "x"}) == ""
    assert tool_name(123) == ""


def test_dict_tools_used_shape() -> None:
    det = TaskDetector(complex_threshold=3, repeat_threshold=3)
    signal = det.record(
        [{"tool": "read_pdf"}, {"tool": "parse_table"}, {"tool": "to_json"}]
    )
    assert signal is not None
    assert signal.signature == signature_of(["read_pdf", "parse_table", "to_json"])


def test_is_rp_tool() -> None:
    assert is_rp_tool("run_character_round")
    assert is_rp_tool("narrate_scene")
    assert not is_rp_tool("read_pdf")


# ─── Draft parser ───────────────────────────────────────────────────────────


def test_parse_valid_json() -> None:
    raw = json.dumps(
        {"name": "parse-pdf", "description": "extract tables", "code": "return {}"}
    )
    assert _parse_skill_draft(raw) == ("parse-pdf", "extract tables", "return {}")


def test_parse_strips_code_fence() -> None:
    raw = "```json\n" + json.dumps(
        {"name": "x", "description": "d", "code": "return 1"}
    ) + "\n```"
    assert _parse_skill_draft(raw) == ("x", "d", "return 1")


def test_parse_missing_field_returns_none() -> None:
    assert _parse_skill_draft(json.dumps({"name": "x", "description": "d"})) is None
    assert _parse_skill_draft(json.dumps({"name": "", "description": "d", "code": "c"})) is None


def test_parse_garbage_returns_none() -> None:
    assert _parse_skill_draft("not json at all") is None
    assert _parse_skill_draft("") is None


# ─── Listener gating ────────────────────────────────────────────────────────


def _listener_plugin() -> SkillWriterPlugin:
    plugin = SkillWriterPlugin.__new__(SkillWriterPlugin)
    plugin._auto_learn_skills = True  # noqa: SLF001
    plugin._detector = TaskDetector(complex_threshold=3, repeat_threshold=3)  # noqa: SLF001
    plugin._auto_create_skill = AsyncMock()  # type: ignore[method-assign]
    return plugin


@pytest.mark.asyncio
async def test_listener_fires_on_complex_turn() -> None:
    plugin = _listener_plugin()
    await plugin._on_response_for_skill_learning(
        {"tools_used": ["a", "b", "c"], "text": "do the thing"}
    )
    plugin._auto_create_skill.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_listener_skips_rp_turn() -> None:
    plugin = _listener_plugin()
    await plugin._on_response_for_skill_learning(
        {"tools_used": ["run_character_round", "narrate_scene", "x"], "text": "rp"}
    )
    plugin._auto_create_skill.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_listener_disabled_does_nothing() -> None:
    plugin = _listener_plugin()
    plugin._auto_learn_skills = False  # noqa: SLF001
    await plugin._on_response_for_skill_learning(
        {"tools_used": ["a", "b", "c"], "text": "x"}
    )
    plugin._auto_create_skill.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_listener_ignores_thin_turn() -> None:
    plugin = _listener_plugin()
    await plugin._on_response_for_skill_learning({"tools_used": ["a"], "text": "x"})
    plugin._auto_create_skill.assert_not_awaited()  # type: ignore[attr-defined]
