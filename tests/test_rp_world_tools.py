"""Tests for plugins/character_chat/rp_world_tools.py (pure authoring helpers)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from plugins.character_chat import rp_world_tools as rwt

_THRESHOLDS = [
    {"at": 70.0, "need": "feed_baby", "urgency": 1},
    {"at": 100.0, "need": "baby_sick", "urgency": 3},
]


def test_define_need_builds_specs_and_state() -> None:
    world = rwt.define_need(
        {},
        entity="baby",
        attribute="hunger",
        value=40.0,
        rate_per_minute=2.5,
        thresholds=_THRESHOLDS,
        minutes_per_tick=2.0,
    )
    assert len(world["specs"]) == 1
    # rate_per_minute 2.5 @ 2 min/tick → rate_per_tick 5.0 in the materialised state
    attr = world["state"]["baby"]["attributes"]["hunger"]
    assert attr["rate_per_tick"] == 5.0
    assert attr["value"] == 40.0
    assert len(attr["thresholds"]) == 2


def test_snapshot_reports_values() -> None:
    world = rwt.define_need(
        {}, entity="baby", attribute="hunger", value=40.0,
        rate_per_minute=2.5, thresholds=_THRESHOLDS, minutes_per_tick=2.0,
    )
    assert rwt.snapshot(world) == {"baby": {"hunger": 40.0}}


def test_redefining_same_entity_attribute_replaces() -> None:
    world = rwt.define_need(
        {}, entity="baby", attribute="hunger", value=40.0,
        rate_per_minute=2.5, thresholds=_THRESHOLDS, minutes_per_tick=2.0,
    )
    world = rwt.define_need(
        world, entity="baby", attribute="hunger", value=10.0,
        rate_per_minute=1.0, thresholds=_THRESHOLDS, minutes_per_tick=2.0,
    )
    assert len(world["specs"]) == 1
    assert rwt.snapshot(world) == {"baby": {"hunger": 10.0}}


def test_multiple_attributes_and_entities() -> None:
    world = rwt.define_need(
        {}, entity="baby", attribute="hunger", value=40.0,
        rate_per_minute=2.5, thresholds=_THRESHOLDS, minutes_per_tick=1.0,
    )
    world = rwt.define_need(
        world, entity="mother", attribute="energy", value=80.0,
        rate_per_minute=-1.0,
        thresholds=[{"at": 20.0, "need": "rest", "comparison": "<="}],
        minutes_per_tick=1.0,
    )
    assert len(world["specs"]) == 2
    assert rwt.snapshot(world) == {"baby": {"hunger": 40.0}, "mother": {"energy": 80.0}}


def test_list_needs_preserves_rate_per_minute() -> None:
    world = rwt.define_need(
        {}, entity="baby", attribute="hunger", value=40.0,
        rate_per_minute=2.5, thresholds=_THRESHOLDS, minutes_per_tick=2.0,
    )
    needs = rwt.list_needs(world)
    assert needs[0]["rate_per_minute"] == 2.5   # authored value, not per-tick
    assert needs[0]["entity"] == "baby"


def test_remove_entity() -> None:
    world = rwt.define_need(
        {}, entity="baby", attribute="hunger", value=40.0,
        rate_per_minute=2.5, thresholds=_THRESHOLDS, minutes_per_tick=1.0,
    )
    world = rwt.remove_entity(world, "baby", minutes_per_tick=1.0)
    assert world["specs"] == []
    assert rwt.snapshot(world) == {}


def test_malformed_threshold_raises() -> None:
    with pytest.raises((ValidationError, ValueError)):
        rwt.define_need(
            {}, entity="baby", attribute="hunger", value=40.0,
            rate_per_minute=2.5,
            thresholds=[{"at": 70.0}],   # missing required 'need'
            minutes_per_tick=1.0,
        )


def test_snapshot_and_list_on_empty_world() -> None:
    assert rwt.snapshot({}) == {}
    assert rwt.list_needs({}) == []
    assert rwt.snapshot({"garbage": 1}) == {}


# ─── advance / resolve (the sim-loop step) ────────────────────────────


def _baby_world(value: float = 65.0) -> dict:
    return rwt.define_need(
        {},
        entity="baby",
        attribute="hunger",
        value=value,
        rate_per_minute=5.0,
        thresholds=_THRESHOLDS,
        minutes_per_tick=1.0,
    )


def test_advance_drifts_and_raises_demand() -> None:
    world, demands = rwt.advance(_baby_world(65.0))   # 65 -> 70
    assert rwt.snapshot(world) == {"baby": {"hunger": 70.0}}
    assert any(d.need == "feed_baby" for d in demands)


def test_advance_no_demand_below_threshold() -> None:
    world, demands = rwt.advance(_baby_world(40.0))   # 40 -> 45
    assert demands == []
    assert rwt.snapshot(world) == {"baby": {"hunger": 45.0}}


def test_advance_escalates_when_ignored() -> None:
    world, demands = rwt.advance(_baby_world(96.0))   # 96 -> 100 (clamped)
    assert {d.need for d in demands} == {"feed_baby", "baby_sick"}


def test_resolve_lowers_value() -> None:
    world = _baby_world(70.0)
    world = rwt.resolve(world, "baby", "hunger", 0.9)   # -0.9*100 → clamp 0
    assert rwt.snapshot(world) == {"baby": {"hunger": 0.0}}


def test_advance_on_empty_world_is_safe() -> None:
    world, demands = rwt.advance({})
    assert demands == []
    assert rwt.snapshot(world) == {}


# ─── caregiver + shared awareness (multi-chat) ────────────────────────


_PRESENT = [
    {"id": "c_shani", "name": "Shani", "age_stage": "adult"},
    {"id": "c_mike", "name": "Mike", "age_stage": "adult"},
    {"id": "c_baby", "name": "baby", "age_stage": "baby"},
]


def _hungry_world(caregiver: str = "Shani") -> dict:
    # value already above threshold; rate 0 so advance() doesn't change it
    return rwt.define_need(
        {},
        entity="baby",
        attribute="hunger",
        value=85.0,
        rate_per_minute=0.0,
        thresholds=[{"at": 70.0, "need": "feed_baby", "urgency": 1}],
        caregiver=caregiver,
        minutes_per_tick=1.0,
    )


def test_define_need_stores_caregiver() -> None:
    world = _hungry_world("Shani")
    assert rwt.caregiver_for(world, "baby", "hunger") == "Shani"
    assert rwt.list_needs(world)[0]["caregiver"] == "Shani"


def test_caregiver_for_unknown_returns_empty() -> None:
    assert rwt.caregiver_for({}, "baby", "hunger") == ""


def test_build_awareness_targets_caregiver() -> None:
    world = _hungry_world("Shani")
    _, demands = rwt.advance(world)
    awareness, obligations = rwt.build_awareness(world, demands, _PRESENT)
    # Shared awareness mentions the baby + a human phrase, shown to all
    assert "baby" in awareness and "hungrig" in awareness
    # Only Shani (the caregiver) gets the strong obligation
    assert set(obligations.keys()) == {"c_shani"}
    assert "HANDLE" in obligations["c_shani"]


def test_build_awareness_fallback_shares_among_adults() -> None:
    world = _hungry_world(caregiver="")   # no designated caregiver
    _, demands = rwt.advance(world)
    _, obligations = rwt.build_awareness(world, demands, _PRESENT)
    # Both adults share the soft duty; the baby itself does not
    assert set(obligations.keys()) == {"c_shani", "c_mike"}


def test_build_awareness_empty_when_no_demands() -> None:
    awareness, obligations = rwt.build_awareness({}, [], _PRESENT)
    assert awareness == ""
    assert obligations == {}
