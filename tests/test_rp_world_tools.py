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
