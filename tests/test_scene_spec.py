"""Tests for the coordination scene-spec authoring layer."""

from __future__ import annotations

from lexy_core.coordination import (
    NeedSpec,
    Threshold,
    WorldState,
    build_world_state,
    specs_from_list,
    specs_to_list,
)


def _baby_spec() -> NeedSpec:
    return NeedSpec(
        entity="baby",
        attribute="hunger",
        value=40.0,
        rate_per_minute=2.5,
        thresholds=[
            Threshold(at=70.0, need="feed_baby", urgency=1),
            Threshold(at=100.0, need="baby_sick", urgency=3),
        ],
    )


def test_build_converts_rate_per_minute_to_per_tick() -> None:
    # 2.5/min at 2 min/tick → 5.0 per tick
    ws = build_world_state("scene", [_baby_spec()], minutes_per_tick=2.0)
    assert ws.get("scene", "baby", "hunger") == 40.0
    ws.tick("scene")   # 40 -> 45
    assert ws.get("scene", "baby", "hunger") == 45.0


def test_build_carries_thresholds_into_demands() -> None:
    ws = build_world_state("scene", [_baby_spec()], minutes_per_tick=2.0)
    # 40 -> 45 -> 50 -> 55 -> 60 -> 65 -> 70 (crosses feed_baby on the 6th tick)
    raised: list[str] = []
    for _ in range(6):
        raised.extend(d.need for d in ws.tick("scene"))
    assert "feed_baby" in raised


def test_minutes_per_tick_zero_is_static() -> None:
    ws = build_world_state("scene", [_baby_spec()], minutes_per_tick=0.0)
    ws.tick("scene")
    assert ws.get("scene", "baby", "hunger") == 40.0   # no drift


def test_multiple_entities_and_attributes() -> None:
    specs = [
        _baby_spec(),
        NeedSpec(entity="mother", attribute="energy", value=80.0, rate_per_minute=-1.0),
    ]
    ws = build_world_state("scene", specs, minutes_per_tick=1.0)
    snap = ws.snapshot("scene")
    assert snap == {"baby": {"hunger": 40.0}, "mother": {"energy": 80.0}}


def test_build_into_existing_world() -> None:
    ws = WorldState()
    build_world_state("a", [_baby_spec()], 1.0, world=ws)
    build_world_state("b", [_baby_spec()], 1.0, world=ws)
    assert ws.get("a", "baby", "hunger") == 40.0
    assert ws.get("b", "baby", "hunger") == 40.0


def test_specs_roundtrip() -> None:
    specs = [_baby_spec()]
    restored = specs_from_list(specs_to_list(specs))
    assert len(restored) == 1
    assert restored[0].entity == "baby"
    assert restored[0].rate_per_minute == 2.5
    assert restored[0].thresholds[1].need == "baby_sick"


def test_built_state_persists_via_world_to_dict() -> None:
    ws = build_world_state("scene", [_baby_spec()], minutes_per_tick=2.0)
    dumped = ws.to_dict("scene")

    restored = WorldState()
    restored.from_dict("scene", dumped)
    # rate (converted) survived → drift continues identically
    restored.tick("scene")   # 40 -> 45
    assert restored.get("scene", "baby", "hunger") == 45.0
