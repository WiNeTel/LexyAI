"""Tests for the coordination WorldState (numeric simulation + demands)."""

from __future__ import annotations

from lexy_core.coordination import Attribute, Threshold, WorldState


def _baby_world() -> WorldState:
    """A scene with a baby whose hunger rises and escalates if ignored."""
    ws = WorldState()
    ws.add_attribute(
        "scene",
        "baby",
        Attribute(
            name="hunger",
            value=60.0,
            rate_per_tick=5.0,
            thresholds=[
                Threshold(at=70.0, need="feed_baby", urgency=1),
                Threshold(at=100.0, need="baby_sick", urgency=3),
            ],
        ),
    )
    return ws


def test_add_attribute_clamps_initial_value() -> None:
    ws = WorldState()
    ws.add_attribute("s", "e", Attribute(name="x", value=999.0, maximum=100.0))
    assert ws.get("s", "e", "x") == 100.0


def test_set_and_apply_clamp() -> None:
    ws = _baby_world()
    assert ws.set("scene", "baby", "hunger", 50.0) == 50.0
    assert ws.apply("scene", "baby", "hunger", -100.0) == 0.0   # clamped at min
    assert ws.apply("scene", "baby", "hunger", 250.0) == 100.0  # clamped at max


def test_get_unknown_returns_zero() -> None:
    ws = WorldState()
    assert ws.get("nope", "nobody", "nothing") == 0.0


def test_tick_advances_value() -> None:
    ws = _baby_world()
    ws.tick("scene")  # 60 -> 65
    assert ws.get("scene", "baby", "hunger") == 65.0


def test_tick_raises_demand_on_newly_crossed_threshold() -> None:
    ws = _baby_world()              # hunger 60, threshold at 70
    assert ws.tick("scene") == []  # 60 -> 65, not crossed yet
    demands = ws.tick("scene")     # 65 -> 70, crosses feed_baby
    assert len(demands) == 1
    d = demands[0]
    assert d.need == "feed_baby"
    assert d.entity == "baby"
    assert d.value == 70.0


def test_no_respam_while_threshold_stays_crossed() -> None:
    ws = _baby_world()
    ws.tick("scene")               # 65
    ws.tick("scene")               # 70 -> feed_baby demand
    more = ws.tick("scene")        # 75, still above 70 but NOT newly crossed
    assert more == []              # no re-spam


def test_ignored_demand_escalates_to_higher_threshold() -> None:
    ws = _baby_world()
    raised: list[str] = []
    for _ in range(9):             # 60 -> 105 (clamped 100) over 9 ticks
        for d in ws.tick("scene"):
            raised.append(d.need)
    # feed_baby at 70 AND baby_sick at 100 both fire (escalation) because
    # nobody called apply() to lower hunger.
    assert "feed_baby" in raised
    assert "baby_sick" in raised


def test_satisfied_demand_prevents_escalation() -> None:
    ws = _baby_world()
    ws.tick("scene")               # 65
    ws.tick("scene")               # 70 -> feed_baby
    # Referee satisfies it: hunger drops well below threshold.
    ws.apply("scene", "baby", "hunger", -50.0)   # 70 -> 20
    raised: list[str] = []
    for _ in range(3):
        raised.extend(d.need for d in ws.tick("scene"))   # 20 -> 35
    assert raised == []            # no escalation, stayed handled


def test_falling_attribute_threshold() -> None:
    ws = WorldState()
    ws.add_attribute(
        "s",
        "hero",
        Attribute(
            name="energy",
            value=30.0,
            rate_per_tick=-5.0,
            thresholds=[Threshold(at=20.0, need="rest", comparison="<=")],
        ),
    )
    assert ws.tick("s") == []      # 30 -> 25
    demands = ws.tick("s")         # 25 -> 20, crosses (<=20)
    assert len(demands) == 1 and demands[0].need == "rest"


def test_evaluate_reports_all_currently_crossed_without_advancing() -> None:
    ws = _baby_world()
    ws.set("scene", "baby", "hunger", 105.0)   # clamped 100, both thresholds crossed
    demands = ws.evaluate("scene")
    needs = {d.need for d in demands}
    assert needs == {"feed_baby", "baby_sick"}
    # evaluate must not change the value
    assert ws.get("scene", "baby", "hunger") == 100.0


def test_static_attribute_does_not_drift() -> None:
    ws = WorldState()
    ws.add_attribute("s", "rock", Attribute(name="mass", value=50.0, rate_per_tick=0.0))
    ws.tick("s")
    assert ws.get("s", "rock", "mass") == 50.0


def test_snapshot_shape() -> None:
    ws = _baby_world()
    snap = ws.snapshot("scene")
    assert snap == {"baby": {"hunger": 60.0}}


def test_to_dict_from_dict_roundtrip() -> None:
    ws = _baby_world()
    ws.tick("scene")               # 65
    dumped = ws.to_dict("scene")

    restored = WorldState()
    restored.from_dict("scene", dumped)
    assert restored.get("scene", "baby", "hunger") == 65.0
    # thresholds survived → ticking still raises the demand
    demands = restored.tick("scene")   # 65 -> 70
    assert demands[0].need == "feed_baby"
