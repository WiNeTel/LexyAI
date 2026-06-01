"""
Lexy AI - Coordination: Scene specification.

The authoring layer for a simulated scene. Mike defines a scene's needs
**explicitly per scene** (his choice): which entity has which attribute,
how fast it drifts, and at which thresholds an obligation appears. A scene
with no spec simply has no simulation — so the feature is opt-in by
construction.

Drift is authored in **real-world minutes** (``rate_per_minute``), not raw
ticks, so the spec is independent of the sim-clock cadence. The cadence
(``minutes_per_tick``) is a config knob on the loop; :func:`build_world_state`
converts minutes→ticks at build time. That realises the "decoupled,
config-tunable sim-clock" decision: change the cadence and every authored
rate scales with it.

This is pure, synchronous data → :class:`WorldState`; persistence of the
resulting state uses the kernel's existing ``WorldState.to_dict`` /
``from_dict``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from lexy_core.coordination.world_state import Attribute, Threshold, WorldState
from lexy_core.utils.logging import get_logger

log = get_logger(module="coordination.scene_spec")


class NeedSpec(BaseModel):
    """Declarative definition of one evolving need on one entity.

    Example — a baby's hunger that rises 2.5/min and demands feeding at 70,
    escalating to sickness at 100::

        NeedSpec(
            entity="baby", attribute="hunger", value=40.0, rate_per_minute=2.5,
            thresholds=[
                Threshold(at=70.0, need="feed_baby", urgency=1),
                Threshold(at=100.0, need="baby_sick", urgency=3),
            ],
        )
    """

    entity: str
    attribute: str
    value: float = 0.0
    minimum: float = 0.0
    maximum: float = 100.0
    rate_per_minute: float = 0.0       # signed: +rises (hunger), -falls (energy)
    thresholds: list[Threshold] = Field(default_factory=list)


def build_world_state(
    scope: str,
    specs: list[NeedSpec],
    minutes_per_tick: float,
    world: WorldState | None = None,
) -> WorldState:
    """Materialise authored ``specs`` into a :class:`WorldState` for ``scope``.

    ``rate_per_minute`` is converted to the kernel's per-tick rate using
    ``minutes_per_tick`` (the loop's cadence). Pass an existing ``world`` to
    add to it, or get a fresh one back.
    """
    ws = world or WorldState()
    for spec in specs:
        ws.add_attribute(
            scope,
            spec.entity,
            Attribute(
                name=spec.attribute,
                value=spec.value,
                minimum=spec.minimum,
                maximum=spec.maximum,
                rate_per_tick=spec.rate_per_minute * minutes_per_tick,
                thresholds=list(spec.thresholds),
            ),
        )
    log.info(
        "scene_spec.built",
        scope=scope,
        needs=len(specs),
        minutes_per_tick=minutes_per_tick,
    )
    return ws


def specs_to_list(specs: list[NeedSpec]) -> list[dict[str, object]]:
    """Serialise authored specs (e.g. to persist the scene definition)."""
    return [spec.model_dump() for spec in specs]


def specs_from_list(data: list[dict[str, object]]) -> list[NeedSpec]:
    """Restore authored specs from :func:`specs_to_list` output."""
    return [NeedSpec.model_validate(item) for item in data]
