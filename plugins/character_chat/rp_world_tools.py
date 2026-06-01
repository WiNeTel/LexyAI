"""
Lexy AI - character_chat: RP world-state authoring helpers.

Pure functions that bridge the coordination kernel's scene-spec layer
(``lexy_core.coordination``) to the per-session ``world.json`` blob managed
by :class:`RPSessionContainer`. Kept out of the (large) plugin file so the
authoring logic is unit-testable on its own.

Persisted blob shape::

    {"specs": [ {NeedSpec...}, ... ], "state": { entity: {Entity...}, ... }}

* ``specs`` is the authored definition (survives edits, drift in real
  minutes — cadence-independent).
* ``state`` is the materialised :class:`WorldState` (current values),
  rebuilt whenever a need is (re)defined and advanced by the sim-loop later.

The two ``*_SCHEMA`` dicts are the JSON schemas for the LLM tools; they live
here next to the logic so the plugin only needs a thin import.
"""

from __future__ import annotations

from typing import Any

from lexy_core.coordination import (
    Demand,
    NeedSpec,
    Threshold,
    WorldState,
    build_world_state,
    specs_from_list,
    specs_to_list,
)
from lexy_core.utils.logging import get_logger

log = get_logger(module="character_chat.rp_world_tools")

# One world per session folder → a single fixed internal scope label.
_SCOPE = "scene"


def _load_specs(world: dict[str, Any]) -> list[NeedSpec]:
    raw = world.get("specs") if isinstance(world, dict) else None
    if not isinstance(raw, list):
        return []
    return specs_from_list(raw)


def _materialise(specs: list[NeedSpec], minutes_per_tick: float) -> dict[str, Any]:
    """Rebuild the persisted blob from authored specs."""
    ws = build_world_state(_SCOPE, specs, minutes_per_tick)
    return {"specs": specs_to_list(specs), "state": ws.to_dict(_SCOPE)}


def define_need(
    world: dict[str, Any],
    *,
    entity: str,
    attribute: str,
    rate_per_minute: float,
    thresholds: list[dict[str, Any]],
    value: float = 0.0,
    minimum: float = 0.0,
    maximum: float = 100.0,
    minutes_per_tick: float,
) -> dict[str, Any]:
    """Add or replace one need on ``entity`` and return the updated blob.

    A need with the same ``(entity, attribute)`` is replaced (re-authoring).
    Raises ``pydantic.ValidationError`` / ``ValueError`` on malformed
    thresholds — the tool handler turns that into an error result.
    """
    specs = [
        s
        for s in _load_specs(world)
        if not (s.entity == entity and s.attribute == attribute)
    ]
    parsed_thresholds = [Threshold.model_validate(t) for t in (thresholds or [])]
    specs.append(
        NeedSpec(
            entity=entity,
            attribute=attribute,
            value=value,
            minimum=minimum,
            maximum=maximum,
            rate_per_minute=rate_per_minute,
            thresholds=parsed_thresholds,
        )
    )
    log.info(
        "rp_world.need_defined", entity=entity, attribute=attribute, needs=len(specs)
    )
    return _materialise(specs, minutes_per_tick)


def remove_entity(
    world: dict[str, Any], entity: str, minutes_per_tick: float
) -> dict[str, Any]:
    """Drop every need belonging to ``entity`` and return the updated blob."""
    specs = [s for s in _load_specs(world) if s.entity != entity]
    return _materialise(specs, minutes_per_tick)


def snapshot(world: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Return ``{entity: {attribute: value}}`` from the persisted state."""
    state = world.get("state") if isinstance(world, dict) else None
    if not isinstance(state, dict) or not state:
        return {}
    ws = WorldState()
    ws.from_dict(_SCOPE, state)
    return ws.snapshot(_SCOPE)


def list_needs(world: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the authored need specs as plain dicts."""
    return specs_to_list(_load_specs(world))


# ─── Sim-loop step (advance + resolve) ───────────────────────────────


def _state_world(world: dict[str, Any]) -> WorldState:
    ws = WorldState()
    state = world.get("state") if isinstance(world, dict) else None
    if isinstance(state, dict) and state:
        ws.from_dict(_SCOPE, state)
    return ws


def _repack(world: dict[str, Any], ws: WorldState) -> dict[str, Any]:
    specs = world.get("specs") if isinstance(world, dict) else None
    return {
        "specs": specs if isinstance(specs, list) else [],
        "state": ws.to_dict(_SCOPE),
    }


def advance(world: dict[str, Any]) -> tuple[dict[str, Any], list[Demand]]:
    """Advance the world one tick and return (updated_world, open_demands).

    Open demands = **all currently-crossed thresholds** (via
    ``WorldState.evaluate``), not just freshly crossed ones — so an unmet
    need keeps re-driving a character (and a higher threshold escalates)
    each tick. This is the stateless equivalent of the kernel
    ``CoordinationLoop``'s open-set, suited to the scheduler-driven sim
    tick (no in-memory state between fires).
    """
    ws = _state_world(world)
    ws.tick(_SCOPE)                      # drift values one tick
    demands = ws.evaluate(_SCOPE)        # every threshold currently crossed
    return _repack(world, ws), demands


def resolve(
    world: dict[str, Any], entity: str, attribute: str, magnitude: float
) -> dict[str, Any]:
    """Apply a satisfied verdict: move the attribute toward safety.

    ``magnitude`` is the referee's 0..1 ruling; :meth:`WorldState.relieve`
    turns it into a signed delta in the right direction.
    """
    ws = _state_world(world)
    ws.relieve(_SCOPE, entity, attribute, magnitude)
    return _repack(world, ws)


# ─── LLM tool schemas ────────────────────────────────────────────────

DEFINE_NEED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "session_id": {"type": "string", "description": "RP-Session-ID"},
        "entity": {
            "type": "string",
            "description": "Wessen Beduerfnis, z.B. 'baby' oder ein Charaktername",
        },
        "attribute": {
            "type": "string",
            "description": "Name des Beduerfnisses, z.B. 'hunger', 'energy'",
        },
        "value": {"type": "number", "description": "Startwert (0-100)", "default": 0},
        "minimum": {"type": "number", "default": 0},
        "maximum": {"type": "number", "default": 100},
        "rate_per_minute": {
            "type": "number",
            "description": (
                "Drift pro Realminute. Positiv = steigt (Hunger), "
                "negativ = faellt (Energie)."
            ),
        },
        "thresholds": {
            "type": "array",
            "description": "Schwellen, die eine Handlungs-Anforderung ausloesen",
            "items": {
                "type": "object",
                "properties": {
                    "at": {"type": "number", "description": "Schwellenwert"},
                    "need": {
                        "type": "string",
                        "description": "Label der Anforderung, z.B. 'feed_baby'",
                    },
                    "comparison": {
                        "type": "string",
                        "enum": [">=", "<="],
                        "default": ">=",
                    },
                    "urgency": {"type": "integer", "default": 1},
                },
                "required": ["at", "need"],
            },
        },
    },
    "required": ["session_id", "entity", "attribute", "rate_per_minute", "thresholds"],
}

WORLD_SNAPSHOT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "session_id": {"type": "string", "description": "RP-Session-ID"},
    },
    "required": ["session_id"],
}
