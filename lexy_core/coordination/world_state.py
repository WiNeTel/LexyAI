"""
Lexy AI - Coordination: WorldState.

The piece that was missing everywhere multi-agent simulation failed: a
shared world-state that **evolves on its own and pushes back when ignored**.

A character's "the baby cries" was pure narration — there was no number
that rose over time and no rule that made things worse if nobody acted.
``WorldState`` adds exactly that:

* **Entities** (characters, objects) own **numeric Attributes** (hunger,
  energy, mood…) that change every tick via ``rate_per_tick``.
* Each Attribute carries **Thresholds**. When a tick pushes an attribute
  across a threshold, a **Demand** is produced — an obligation ("feed the
  baby"), not just a notification.
* The :class:`Referee` later decides whether a narrated action satisfied a
  demand; if so it calls :meth:`WorldState.apply` to lower the number, if
  not the demand stays open and the next higher threshold escalates.

This module is pure, synchronous, LLM-free logic so it is fully
deterministic and unit-testable. Persistence (per RP session) is wired in
a later phase via ``to_dict`` / ``from_dict``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from lexy_core.utils.logging import get_logger

log = get_logger(module="coordination.world_state")

_RISING = ">="
_FALLING = "<="


class Threshold(BaseModel):
    """A line that, once crossed, raises a :class:`Demand`.

    ``comparison`` is ``">="`` for attributes that *rise* into trouble
    (hunger) or ``"<="`` for attributes that *fall* into trouble (energy).
    """

    at: float
    need: str                       # demand label, e.g. "feed_baby"
    comparison: str = _RISING        # ">=" | "<="
    urgency: int = 1                 # higher thresholds = more urgent

    def is_crossed(self, value: float) -> bool:
        """True if ``value`` currently sits on the wrong side of the line."""
        if self.comparison == _FALLING:
            return value <= self.at
        return value >= self.at


class Attribute(BaseModel):
    """A single numeric, time-evolving quantity of an entity."""

    name: str
    value: float = 0.0
    minimum: float = 0.0
    maximum: float = 100.0
    rate_per_tick: float = 0.0       # signed: +rises (hunger), -falls (energy)
    thresholds: list[Threshold] = Field(default_factory=list)

    def clamped(self, value: float) -> float:
        return max(self.minimum, min(self.maximum, value))


class Entity(BaseModel):
    """A character or object that owns attributes."""

    id: str
    attributes: dict[str, Attribute] = Field(default_factory=dict)


class Demand(BaseModel):
    """An obligation raised when an attribute crosses a threshold."""

    scope: str
    entity: str
    attribute: str
    need: str
    urgency: int
    value: float                     # attribute value at the moment of crossing


class WorldState:
    """In-memory, deterministic world-state grouped by ``scope``.

    A ``scope`` is one arena (an RP scene). The owning plugin persists the
    serialised form (:meth:`to_dict`) via its session store; this class
    itself holds no I/O.
    """

    def __init__(self) -> None:
        self._scopes: dict[str, dict[str, Entity]] = {}

    # ─── Construction ────────────────────────────────────────────────

    def add_entity(self, scope: str, entity_id: str) -> Entity:
        """Create (or return existing) entity in ``scope``."""
        entities = self._scopes.setdefault(scope, {})
        if entity_id not in entities:
            entities[entity_id] = Entity(id=entity_id)
        return entities[entity_id]

    def add_attribute(self, scope: str, entity_id: str, attribute: Attribute) -> None:
        """Attach an attribute to an entity (creating the entity if needed)."""
        entity = self.add_entity(scope, entity_id)
        attribute.value = attribute.clamped(attribute.value)
        entity.attributes[attribute.name] = attribute

    # ─── Read / mutate ───────────────────────────────────────────────

    def get(self, scope: str, entity_id: str, attr: str) -> float:
        """Return an attribute value, or ``0.0`` if unknown."""
        attribute = self._attr(scope, entity_id, attr)
        return attribute.value if attribute else 0.0

    def set(self, scope: str, entity_id: str, attr: str, value: float) -> float:
        """Set an attribute to ``value`` (clamped). Returns the new value."""
        attribute = self._attr(scope, entity_id, attr)
        if attribute is None:
            return 0.0
        attribute.value = attribute.clamped(value)
        return attribute.value

    def apply(self, scope: str, entity_id: str, attr: str, delta: float) -> float:
        """Add ``delta`` to an attribute (clamped). Returns the new value.

        This is how the :class:`Referee` closes the loop — e.g. a satisfied
        ``feed_baby`` demand calls ``apply(scope, "baby", "hunger", -40)``.
        """
        attribute = self._attr(scope, entity_id, attr)
        if attribute is None:
            return 0.0
        attribute.value = attribute.clamped(attribute.value + delta)
        return attribute.value

    def relieve(self, scope: str, entity_id: str, attr: str, fraction: float) -> float:
        """Move an attribute toward safety by ``fraction`` of its span.

        Direction is derived from the attribute's drift: a rising-into-trouble
        attribute (hunger, ``rate >= 0``) is lowered, a falling-into-trouble
        one (energy, ``rate < 0``) is raised. ``fraction`` is typically the
        referee's ``magnitude`` (0..1). Returns the new value (clamped).
        """
        attribute = self._attr(scope, entity_id, attr)
        if attribute is None:
            return 0.0
        span = attribute.maximum - attribute.minimum
        direction = -1.0 if attribute.rate_per_tick >= 0.0 else 1.0
        return self.apply(scope, entity_id, attr, direction * fraction * span)

    # ─── Simulation ──────────────────────────────────────────────────

    def tick(self, scope: str, ticks: int = 1) -> list[Demand]:
        """Advance every attribute in ``scope`` and return NEWLY crossed demands.

        Only thresholds that flip from "not crossed" to "crossed" during this
        tick produce a demand — so a sustained high value does not re-spam.
        Ongoing escalation is the loop's job (it keeps a demand open until
        the referee satisfies it; the next *higher* threshold crossing
        naturally yields a fresh, more urgent demand).
        """
        demands: list[Demand] = []
        for entity_id, entity in self._scopes.get(scope, {}).items():
            for attribute in entity.attributes.values():
                if attribute.rate_per_tick == 0.0 and ticks >= 0:
                    # No drift → no new crossings possible from a tick.
                    continue
                old = attribute.value
                new = attribute.clamped(old + attribute.rate_per_tick * ticks)
                attribute.value = new
                for threshold in attribute.thresholds:
                    if not threshold.is_crossed(old) and threshold.is_crossed(new):
                        demand = Demand(
                            scope=scope,
                            entity=entity_id,
                            attribute=attribute.name,
                            need=threshold.need,
                            urgency=threshold.urgency,
                            value=new,
                        )
                        demands.append(demand)
                        log.info(
                            "world_state.demand_raised",
                            scope=scope,
                            entity=entity_id,
                            need=threshold.need,
                            urgency=threshold.urgency,
                            value=round(new, 2),
                        )
        return demands

    def evaluate(self, scope: str) -> list[Demand]:
        """Return demands for ALL currently-crossed thresholds (no advance).

        Used on load/init to reconcile open demands without mutating values.
        """
        demands: list[Demand] = []
        for entity_id, entity in self._scopes.get(scope, {}).items():
            for attribute in entity.attributes.values():
                for threshold in attribute.thresholds:
                    if threshold.is_crossed(attribute.value):
                        demands.append(
                            Demand(
                                scope=scope,
                                entity=entity_id,
                                attribute=attribute.name,
                                need=threshold.need,
                                urgency=threshold.urgency,
                                value=attribute.value,
                            )
                        )
        return demands

    # ─── Serialisation (for per-session persistence in a later phase) ──

    def snapshot(self, scope: str) -> dict[str, dict[str, float]]:
        """Compact ``{entity: {attr: value}}`` view for prompts/UI/logs."""
        return {
            entity_id: {name: attr.value for name, attr in entity.attributes.items()}
            for entity_id, entity in self._scopes.get(scope, {}).items()
        }

    def to_dict(self, scope: str) -> dict[str, object]:
        """Full serialisable state of a scope (round-trips via from_dict)."""
        return {
            entity_id: entity.model_dump()
            for entity_id, entity in self._scopes.get(scope, {}).items()
        }

    def from_dict(self, scope: str, data: dict[str, object]) -> None:
        """Restore a scope from :meth:`to_dict` output (replaces existing)."""
        entities: dict[str, Entity] = {}
        for entity_id, raw in data.items():
            entities[entity_id] = Entity.model_validate(raw)
        self._scopes[scope] = entities

    # ─── Internal ────────────────────────────────────────────────────

    def _attr(self, scope: str, entity_id: str, attr: str) -> Attribute | None:
        entity = self._scopes.get(scope, {}).get(entity_id)
        if entity is None:
            return None
        return entity.attributes.get(attr)
