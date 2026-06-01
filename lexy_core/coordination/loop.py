"""
Lexy AI - Coordination: CoordinationLoop.

Where the AutoScientists heartbeat — *read the state, act, write back,
verify the result* — finally closes for Lexy. One :meth:`tick` ties the
three primitives together:

1. **read** — :class:`WorldState.tick` advances the numbers and raises new
   :class:`Demand` obligations (the baby's hunger crossed a threshold).
2. **act** — a ``narrate`` callable drives the responsible character via a
   dynamic situational prompt; the character replies in free prose.
3. **verify** — the :class:`Referee` rules whether that prose actually
   satisfied the demand.
4. **consequence** — satisfied → :meth:`WorldState.relieve` lowers the
   number and the demand closes; not satisfied → the demand stays open and
   the rising attribute escalates on the next tick.

Everything external (how a character narrates, how the LLM is called) is
injected, so the loop is deterministic and unit-testable with stubs. The
real RP wiring (group_turn / pulse_generator / scheduler tick) is a later
phase that supplies those callables.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from lexy_core.coordination.blackboard import Blackboard
from lexy_core.coordination.convergence import LLMChat
from lexy_core.coordination.referee import Referee
from lexy_core.coordination.world_state import Demand, WorldState
from lexy_core.utils.logging import get_logger

log = get_logger(module="coordination.loop")

# Async ``(demand) -> narration`` — produces the character's in-voice reply
# to the situation the demand describes.
Narrator = Callable[[Demand], Awaitable[str]]


class LoopConfig(BaseModel):
    """Tunables for the coordination loop."""

    # Fraction of an attribute's span relieved at referee magnitude 1.0.
    relief_fraction_full: float = 1.0
    referee_brain: str = "e4b"


class TickReport(BaseModel):
    """What happened during one :meth:`CoordinationLoop.tick`."""

    raised: list[Demand] = Field(default_factory=list)
    satisfied: list[str] = Field(default_factory=list)    # "entity:need"
    still_open: list[str] = Field(default_factory=list)    # "entity:need"


def _key(demand: Demand) -> str:
    return f"{demand.entity}:{demand.need}"


class CoordinationLoop:
    """Drives the read→act→verify→consequence cycle for a scope."""

    def __init__(
        self,
        world_state: WorldState,
        referee: Referee,
        llm_chat: LLMChat,
        blackboard: Blackboard | None = None,
        config: LoopConfig | None = None,
    ) -> None:
        self._world = world_state
        self._referee = referee
        self._llm = llm_chat
        self._bb = blackboard
        self._cfg = config or LoopConfig()
        # scope -> {"entity:need": Demand} still awaiting satisfaction
        self._open: dict[str, dict[str, Demand]] = {}

    async def tick(
        self,
        scope: str,
        narrate: Narrator,
        ticks: int = 1,
    ) -> TickReport:
        """Advance ``scope`` one step and drive every open demand to a verdict.

        Args:
            scope: The arena (RP scene) id.
            narrate: Async ``(demand) -> narration`` producing the responsible
                character's in-voice reply to the situation.
            ticks: How many world-ticks to advance before driving demands.
        """
        open_set = self._open.setdefault(scope, {})

        # 1. read — advance the world and register newly raised demands.
        raised = self._world.tick(scope, ticks)
        for demand in raised:
            open_set[_key(demand)] = demand
            await self._post(scope, "world", "demand", f"{demand.need} ({demand.entity})", demand)

        satisfied: list[str] = []
        still_open: list[str] = []

        # 2-4. for every open demand: act → verify → consequence.
        for key, demand in list(open_set.items()):
            narration = await narrate(demand)
            verdict = await self._referee.adjudicate(
                demand, narration, self._llm, brain=self._cfg.referee_brain
            )
            if verdict.satisfied:
                new_value = self._world.relieve(
                    scope,
                    demand.entity,
                    demand.attribute,
                    verdict.magnitude * self._cfg.relief_fraction_full,
                )
                del open_set[key]
                satisfied.append(key)
                log.info(
                    "loop.demand_satisfied",
                    scope=scope, need=demand.need, entity=demand.entity,
                    new_value=round(new_value, 2),
                )
                await self._post(
                    scope, "referee", "decision",
                    f"{demand.need} erfuellt: {verdict.rationale}", demand,
                )
            else:
                still_open.append(key)
                log.info(
                    "loop.demand_unmet",
                    scope=scope, need=demand.need, entity=demand.entity,
                )
                await self._post(
                    scope, "referee", "decision",
                    f"{demand.need} NICHT erfuellt: {verdict.rationale}", demand,
                )

        return TickReport(raised=raised, satisfied=satisfied, still_open=still_open)

    def open_demands(self, scope: str) -> list[Demand]:
        """Return the demands still awaiting satisfaction in ``scope``."""
        return list(self._open.get(scope, {}).values())

    async def _post(
        self, scope: str, author: str, kind: str, body: str, demand: Demand
    ) -> None:
        if self._bb is None:
            return
        await self._bb.post(scope, author, kind, body, demand.model_dump())
