"""
Lexy AI - Coordination kernel.

The shared substrate behind every multi-agent feature in Lexy. It exists
because each previous multi-agent attempt (the baby scheduler, MCP Town,
multi-character RP) failed the same way: characters *narrated* instead of
*simulating* — there was no shared world-state with consequence, no
structured intent, and no closed loop verifying that an action changed the
state. See ``docs/architecture/agent-coordination.md`` for the full design.

This package provides the reusable primitives, decoupled from any plugin:

* :class:`Blackboard` — the shared "schwarzes Brett": append-only posts plus
  a per-scope key-value fact store, persisted via aiosqlite. The substrate
  for BOTH the simulation loop and the deliberation loop.
* :class:`ConvergenceDetector` — generalised from ``expert_panel``: extracts
  agreement points across contributions so a discussion knows when to stop.
* :class:`WorldState` — numeric, time-evolving entity attributes with
  thresholds that raise :class:`Demand` obligations. The "Zustand mit
  Konsequenz" that every previous simulation attempt lacked.

* :class:`Referee` — the game master: reads a character's narration against
  an open demand and rules whether it was actually satisfied (concrete
  action vs. mere commenting). The loop applies the verdict to the world.

* :class:`CoordinationLoop` — one :meth:`~CoordinationLoop.tick` ties
  world-state + referee + blackboard into the read→act→verify→consequence
  cycle. The RP plugin drives it from the scheduler tick.
"""

from __future__ import annotations

from lexy_core.coordination.blackboard import POST_KINDS, Blackboard, Post
from lexy_core.coordination.convergence import (
    ConvergenceDetector,
    ConvergenceResult,
)
from lexy_core.coordination.fact_extractor import FactExtractor
from lexy_core.coordination.loop import (
    CoordinationLoop,
    LoopConfig,
    Narrator,
    TickReport,
)
from lexy_core.coordination.referee import Referee, Verdict
from lexy_core.coordination.scene_director import (
    SceneDirector,
    looks_like_has_dependent,
)
from lexy_core.coordination.scene_spec import (
    NeedSpec,
    build_world_state,
    specs_from_list,
    specs_to_list,
)
from lexy_core.coordination.world_state import (
    Attribute,
    Demand,
    Entity,
    Threshold,
    WorldState,
)

__all__ = [
    "Blackboard",
    "Post",
    "POST_KINDS",
    "ConvergenceDetector",
    "ConvergenceResult",
    "WorldState",
    "Attribute",
    "Threshold",
    "Entity",
    "Demand",
    "Referee",
    "Verdict",
    "FactExtractor",
    "SceneDirector",
    "looks_like_has_dependent",
    "CoordinationLoop",
    "LoopConfig",
    "TickReport",
    "Narrator",
    "NeedSpec",
    "build_world_state",
    "specs_to_list",
    "specs_from_list",
]
