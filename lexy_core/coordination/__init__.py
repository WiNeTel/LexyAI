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

Later phases add ``Referee`` (adjudicates whether a narrated action
satisfied a demand) and ``CoordinationLoop`` (ties them to the scheduler
tick).
"""

from __future__ import annotations

from lexy_core.coordination.blackboard import POST_KINDS, Blackboard, Post
from lexy_core.coordination.convergence import (
    ConvergenceDetector,
    ConvergenceResult,
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
]
