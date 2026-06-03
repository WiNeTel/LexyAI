"""
Lexy AI — Task pattern detector (Phase P4).

Decides, from the tools a turn used, whether the work is *skill-worthy* — the
trigger for autonomous skill creation. Two signals:

* **complex** — a single turn used at least ``complex_threshold`` tools
  (one big multi-step task worth capturing).
* **repeated** — the same ordered tool sequence has now been seen
  ``repeat_threshold`` times across turns (a recurring routine).

Turns that touch RP / character tools are ignored outright, so the global
skill learner never learns from roleplay sessions. The detector is pure
(in-memory counters, no DB, no LLM) so it's trivially unit-testable; the
plugin owns the actual drafting once a signal fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# Substrings that mark a tool as RP/character-related — never learn from these.
_RP_TOOL_MARKERS: tuple[str, ...] = (
    "character",
    "run_round",
    "run_character",
    "rp_",
    "narrate",
    "scene",
)


def tool_name(entry: Any) -> str:
    """Extract a tool name from a ``tools_used`` entry (str or dict)."""
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        for key in ("tool", "name", "tool_name"):
            value = entry.get(key)
            if value:
                return str(value).strip()
    return ""


def signature_of(names: list[str]) -> str:
    """Stable signature for an ordered tool sequence."""
    return ">".join(names)


def is_rp_tool(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _RP_TOOL_MARKERS)


@dataclass
class TaskSignal:
    """A skill-worthy pattern the detector surfaced."""

    signature: str
    reason: str  # "complex" | "repeated"
    tools: list[str]


@dataclass
class TaskDetector:
    """Stateful (in-memory) detector of skill-worthy tool patterns."""

    complex_threshold: int = 8
    repeat_threshold: int = 3
    _counts: dict[str, int] = field(default_factory=dict)
    _signalled: set[str] = field(default_factory=set)

    def record(self, tools_used: Iterable[Any]) -> TaskSignal | None:
        """Feed one turn's tools; return a signal if it's skill-worthy.

        Idempotent per signature: once a signature has fired a signal it
        won't fire again (so the same routine isn't proposed repeatedly).
        Returns ``None`` for RP turns, empty turns, and not-yet-worthy ones.
        """
        names = [n for n in (tool_name(t) for t in tools_used) if n]
        if not names:
            return None
        if any(is_rp_tool(n) for n in names):
            return None  # never learn from roleplay turns

        signature = signature_of(names)
        if signature in self._signalled:
            return None

        # A single big multi-tool turn is worth capturing on its own.
        if len(names) >= self.complex_threshold:
            self._signalled.add(signature)
            return TaskSignal(signature, "complex", names)

        # Otherwise count repetitions across turns.
        count = self._counts.get(signature, 0) + 1
        self._counts[signature] = count
        if count >= self.repeat_threshold:
            self._signalled.add(signature)
            return TaskSignal(signature, "repeated", names)
        return None
