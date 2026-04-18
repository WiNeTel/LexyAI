"""
Context-window budget manager for character_chat prompts.

The orchestrator builds prompts from many unbounded sources (persona,
example-dialog, scenario, session history, previous turns in the round,
memory recall, user message). Without a budget, heavy Silly-Tavern
character cards plus a long session history plus three previous speakers
can easily overrun a 16K-token context window.

This module provides:

1. :class:`PromptSection` — a named, priority-tagged block of prompt text
   that can optionally reduce itself (e.g. "history" from 6 → 4 → 2 msgs)
   or get hard-trimmed to a token cap.
2. :class:`ContextBudget` — assembles sections under
   ``context_size - max_output - safety_margin`` tokens with progressive
   trimming from LOW priority to HIGH.
3. :func:`estimate_tokens` / :func:`trim_to_tokens` — cheap char-based
   token math (≈ 3.5 chars/token for German/mixed text) that needs no
   external tokenizer dependency.

Design goal: **fully dynamic budget.** The orchestrator reads the current
``context_size`` via a callback *per turn*, so changing ``routing.yaml``
or switching the brain is picked up without any plugin code change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable


log = logging.getLogger(__name__)


# ─── Token math ──────────────────────────────────────────────────────────────

# For Gemma-4 tokenizer on German/mixed text this is 3.3-3.8 chars/token.
# We pick 3.5 as a conservative middle ground — slightly overcounts tokens
# on English-heavy text, which is fine because we prefer to trim a little
# too eagerly than to overflow.
_CHARS_PER_TOKEN: float = 3.5


def estimate_tokens(text: str) -> int:
    """Cheap char-based estimate. 0 for empty strings, min 1 otherwise."""
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def trim_to_tokens(text: str, max_tokens: int) -> str:
    """Hard-truncate ``text`` to fit within ``max_tokens`` (char-level).

    Prefers cutting at a paragraph / sentence / word boundary in the last
    40 % of the allowed range so the trim doesn't land mid-word. Adds an
    ellipsis when actual truncation happened.
    """
    if max_tokens <= 0 or not text:
        return text
    max_chars = int(max_tokens * _CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Walk through progressively finer boundaries.
    floor = int(max_chars * 0.6)
    for sep in ("\n\n", "\n", ". ", "! ", "? ", "; ", " "):
        idx = cut.rfind(sep)
        if idx >= floor:
            return cut[: idx + len(sep)].rstrip() + "…"
    return cut.rstrip() + "…"


# ─── Section model ───────────────────────────────────────────────────────────


class Priority(IntEnum):
    """Trim priority. Lower = dropped/trimmed first."""

    LOW = 0       # e.g. example_dialog, other_characters — nice-to-have
    MEDIUM = 1    # history, memory, scenario — context, reducible
    HIGH = 2      # persona, user_message, prev_turns — important
    MUST = 3      # rules, identity, "du bist dran" — never touched


# Callable that re-renders a reducible section at a smaller size.
# Signature: reduce_fn(step) -> str.
ReduceFn = Callable[[int], str]


@dataclass
class PromptSection:
    """A named piece of prompt with priority and trim hints.

    Fields
    ------
    name : short identifier used in the trim log
    priority : Priority enum, drives the trim order
    text : current text (mutable — the budget rewrites it)
    role : "system" or "user" — reassembly target after fitting
    max_tokens : soft cap applied up front (0 = no cap)
    reduce_fn + reduce_steps : for MEDIUM sections that can "shrink"
        gradually (e.g. history 6→4→2 messages). The manager calls
        ``reduce_fn(step)`` for each step in ``reduce_steps`` until
        budget fits.
    """

    name: str
    priority: Priority
    text: str
    role: str = "system"
    max_tokens: int = 0
    reduce_fn: ReduceFn | None = None
    reduce_steps: list[int] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


# ─── Budget manager ──────────────────────────────────────────────────────────


class ContextBudget:
    """Fits a set of :class:`PromptSection`s under a token ceiling.

    Instantiate **per turn** with the brain's current ``context_size``
    value — passing it in explicitly keeps the manager stateless and
    makes it trivially test-friendly.

    Trimming pipeline
    -----------------
    * Step 0: apply each section's individual ``max_tokens`` cap
    * Step 1: drop LOW sections (biggest first) until fit
    * Step 2: call ``reduce_fn`` on MEDIUM sections with ``reduce_steps``
    * Step 3: hard-trim remaining MEDIUM text down to the overflow
    * Step 4: (last resort) trim HIGH section text — but never below
      a floor (200 tok) so persona never vanishes entirely
    * MUST sections are never touched

    Returns
    -------
    ``(fitted_sections, trim_log)`` — the log is a list of human-readable
    strings the orchestrator can forward to structured logging.
    """

    # Reserved tokens for model/tokenizer overhead beyond the raw text.
    DEFAULT_SAFETY_MARGIN: int = 256
    # Floor for HIGH sections during emergency trim — we never squash
    # e.g. persona to zero because the character would lose its voice.
    HIGH_FLOOR_TOKENS: int = 200
    # Safe minimum context size; below this we log a warning.
    MIN_CONTEXT_SIZE: int = 1024

    def __init__(
        self,
        context_size: int,
        max_output_tokens: int = 320,
        safety_margin: int = DEFAULT_SAFETY_MARGIN,
    ) -> None:
        self.context_size = max(self.MIN_CONTEXT_SIZE, int(context_size or 0))
        self.max_output = max(1, int(max_output_tokens or 0))
        self.safety_margin = max(0, int(safety_margin or 0))
        self.available = max(
            256, self.context_size - self.max_output - self.safety_margin
        )

    # ─── Public API ──────────────────────────────────────────────────

    def fit_sections(
        self, sections: list[PromptSection]
    ) -> tuple[list[PromptSection], list[str]]:
        """Trim ``sections`` in-place to fit, return (sections, trim_log)."""
        trim_log: list[str] = []

        # Step 0 — per-section caps
        for s in sections:
            if s.max_tokens > 0 and s.tokens > s.max_tokens:
                before = s.tokens
                s.text = trim_to_tokens(s.text, s.max_tokens)
                trim_log.append(
                    f"{s.name}: capped {before}→{s.tokens}tok"
                )

        if self._total(sections) <= self.available:
            return sections, trim_log

        # Step 1 — drop LOW sections (biggest first to save more ops)
        low_sections = sorted(
            (s for s in sections if s.priority == Priority.LOW and s.text),
            key=lambda s: s.tokens,
            reverse=True,
        )
        for s in low_sections:
            if self._total(sections) <= self.available:
                break
            before = s.tokens
            s.text = ""
            trim_log.append(f"{s.name}: dropped (was {before}tok)")

        if self._total(sections) <= self.available:
            return sections, trim_log

        # Step 2 — reduce MEDIUM sections via reduce_fn
        for s in sections:
            if self._total(sections) <= self.available:
                break
            if s.priority != Priority.MEDIUM:
                continue
            if not s.reduce_fn or not s.reduce_steps:
                continue
            for step in s.reduce_steps:
                if self._total(sections) <= self.available:
                    break
                before = s.tokens
                try:
                    s.text = s.reduce_fn(step) or ""
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "context_budget.reduce_fn_failed",
                        section=s.name,
                        error=str(exc),
                    )
                    break
                trim_log.append(
                    f"{s.name}: reduced to {step} items "
                    f"({before}→{s.tokens}tok)"
                )

        if self._total(sections) <= self.available:
            return sections, trim_log

        # Step 3 — hard-trim MEDIUM text sections proportional to overflow
        medium_sections = [
            s for s in sections if s.priority == Priority.MEDIUM and s.text
        ]
        for s in medium_sections:
            if self._total(sections) <= self.available:
                break
            overshoot = self._total(sections) - self.available
            if overshoot <= 0:
                break
            target = max(50, s.tokens - overshoot)
            if target >= s.tokens:
                continue
            before = s.tokens
            s.text = trim_to_tokens(s.text, target)
            trim_log.append(
                f"{s.name}: hard-trim {before}→{s.tokens}tok"
            )

        if self._total(sections) <= self.available:
            return sections, trim_log

        # Step 4 — last-resort trim of HIGH sections (biggest first)
        high_sections = sorted(
            (s for s in sections if s.priority == Priority.HIGH and s.text),
            key=lambda s: s.tokens,
            reverse=True,
        )
        for s in high_sections:
            if self._total(sections) <= self.available:
                break
            overshoot = self._total(sections) - self.available
            if overshoot <= 0:
                break
            target = max(self.HIGH_FLOOR_TOKENS, s.tokens - overshoot)
            if target >= s.tokens:
                continue
            before = s.tokens
            s.text = trim_to_tokens(s.text, target)
            trim_log.append(
                f"{s.name}: emergency trim {before}→{s.tokens}tok"
            )

        total = self._total(sections)
        if total > self.available:
            trim_log.append(
                f"WARNING: still {total - self.available}tok over budget "
                f"(available={self.available}, context_size={self.context_size})"
            )
            log.warning(
                "context_budget.over_budget",
                total=total,
                available=self.available,
                context_size=self.context_size,
            )

        return sections, trim_log

    # ─── Diagnostics ─────────────────────────────────────────────────

    def report(
        self, sections: list[PromptSection]
    ) -> dict[str, int | list[dict[str, int | str]]]:
        """Small summary useful for tests / log-diving."""
        total = self._total(sections)
        return {
            "context_size": self.context_size,
            "available": self.available,
            "max_output": self.max_output,
            "safety_margin": self.safety_margin,
            "total_used": total,
            "headroom": max(0, self.available - total),
            "sections": [
                {
                    "name": s.name,
                    "priority": int(s.priority),
                    "tokens": s.tokens,
                    "role": s.role,
                }
                for s in sections
            ],
        }

    # ─── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _total(sections: list[PromptSection]) -> int:
        return sum(s.tokens for s in sections if s.text)


__all__ = [
    "ContextBudget",
    "PromptSection",
    "Priority",
    "estimate_tokens",
    "trim_to_tokens",
]
