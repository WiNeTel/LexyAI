"""
Lorebook activation engine — chooses which entries fire for one round.

The engine takes:

* the active character card (the speaker for the current turn)
* the current session_id
* the user message + the trigger pulse text + recent history
* a list of all lorebooks the system can see (global + character-scoped
  for the speaker + session-scoped for the current session)

…and returns a list of :class:`ActivatedLore` items grouped by their
**position**, ready for ``_build_turn_sections`` to inject into the
prompt at the right slot.

Activation rules — designed to mirror SillyTavern's "World Info":

1. Skip disabled lorebooks and entries.
2. ``always_on`` entries fire every round, no key match needed.
3. For other entries: scan the user message + the last
   ``entry.scan_depth`` chat lines (also the pulse_text + the active
   char's state values) for any of the entry's keys. Substring,
   case-insensitive. First match wins — no recursive scanning yet.
4. Sort the activated set by ``priority`` ASC, then ``name`` ASC.
5. Truncate to the lorebook's ``token_budget`` — characters cumulative,
   later entries dropped silently when the budget is exhausted.

The engine does NOT format Markdown — it returns the raw entry content
plus a small header (``### {entry.name}``) per item. The prompt builder
joins them per position with blank lines.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

from .character_card import CharacterCard
from .lorebook_store import (
    POSITION_AFTER_PERSONA,
    POSITION_BEFORE_HISTORY,
    POSITION_BEFORE_PERSONA,
    POSITION_BEFORE_SCENARIO,
    POSITION_BEFORE_USER_MESSAGE,
    SCOPE_CHARACTER,
    SCOPE_GLOBAL,
    SCOPE_SESSION,
    Lorebook,
    LoreEntry,
)


log = logging.getLogger(__name__)


# Approximate chars-per-token used when budgeting. We deliberately
# match :mod:`context_budget`'s value so a Lorebook saying
# ``token_budget=1000`` reserves ~1000 tokens of context.
_CHARS_PER_TOKEN = 3.5


@dataclass
class ActivatedLore:
    """One entry that fired this round."""

    entry_id: str
    name: str
    content: str
    position: str
    priority: int
    matched_key: str = ""           # which key fired ("" for always_on)
    lorebook_id: str = ""
    lorebook_name: str = ""

    def render_block(self) -> str:
        """The formatted block injected into the prompt."""
        head = f"### {self.name}".strip()
        body = (self.content or "").strip()
        return f"{head}\n{body}" if body else head


@dataclass
class ActivationResult:
    """Output of :meth:`LorebookEngine.activate` — items grouped by position."""

    by_position: dict[str, list[ActivatedLore]] = field(default_factory=dict)
    skipped_budget: int = 0          # entries dropped due to budget
    scanned_text: str = ""           # the haystack used for matching (debug)

    def all_items(self) -> list[ActivatedLore]:
        out: list[ActivatedLore] = []
        for items in self.by_position.values():
            out.extend(items)
        return out


# ─── Engine ─────────────────────────────────────────────────────────


class LorebookEngine:
    """Stateless activator — call :meth:`activate` per round."""

    def __init__(self) -> None:
        # ``_compile_keys_cache`` lets the same key string compile only
        # once across many rounds. Engines are long-lived (one per
        # plugin) so the cache is bounded by the number of unique keys.
        self._compile_keys_cache: dict[str, re.Pattern[str]] = {}

    # ─── Public ──────────────────────────────────────────────────────

    def activate(
        self,
        *,
        speaker: CharacterCard,
        session_id: str,
        history: list[dict[str, str]],
        user_message: str = "",
        pulse_text: str = "",
        lorebooks: list[Lorebook],
        entries: dict[str, list[LoreEntry]],
    ) -> ActivationResult:
        """Resolve which lore entries fire and where they land.

        ``entries`` maps ``lorebook_id → list[LoreEntry]`` so the
        caller can pass a pre-fetched bundle (avoids one DB round-trip
        per book).
        """
        result = ActivationResult()
        active_books = self._filter_books(
            lorebooks, speaker_id=speaker.id, session_id=session_id,
        )
        if not active_books:
            return result

        haystack_full = self._build_haystack(
            speaker=speaker, history=history,
            user_message=user_message, pulse_text=pulse_text,
        )
        result.scanned_text = haystack_full[:400]  # debug excerpt

        # Per book, decide which of its entries fire, then merge.
        # Token budget is enforced PER BOOK so a chatty global book
        # can't starve a character-specific one.
        for book in active_books:
            if not book.enabled:
                continue
            book_entries = entries.get(book.id) or []
            if not book_entries:
                continue
            activated_for_book = self._activate_book(
                book=book,
                book_entries=book_entries,
                haystack_full=haystack_full,
                history=history,
                user_message=user_message,
                pulse_text=pulse_text,
                speaker=speaker,
            )
            for item in activated_for_book["items"]:
                result.by_position.setdefault(item.position, []).append(item)
            result.skipped_budget += activated_for_book["skipped"]

        # Per-position deterministic order so the prompt stays stable.
        for position, items in result.by_position.items():
            items.sort(key=lambda it: (it.priority, it.name.lower()))

        return result

    # ─── Filtering ──────────────────────────────────────────────────

    def _filter_books(
        self,
        lorebooks: list[Lorebook],
        *,
        speaker_id: str,
        session_id: str,
    ) -> list[Lorebook]:
        """Return only books visible from the current round's vantage."""
        visible: list[Lorebook] = []
        for book in lorebooks:
            if not book.enabled:
                continue
            if book.scope == SCOPE_GLOBAL:
                visible.append(book)
            elif book.scope == SCOPE_CHARACTER:
                if book.scope_id == speaker_id:
                    visible.append(book)
            elif book.scope == SCOPE_SESSION:
                if book.scope_id == session_id:
                    visible.append(book)
            # Unknown scopes are silently dropped — defensive.
        return visible

    # ─── Activation per book ────────────────────────────────────────

    def _activate_book(
        self,
        *,
        book: Lorebook,
        book_entries: list[LoreEntry],
        haystack_full: str,
        history: list[dict[str, str]],
        user_message: str,
        pulse_text: str,
        speaker: CharacterCard,
    ) -> dict[str, object]:
        # Sort entries deterministically before applying the budget so
        # truncation has predictable semantics.
        sorted_entries = sorted(
            (e for e in book_entries if e.enabled),
            key=lambda e: (e.priority, e.name.lower()),
        )

        budget_chars = max(0, int(book.token_budget) * _CHARS_PER_TOKEN)
        consumed_chars = 0
        items: list[ActivatedLore] = []
        skipped = 0

        for entry in sorted_entries:
            # Build per-entry haystack — entries can specify scan_depth
            # which only sees the last N messages.
            if entry.always_on:
                matched_key = ""
                fired = True
            else:
                if not entry.keys:
                    continue
                local_haystack = self._haystack_for_entry(
                    history=history,
                    user_message=user_message,
                    pulse_text=pulse_text,
                    speaker=speaker,
                    scan_depth=entry.scan_depth,
                )
                matched_key = self._first_match(entry.keys, local_haystack)
                fired = bool(matched_key)
            if not fired:
                continue

            block_chars = len(entry.content) + len(entry.name) + 5  # "### \n"
            if (
                budget_chars > 0
                and consumed_chars + block_chars > budget_chars
                and consumed_chars > 0
            ):
                # The first entry always lands even if its content
                # exceeds the budget — better one (potentially long)
                # entry than nothing at all.
                skipped += 1
                continue
            consumed_chars += block_chars
            items.append(
                ActivatedLore(
                    entry_id=entry.id,
                    name=entry.name,
                    content=entry.content,
                    position=entry.position,
                    priority=entry.priority,
                    matched_key=matched_key,
                    lorebook_id=book.id,
                    lorebook_name=book.name,
                )
            )

        return {"items": items, "skipped": skipped}

    # ─── Haystack helpers ───────────────────────────────────────────

    def _build_haystack(
        self,
        *,
        speaker: CharacterCard,
        history: list[dict[str, str]],
        user_message: str,
        pulse_text: str,
    ) -> str:
        """All-content haystack used for full-history scans.

        Scans the last 8 history messages + user_msg + pulse + speaker
        state values + speaker persona text. The state values catch
        cases like *"set clothing=Lederrüstung"* triggering the
        "Lederrüstung"-Lorebook entry.
        """
        parts: list[str] = []
        for msg in history[-8:]:
            content = str(msg.get("content") or "").strip()
            if content:
                parts.append(content)
        if user_message:
            parts.append(user_message)
        if pulse_text:
            parts.append(pulse_text)
        for v in (speaker.state or {}).values():
            v = str(v or "").strip()
            if v:
                parts.append(v)
        # Persona text is also fair game — characters often mention
        # world-elements in their persona.
        persona = (speaker.persona or "").strip()
        if persona:
            parts.append(persona)
        return "\n".join(parts).lower()

    def _haystack_for_entry(
        self,
        *,
        history: list[dict[str, str]],
        user_message: str,
        pulse_text: str,
        speaker: CharacterCard,
        scan_depth: int,
    ) -> str:
        depth = max(0, int(scan_depth))
        parts: list[str] = []
        if depth > 0:
            for msg in history[-depth:]:
                content = str(msg.get("content") or "").strip()
                if content:
                    parts.append(content)
        if user_message:
            parts.append(user_message)
        if pulse_text:
            parts.append(pulse_text)
        # Always include the active char's state values in the local
        # haystack — they describe the "now" and entries should be
        # able to fire on changing-state matches.
        for v in (speaker.state or {}).values():
            v = str(v or "").strip()
            if v:
                parts.append(v)
        return "\n".join(parts).lower()

    def _first_match(self, keys: Iterable[str], haystack_lower: str) -> str:
        """Return the first key (verbatim) that appears as a substring.

        Single-word keys use a word-boundary check so "rage" doesn't
        fire on "courage". Multi-word keys fall back to plain substring
        match because word-boundary on phrases is fiddly.
        """
        if not haystack_lower:
            return ""
        for raw in keys:
            key = (raw or "").strip()
            if not key:
                continue
            lower = key.lower()
            if " " in lower:
                if lower in haystack_lower:
                    return key
                continue
            pattern = self._compile_keys_cache.get(lower)
            if pattern is None:
                pattern = re.compile(
                    rf"(?<![a-z0-9_äöüß]){re.escape(lower)}(?![a-z0-9_äöüß])",
                    re.IGNORECASE,
                )
                self._compile_keys_cache[lower] = pattern
            if pattern.search(haystack_lower):
                return key
        return ""


# ─── Position-aware section-key mapping ─────────────────────────────


# Map each ``position`` to the prompt section name (built by the
# orchestrator) that the lore block should sit BEFORE / AFTER. The
# orchestrator's ``_build_turn_sections`` looks up the activations and
# inserts a ``lorebook_<position>`` section accordingly.
SECTION_NAME_PER_POSITION: dict[str, str] = {
    POSITION_BEFORE_PERSONA: "lorebook_before_persona",
    POSITION_AFTER_PERSONA: "lorebook_after_persona",
    POSITION_BEFORE_SCENARIO: "lorebook_before_scenario",
    POSITION_BEFORE_HISTORY: "lorebook_before_history",
    POSITION_BEFORE_USER_MESSAGE: "lorebook_before_user_message",
}


def render_position_block(items: list[ActivatedLore]) -> str:
    """Stitch all entries for one position into one prompt block."""
    if not items:
        return ""
    head = "## Lorebook"
    body = "\n\n".join(item.render_block() for item in items)
    return f"{head}\n{body}"
