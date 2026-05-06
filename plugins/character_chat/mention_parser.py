"""
Natural-language mention parser for character_chat.

The original ``_parse_at_mentions`` (in :mod:`group_turn`) requires the user
to write ``@Mara`` to force Mara into position 0 of the speaker queue. In
practice users address characters naturally — *"Mara, schau mal zum Captain.
Drell, sicher die Tür!"* — without any prefix. This module adds a
deterministic, regex-based detector for plain-name mentions that runs
**before** the LLM-based ``_ask_llm_for_order`` fallback, so a clear user
message turns into a clear forced-speaker order without spending a token.

Both parsers return ``character_id`` lists in **first-occurrence order**,
which is what :class:`GroupTurnOrchestrator._pick_speakers` already expects.
"""

from __future__ import annotations

import re
from typing import Iterable

from .character_card import CharacterCard


# A word boundary anchor used both at start and end of the name. We don't use
# Python's ``\b`` directly because German names contain umlauts that fall
# outside the default word-class for some Unicode locales, so we hand-roll a
# match that allows letters/digits/underscore on either side. Word-boundary
# is enforced by the surrounding character classes in ``_compile_pattern``.
_NAME_CHARS = r"A-Za-zÀ-ÿ0-9_"


def _compile_pattern(name: str) -> re.Pattern[str]:
    """Build a case-insensitive word-boundary pattern for a single name.

    The name itself is escaped so RP characters with punctuation in their
    name (e.g. "St. John") don't break the regex.
    """
    escaped = re.escape(name)
    # Word boundary that respects accents/umlauts.
    pattern = rf"(?<![{_NAME_CHARS}]){escaped}(?![{_NAME_CHARS}])"
    return re.compile(pattern, re.IGNORECASE)


def parse_nl_mentions(
    text: str,
    candidates: Iterable[CharacterCard],
) -> list[str]:
    """Return ``character_id``s mentioned in ``text``, in occurrence order.

    Detection rules:

    * Each character's exact ``name`` is searched with case-insensitive
      word-boundary matching (so "Mara" matches "Mara,", "Mara!", "MARA",
      but NOT "Maraschino").
    * The order in the returned list is the order of **first occurrence**
      in ``text``. Ties (very rare — same offset can't happen) fall back
      to candidate iteration order.
    * Characters whose name does not appear are simply omitted.
    * Empty text or no candidates → empty list.

    No ``@`` prefix is required. The existing ``_parse_at_mentions`` still
    runs first for the explicit case, so power users who type ``@Name``
    keep that contract; this parser only kicks in when the explicit form
    yielded nothing.
    """
    if not text:
        return []

    cards = [c for c in candidates if (c.name or "").strip()]
    if not cards:
        return []

    # Build (offset, char_id) pairs. We drop duplicates per character (keep
    # earliest offset) so a name mentioned twice doesn't claim two slots.
    earliest: dict[str, int] = {}
    for card in cards:
        match = _compile_pattern(card.name).search(text)
        if match is None:
            continue
        # If multiple cards share the same name (it can happen — collision is
        # only enforced for non-archived characters), keep the first one we
        # encounter and skip the rest. The store's name-uniqueness guarantee
        # makes this a defensive safeguard, not a real failure mode.
        if card.id in earliest:
            continue
        earliest[card.id] = match.start()

    # Sort by occurrence order in the source text.
    return [cid for cid, _ in sorted(earliest.items(), key=lambda kv: kv[1])]


__all__ = ["parse_nl_mentions"]
