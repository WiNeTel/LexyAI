"""
Character state-update parser.

Each character may emit a ``<state>...</state>`` block at the end of its
turn. The block contains semicolon-separated ``key=value`` pairs naming
the character's current location, mood, and last action — values the LLM
itself decides based on its turn. The parser:

1. Strips the block from the visible turn text (so the user only sees the
   in-character reply).
2. Returns a dict of state updates suitable for ``update_character(state=...)``.
3. Is robust against missing blocks, multiple blocks, and malformed
   content. A bad block is silently dropped — the turn still goes
   through, just without a state update.

The set of recognised keys is intentionally small (``location``, ``mood``,
``last_action``) so the prompt instruction stays short and the LLM doesn't
hallucinate ten new fields per turn. Unknown keys are dropped on parse.
"""

from __future__ import annotations

import re
from typing import Final


# Anchor keys that we always render with a localised label and, when set,
# always show in the prompt. The full state dict can carry ANY additional
# string-keyed fields the LLM finds useful (clothing, posture, injury,
# proximity, ...) — those render under a generic "## Sonstiges" block.
ANCHOR_STATE_KEYS: Final[tuple[str, ...]] = (
    "location",
    "mood",
    "last_action",
    "clothing",      # NEW (Mike's "nackt"-Beispiel) — physical appearance
    "posture",       # NEW — sitting / standing / lying / kneeling
    "condition",     # NEW — health / injury / fatigue
)

# Backwards-compat alias kept for any external test that imports this.
KNOWN_STATE_KEYS: Final[frozenset[str]] = frozenset(ANCHOR_STATE_KEYS)

# Hard cap on individual values. Without it, the LLM can return a paragraph
# inside a single field and turn the state into a memory dump.
_VALUE_MAX_LEN: Final[int] = 120

# Free-form key cap — keep keys short and snake_case so the prompt stays
# readable. We strip non-allowed chars defensively.
_KEY_MAX_LEN: Final[int] = 32
_KEY_RE: Final = re.compile(r"^[a-z][a-z0-9_]*$")

# Match `<state>...</state>` non-greedily, case-insensitive, across newlines.
_STATE_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    r"<state>(.*?)</state>",
    re.IGNORECASE | re.DOTALL,
)


def parse_state_block(content: str) -> tuple[str, dict[str, str]]:
    """Extract state updates from a turn's content and return cleaned text.

    Behaviour:
    * **No block present** → returns ``(content, {})`` unchanged.
    * **Single block** → strips it, parses ``key=value; key=value`` pairs,
      keeps only :data:`KNOWN_STATE_KEYS`, truncates each value.
    * **Multiple blocks** → strips all of them, merges left-to-right (later
      blocks override earlier ones for the same key).
    * **Malformed inner content** → block is stripped but ``{}`` is returned,
      so the turn text still gets cleaned even when the LLM produced
      garbage between the tags.

    Returns:
        ``(cleaned_text, {state_key: state_value, ...})``
    """
    if not content or "<state>" not in content.lower():
        return content, {}

    updates: dict[str, str] = {}
    for match in _STATE_BLOCK_RE.finditer(content):
        body = match.group(1)
        for pair in re.split(r"[;\n]+", body):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            key, _, value = pair.partition("=")
            key = key.strip().lower()
            value = value.strip().strip("'\"")
            # Validate the key shape — anchor keys are always allowed; for
            # everything else we require snake_case so the LLM can't smuggle
            # weird whitespace or symbols into the dict that would break
            # the renderer or the SQL serialiser.
            if not key or len(key) > _KEY_MAX_LEN:
                continue
            if key not in ANCHOR_STATE_KEYS and not _KEY_RE.match(key):
                continue
            if len(value) > _VALUE_MAX_LEN:
                value = value[:_VALUE_MAX_LEN].rstrip() + "…"
            updates[key] = value

    cleaned = _STATE_BLOCK_RE.sub("", content).strip()
    return cleaned, updates


def merge_state(
    current: dict[str, str], updates: dict[str, str]
) -> dict[str, str]:
    """Apply ``updates`` on top of ``current`` and return the merged dict.

    Empty-string values in ``updates`` mean "clear this field". This lets a
    character explicitly drop a state slot without us inventing a sentinel.

    Free-form keys are accepted as long as they pass :func:`parse_state_block`'s
    validation (snake_case, ≤32 chars). Anchor keys always pass. Unknown /
    malformed keys in ``updates`` are silently dropped.
    """
    merged: dict[str, str] = dict(current or {})
    for key, value in (updates or {}).items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        key = key.strip().lower()
        if not key or len(key) > _KEY_MAX_LEN:
            continue
        if key not in ANCHOR_STATE_KEYS and not _KEY_RE.match(key):
            continue
        if value == "":
            merged.pop(key, None)
        else:
            if len(value) > _VALUE_MAX_LEN:
                value = value[:_VALUE_MAX_LEN].rstrip() + "…"
            merged[key] = value
    return merged


__all__ = [
    "parse_state_block",
    "merge_state",
    "ANCHOR_STATE_KEYS",
    "KNOWN_STATE_KEYS",
]
