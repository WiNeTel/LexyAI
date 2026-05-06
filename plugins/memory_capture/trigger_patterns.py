"""
Phase 12.B — Trigger-phrase detection for explicit "remember this" requests.

Pure regex, no LLM call — runs synchronously in the
``before_user_input`` hook of the main agent. Two phrasings to handle:

* **Fact AFTER trigger** — *"merke dir, ich wohne am Nordpol"* /
  *"remember that I'm vegan"*. Captured group is everything past
  the trigger word.
* **Fact BEFORE trigger** — *"Lexy ich wohne am Nordpol, merke dir
  das"* / *"I'm vegan, remember this"*. Captured group is everything
  before the trailing trigger phrase.

Order matters: the BEFORE-trigger patterns are anchored to ``$`` (end
of message) so they only match the trailing-trigger form. We try
them FIRST — otherwise the more general AFTER-trigger regex would
catch "merke dir das" and capture only "das", missing the real fact
that lived before it.

Filler-only triggers (``Lexy merke dir das`` with no fact, or
``remember this`` standalone) get rejected via two guards:
1. minimum fact length (5 chars) — kills 1-word demonstratives,
2. fact must contain whitespace — must be at least two tokens.

Both keep the precision-recall tradeoff sane: false positives are
worse than false negatives here, because a captured noise fact
ends up in the user's facts collection forever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TriggerPattern:
    lang: str
    regex: re.Pattern[str]
    # When True, the regex captures the fact AFTER the trigger word
    # ("merke dir, FACT"). When False, the fact comes BEFORE
    # ("FACT, merke dir das"). Different post-processing for each.
    fact_after_trigger: bool


# IMPORTANT: BEFORE-trigger patterns must come first because they're
# anchored on ``$`` and would otherwise lose to the more general
# AFTER-trigger regex matching the ending "merke dir das" and
# capturing only "das" as the fact.
_TRIGGER_PATTERNS: tuple[TriggerPattern, ...] = (
    # ── German: fact BEFORE trigger ("X, merke dir das") ────────
    TriggerPattern(
        lang="de",
        regex=re.compile(
            r"^(?P<fact>.+?)[,\s]+"
            r"(?:bitte\s+)?"
            r"(?:merk(?:e|st)?\s+dir\s+das"
            r"|merk[e]?\s+das"
            r"|behalt[e]?\s+(?:dir\s+)?das"
            r"|speicher[e]?\s+das)"
            r"\s*[\.\!\?]?\s*$",
            re.IGNORECASE,
        ),
        fact_after_trigger=False,
    ),

    # ── English: fact BEFORE trigger ────────────────────────────
    TriggerPattern(
        lang="en",
        regex=re.compile(
            r"^(?P<fact>.+?),\s*"
            r"(?:please\s+)?"
            r"(?:remember\s+(?:this|that)"
            r"|save\s+this"
            r"|keep\s+this\s+in\s+mind"
            r"|don['']t\s+forget(?:\s+this)?)"
            r"\s*[\.\!\?]?\s*$",
            re.IGNORECASE,
        ),
        fact_after_trigger=False,
    ),

    # ── German: fact AFTER trigger ──────────────────────────────
    # ``merke\s+dir`` followed by *something* separator then fact.
    TriggerPattern(
        lang="de",
        regex=re.compile(
            r"merke\s+dir(?:\s+bitte)?[\s,:]+(?:dass\s+)?(?P<fact>\S.{2,}.*)",
            re.IGNORECASE,
        ),
        fact_after_trigger=True,
    ),
    TriggerPattern(
        lang="de",
        regex=re.compile(
            r"vergiss\s+nicht[\s,:]+(?:dass\s+)?(?P<fact>\S.{2,}.*)",
            re.IGNORECASE,
        ),
        fact_after_trigger=True,
    ),
    TriggerPattern(
        lang="de",
        regex=re.compile(
            r"bitte\s+(?:merken|merk\s+dir|behalt[e]?\s+im\s+kopf)[\s:,]+(?P<fact>\S.{2,}.*)",
            re.IGNORECASE,
        ),
        fact_after_trigger=True,
    ),
    TriggerPattern(
        lang="de",
        regex=re.compile(
            r"speicher(?:e|n)\s+(?:das\s+)?(?:bitte\s+)?(?:in\s+)?(?:facts?\s+)?[\s:,]+(?P<fact>\S.{2,}.*)",
            re.IGNORECASE,
        ),
        fact_after_trigger=True,
    ),

    # ── English: fact AFTER trigger ─────────────────────────────
    TriggerPattern(
        lang="en",
        regex=re.compile(
            r"remember\s+(?:that\s+|this:\s*|this,\s+)(?P<fact>\S.{2,}.*)",
            re.IGNORECASE,
        ),
        fact_after_trigger=True,
    ),
    TriggerPattern(
        lang="en",
        regex=re.compile(
            r"don['']t\s+forget[\s,:]+(?:that\s+)?(?P<fact>\S.{2,}.*)",
            re.IGNORECASE,
        ),
        fact_after_trigger=True,
    ),
    TriggerPattern(
        lang="en",
        regex=re.compile(
            r"keep\s+in\s+mind[\s:,]+(?P<fact>\S.{2,}.*)",
            re.IGNORECASE,
        ),
        fact_after_trigger=True,
    ),
    TriggerPattern(
        lang="en",
        regex=re.compile(
            r"please\s+(?:save|store|remember)[\s:,]+(?P<fact>\S.{2,}.*)",
            re.IGNORECASE,
        ),
        fact_after_trigger=True,
    ),
)


# Address-prefix variants we strip from BEFORE-trigger captures so
# stored facts don't keep "Lexy " at the front. Order matters here:
# the longer prefixes must come before the shorter ones.
_ADDRESS_PREFIXES = (
    "hey lexy ",
    "hi lexy ",
    "hallo lexy ",
    "lexy, ",
    "lexy ",
)


# Single-word demonstratives that sometimes survive a sloppy regex
# match — e.g. "merke dir das" → "das". Reject these as facts.
_DEMONSTRATIVE_NOISE = frozenset({
    "das", "dies", "es", "this", "that",
    "the thing", "die sache",
})


def _strip_addressing(fact: str) -> str:
    """Remove a leading ``Lexy ``/``Hey Lexy ``/etc. address."""
    lower = fact.lower()
    for prefix in _ADDRESS_PREFIXES:
        if lower.startswith(prefix):
            return fact[len(prefix):].lstrip()
    return fact


def _is_meaningful_fact(fact: str) -> bool:
    """Reject filler-only matches.

    Rules: at least 5 chars after trim, must contain whitespace
    (i.e. multi-word), and must not be a known noise demonstrative.
    """
    cleaned = fact.strip()
    if len(cleaned) < 5:
        return False
    if " " not in cleaned:
        return False
    if cleaned.lower() in _DEMONSTRATIVE_NOISE:
        return False
    return True


def extract_fact(
    user_message: str,
    *,
    enabled_languages: tuple[str, ...] = ("de", "en"),
) -> tuple[str, str] | None:
    """Return ``(language, fact)`` if a trigger matched, else ``None``.

    Args:
        user_message: the raw user input.
        enabled_languages: subset of ``("de", "en")`` to scan; passing
            an empty tuple disables detection entirely.

    Returns:
        ``(lang, fact)`` on a clean match, ``None`` if nothing fit
        OR the captured fact failed the meaningful-fact guard.
    """
    if not user_message or not enabled_languages:
        return None

    text = user_message.strip()
    if not text:
        return None

    for pattern in _TRIGGER_PATTERNS:
        if pattern.lang not in enabled_languages:
            continue
        match = pattern.regex.search(text)
        if not match:
            continue
        fact = match.group("fact").strip()
        # Trim trailing punctuation ("merke dir das.") so we don't
        # save dangling periods.
        fact = fact.rstrip(".!? ").strip()
        if pattern.fact_after_trigger:
            # AFTER-trigger captures might include a residual address
            # ("merke dir, Lexy ich wohne…").
            fact = _strip_addressing(fact)
        else:
            # BEFORE-trigger captures often start with "Lexy …" —
            # strip the address before storing.
            fact = _strip_addressing(fact)
        if not _is_meaningful_fact(fact):
            continue
        return (pattern.lang, fact)
    return None
