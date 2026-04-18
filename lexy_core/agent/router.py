"""
Lexy AI - BrainRouter.

Two-brain router that picks between **E4B** (fast, cheap, good for chat,
tool dispatch, classification) and **A4B** (deep, slow, good for code,
analysis, multi-step reasoning, long explanations).

Precedence
----------

1. **Explicit caller override** – ``brain="e4b"`` or ``"a4b"`` is always honoured.
2. **Regex rules from ``config.routing.rules``** – each entry maps a pattern
   to a brain. The first match wins. Use these for strong domain hints
   (e.g. "always run ``code``/``analysis`` questions on A4B").
3. **Complexity score** – a lightweight heuristic that looks at length,
   Chinese/Japanese/Korean density, code indicators, multi-step markers,
   question depth, and explicit "think about this" requests. Score > 0.5
   routes to A4B, otherwise E4B.
4. **Default** – ``RoutingConfig.default_brain`` as a final fallback.

The router returns ``(brain_name, reason)`` where *reason* is one of
``{explicit, rule, complexity, default}`` so callers can log the decision
and expose it to the UI.
"""

from __future__ import annotations

import re

from lexy_core.config import RoutingConfig
from lexy_core.utils.logging import get_logger

log = get_logger(module="brain_router")


# ── Complexity heuristic ────────────────────────────────────────────────────

# Strong A4B hints — the model that is good at code, long-form reasoning,
# structured analysis. Weight 2.0.
_A4B_STRONG = (
    r"code",
    r"coden",
    r"codieren",
    r"refactor",
    r"algorithm[a-z]*",
    r"algorithmus",
    r"algorithmen",
    r"implementier[a-z]*",
    r"implement[a-z]*",
    r"architekt[a-z]*",
    r"architecture",
    r"analys[a-z]+",
    r"analyz[a-z]+",
    r"vergleich[a-z]*",
    r"compare",
    r"erkl[aä]r[a-z]* (?:mir |me |detailliert|in depth|step[- ]by[- ]step)",
    r"explain (?:in detail|step[- ]by[- ]step|thoroughly)",
    r"plan(?:ung|e|st)?",
    r"strategi[a-z]*",
    r"begr[uü]nd[a-z]*",
    r"beweis[a-z]*",
    r"prov[a-z]+",
    r"optimi[zs][a-z]*",
    r"debug",
    r"stack trace",
    r"traceback",
    r"python",
    r"typescript",
    r"javascript",
    r"rust",
    r"golang",
    r"c\+\+",
    r"sql query",
    r"regex",
    r"regul[aä]re? ausdr[uü]ck",
    r"denk (?:mal |kurz |sorgf[aä]ltig |gr[uü]ndlich |step)",
    r"think (?:about|carefully|through|step)",
    r"\bwarum\b.*\?",  # "why" questions usually need reasoning
    r"\bwieso\b.*\?",
    r"\bweshalb\b.*\?",
    r"how does .* work",
    r"wie funktion[a-z]+",
    r"schreib (?:mir )?(?:einen?|ein|eine) (?:skript|programm|funktion|klasse|methode)",
    r"write (?:a |an )?(?:script|program|function|class|method|test)",
    r"fehler.*(?:finden|beheben)",
    r"fix.*(?:bug|error|issue)",
    r"review",
)

# Weak A4B hints — domain vocabulary that *might* be deep. Weight 0.4 each.
_A4B_WEAK = (
    r"erkl[aä]r",
    r"explain",
    r"vergleich",
    r"compare",
    r"research",
    r"recherche",
    r"design",
    r"entwurf",
    r"konzept",
    r"concept",
    r"pattern",
    r"trade[- ]?off",
    r"best practice",
    r"unterschied",
    r"difference",
)

# Strong E4B hints — obviously light-weight. Weight -1.5.
_E4B_STRONG = (
    r"^\s*(?:hi|hallo|hey|moin|gruß|servus|yo|hola|bonjour)[!.,\s]*$",
    r"^\s*(?:danke|thanks|thx|bitte|please|ok|okay|alright|klar)[!.,\s]*$",
    r"^\s*(?:ja|nein|jop|nope|yes|no|yep)[!.,\s]*$",
    r"^\s*(?:bis sp[aä]ter|tsch[uü]ss|bye|goodbye|cya|gn8|gute nacht)[!.,\s]*$",
    r"wer bist du\??",
    r"who are you\??",
    r"wie hei[ßs]t du\??",
    r"what(?:'| i)s your name\??",
    r"wie geht(?:'?s| es dir)",
    r"how are you",
)

# Medium E4B hints — quick factual look-ups / simple tool calls. Weight -0.5.
_E4B_WEAK = (
    r"wetter",
    r"weather",
    r"temperatur",
    r"uhrzeit",
    r"what time",
    r"wie sp[aä]t",
    r"timer",
    r"wecker",
    r"erinner mich",
    r"remind me",
    r"was ist heute",
)


def _compile_words(words: tuple[str, ...]) -> list[re.Pattern[str]]:
    return [re.compile(w, re.IGNORECASE | re.UNICODE) for w in words]


_A4B_STRONG_RE = _compile_words(_A4B_STRONG)
_A4B_WEAK_RE = _compile_words(_A4B_WEAK)
_E4B_STRONG_RE = _compile_words(_E4B_STRONG)
_E4B_WEAK_RE = _compile_words(_E4B_WEAK)

_CODE_FENCE_RE = re.compile(r"```")
_MULTI_SENTENCE_RE = re.compile(r"[.!?]\s+\S")


def _complexity_score(text: str) -> float:
    """
    Rough complexity score in roughly [-2, 4+].

    Positive → prefer A4B, negative → prefer E4B, zero-ish → default.
    """
    clean = text.strip()
    if not clean:
        return 0.0

    score = 0.0

    # Length signal — long inputs usually need A4B
    length = len(clean)
    if length > 400:
        score += 1.5
    elif length > 200:
        score += 0.8
    elif length > 80:
        score += 0.3
    elif length < 30:
        score -= 0.4

    # Sentences
    sentence_count = len(_MULTI_SENTENCE_RE.findall(clean)) + 1
    if sentence_count >= 4:
        score += 0.6
    elif sentence_count >= 2:
        score += 0.2

    # Code fences or inline backticks dominate → A4B
    if _CODE_FENCE_RE.search(clean):
        score += 2.0
    elif clean.count("`") >= 2:
        score += 0.6

    # Strong A4B hints
    for pattern in _A4B_STRONG_RE:
        if pattern.search(clean):
            score += 2.0
            break  # one strong hint is enough; prevents double-count

    # Weak A4B hints accumulate
    for pattern in _A4B_WEAK_RE:
        if pattern.search(clean):
            score += 0.4

    # Strong E4B hints — immediate short-circuit
    for pattern in _E4B_STRONG_RE:
        if pattern.search(clean):
            score -= 1.5
            break

    for pattern in _E4B_WEAK_RE:
        if pattern.search(clean):
            score -= 0.5

    return score


class BrainRouter:
    """
    Routes a chat turn to a brain.

    Supported brains: ``e4b`` (fast), ``a4b`` (deep), ``multi`` (4B multimodal
    — shares the llama.cpp server with the voice_gemma4 STT plugin).

    See the module docstring for the precedence list.
    """

    # Score above this promotes to A4B (when default is e4b/multi).
    COMPLEXITY_PROMOTE_A4B = 0.5
    # Score below this DOWNGRADES to a lighter brain (when default is a4b).
    # Intentionally strict: only clear trivia (-1.5 from a _E4B_STRONG match)
    # should bypass the big brain. This matches the user's hardware preference:
    # "I have the big model, use it."
    COMPLEXITY_DROP_FROM_A4B = -1.2
    # Any known brain name the router accepts as an explicit override. New
    # entries in config.brains can be passed through here without editing
    # this class — the list is just a hard filter against typos.
    KNOWN_BRAINS: tuple[str, ...] = ("e4b", "a4b", "multi")

    def __init__(self, config: RoutingConfig) -> None:
        self._config = config
        self._compiled_rules = [
            (re.compile(rule.pattern, re.IGNORECASE), rule.brain)
            for rule in config.rules
        ]

    def route(self, text: str, requested: str = "auto") -> tuple[str, str]:
        """Pick a brain. Returns ``(brain_name, reason)``."""
        if requested in self.KNOWN_BRAINS:
            return requested, "explicit"

        for pattern, brain in self._compiled_rules:
            if pattern.search(text):
                log.debug("router.match", brain=brain, pattern=pattern.pattern)
                return brain, "rule"

        score = _complexity_score(text)
        default_brain = self._config.default_brain
        log.debug(
            "router.complexity",
            score=round(score, 2),
            length=len(text),
            default=default_brain,
        )

        # Case A: A4B is the default → only clear trivia drops to a lighter brain.
        if default_brain == "a4b":
            if score <= self.COMPLEXITY_DROP_FROM_A4B:
                # Trivia. Prefer e4b; fall back to multi if e4b isn't configured.
                if "e4b" in self._config_brain_names():
                    return "e4b", "complexity"
                if "multi" in self._config_brain_names():
                    return "multi", "complexity"
            return "a4b", "default"

        # Case B: a smaller brain is the default → promote to A4B on complexity.
        if score >= self.COMPLEXITY_PROMOTE_A4B:
            return "a4b", "complexity"
        if score <= -self.COMPLEXITY_PROMOTE_A4B:
            # Still want a downgrade? Only if default is already a small brain.
            return default_brain, "complexity"

        return default_brain, "default"

    def _config_brain_names(self) -> set[str]:
        """Placeholder for future integration with LexyConfig.brains.

        The router only knows routing rules today, not the full brain list.
        We keep this helper so the complexity branch can do availability
        checks once we wire LexyConfig through (trivial next step).
        """
        return {"e4b", "a4b", "multi"}

    def score(self, text: str) -> float:
        """Expose the raw complexity score for debugging and tests."""
        return _complexity_score(text)
