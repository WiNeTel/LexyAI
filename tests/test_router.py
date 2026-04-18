"""Tests for BrainRouter (complexity-based routing)."""

from __future__ import annotations

from lexy_core.agent.router import BrainRouter
from lexy_core.config import RoutingConfig, RoutingRule


def _router(**kwargs: object) -> BrainRouter:
    cfg = RoutingConfig(default_brain="e4b", rules=[])
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return BrainRouter(cfg)


# ── Explicit override ──────────────────────────────────────────────────────


def test_explicit_override_always_wins() -> None:
    router = _router()
    assert router.route("anything goes here", requested="a4b") == ("a4b", "explicit")
    assert router.route("anything goes here", requested="e4b") == ("e4b", "explicit")
    assert router.route("anything goes here", requested="multi") == ("multi", "explicit")


def test_unknown_brain_name_falls_through() -> None:
    """Unknown brain names are ignored (fall through to heuristic/default)."""
    router = _router()
    # 'nope' is not a known brain → should NOT be returned as explicit
    brain, reason = router.route("hi", requested="nope")
    assert brain != "nope"
    assert reason != "explicit"


# ── Regex rules from config ────────────────────────────────────────────────


def test_regex_rule_beats_heuristic() -> None:
    cfg = RoutingConfig(
        default_brain="e4b",
        rules=[RoutingRule(pattern=r"\bforever\b", brain="a4b")],
    )
    router = BrainRouter(cfg)
    brain, reason = router.route("hi")  # no rule match → default
    assert brain == "e4b"
    brain, reason = router.route("let this run forever please")
    assert brain == "a4b"
    assert reason == "rule"


# ── Complexity: E4B for simple chat ────────────────────────────────────────


def test_greeting_goes_to_e4b() -> None:
    router = _router()
    for text in [
        "hi",
        "Hallo",
        "danke!",
        "bis später",
        "wer bist du?",
        "wie geht es dir?",
    ]:
        brain, reason = router.route(text)
        assert brain == "e4b", f"{text!r} routed to {brain}"


def test_short_factual_queries_stay_on_e4b() -> None:
    router = _router()
    for text in [
        "wie spät ist es?",
        "timer bitte für 5 minuten",
        "erinnere mich in einer stunde",
        "Wetter in Hechthausen?",
    ]:
        brain, reason = router.route(text)
        assert brain == "e4b", f"{text!r} routed to {brain}"


# ── Complexity: A4B for code / deep questions ─────────────────────────────


def test_code_block_goes_to_a4b() -> None:
    router = _router()
    text = (
        "Kannst du mir diesen Python-Code reviewen?\n"
        "```python\n"
        "def foo(x): return x + 1\n"
        "```"
    )
    brain, reason = router.route(text)
    assert brain == "a4b"
    assert reason in ("complexity", "rule")


def test_analysis_request_goes_to_a4b() -> None:
    router = _router()
    text = (
        "Erklär mir bitte Schritt für Schritt, wie ein B-Tree funktioniert "
        "und welche Trade-offs es gegenüber einem Hash-Index gibt."
    )
    brain, _ = router.route(text)
    assert brain == "a4b"


def test_why_question_goes_to_a4b() -> None:
    router = _router()
    text = "Warum ist der Himmel blau?"
    brain, _ = router.route(text)
    assert brain == "a4b"


def test_write_function_goes_to_a4b() -> None:
    router = _router()
    text = "schreib mir eine funktion die eine liste sortiert"
    brain, _ = router.route(text)
    assert brain == "a4b"


def test_long_input_goes_to_a4b() -> None:
    router = _router()
    text = "Lorem ipsum dolor sit amet " * 30  # ~ 800 chars of filler
    brain, _ = router.route(text)
    assert brain == "a4b"


# ── Complexity score is exposed for tests ─────────────────────────────────


def test_a4b_default_keeps_big_brain_for_normal_questions() -> None:
    """
    With default_brain=a4b, most neutral questions stay on a4b and only
    unambiguous trivia drops down.
    """
    cfg = RoutingConfig(default_brain="a4b", rules=[])
    router = BrainRouter(cfg)

    # Clear trivia → e4b
    for text in ["hi", "danke", "wer bist du?"]:
        brain, reason = router.route(text)
        assert brain == "e4b", f"{text!r} should drop to e4b, got {brain}"

    # Normal questions → keep a4b (user has the hardware)
    for text in [
        "Wie ist das Wetter in Berlin heute?",
        "Kannst du mir das genauer erklären?",
        "Ist das sicher?",
        "Was heißt das auf Englisch?",
    ]:
        brain, _ = router.route(text)
        assert brain == "a4b", f"{text!r} should stay on a4b, got {brain}"


def test_complexity_score_for_debugging() -> None:
    router = _router()
    assert router.score("hi") < 0
    assert router.score("Warum ist der Himmel blau? Kannst du das erklären?") > 0
    assert router.score("Schreib mir einen Python-Code für Quicksort") > 0
