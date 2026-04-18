"""Unit tests for the character_chat context-budget manager.

Covers only the NEW logic introduced in this phase:

* Token estimation + trim_to_tokens boundary behaviour
* Priority-based drop ordering (LOW first, MUST never)
* reduce_fn progression for MEDIUM sections
* Dynamic context-size reaction via the fit_sections() pathway
* ``fit_sections`` reports what it trimmed

The existing group_turn tests cover the orchestrator's prompting
contract end-to-end, so these tests deliberately focus on the budget
mechanics in isolation.
"""

from __future__ import annotations

import pytest

from plugins.character_chat.context_budget import (
    ContextBudget,
    Priority,
    PromptSection,
    estimate_tokens,
    trim_to_tokens,
)


# ─── Token math ──────────────────────────────────────────────────────────────


def test_estimate_tokens_empty_returns_zero() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0  # type: ignore[arg-type]


def test_estimate_tokens_short_returns_at_least_one() -> None:
    assert estimate_tokens("a") == 1
    # A ~70-char string should be ~20 tokens at 3.5 chars/token.
    assert estimate_tokens("x" * 70) == 20


def test_trim_to_tokens_no_cap_leaves_text_alone() -> None:
    t = "Lorem ipsum " * 100
    assert trim_to_tokens(t, 0) == t
    assert trim_to_tokens("", 100) == ""


def test_trim_to_tokens_respects_cap_with_ellipsis() -> None:
    # 350 chars → max ~100 tokens at 3.5 chars/token. Cap at 20 tokens = 70 chars.
    text = "Erster Satz. Zweiter Satz. Dritter Satz. Vierter Satz. Fünfter Satz."
    trimmed = trim_to_tokens(text, 5)
    # Must be shorter than original
    assert len(trimmed) < len(text)
    # Must end with an ellipsis marker
    assert trimmed.endswith("…")


def test_trim_to_tokens_prefers_sentence_boundary() -> None:
    text = "A long first sentence that spans many words. Short one."
    trimmed = trim_to_tokens(text, 12)
    # Should cut at the period boundary in the upper 60% of the range,
    # never mid-word.
    assert not trimmed.endswith(" …")
    assert trimmed.endswith("…")


# ─── ContextBudget behaviour ─────────────────────────────────────────────────


def _sections_demo() -> list[PromptSection]:
    """Build a mini prompt spanning all priorities."""
    return [
        PromptSection("identity", Priority.MUST, "Du bist Luna.", role="system"),
        PromptSection("rules", Priority.MUST, "## Regeln\nSei kurz.", role="system"),
        PromptSection(
            "persona",
            Priority.HIGH,
            "## Persona\n" + ("Luna ist komplex. " * 200),  # ~3400 chars
            role="system",
            max_tokens=1500,
        ),
        PromptSection(
            "scenario",
            Priority.MEDIUM,
            "## Szenario\n" + ("Ein Szenario. " * 100),  # ~1400 chars
            role="system",
            max_tokens=500,
        ),
        PromptSection(
            "example_dialog",
            Priority.LOW,
            "## Beispiel-Dialog\n" + ("Luna: Hallo! " * 100),
            role="system",
            max_tokens=500,
        ),
        PromptSection(
            "history",
            Priority.MEDIUM,
            "## Bisheriger Chat\n" + ("Mike: " + "x" * 40 + "\n") * 6,
            role="user",
            reduce_fn=lambda n: "## Bisheriger Chat\n" + ("Mike: x\n" * n),
            reduce_steps=[4, 2],
        ),
        PromptSection(
            "user_message",
            Priority.HIGH,
            "## User\nHi Luna",
            role="user",
        ),
        PromptSection(
            "instruction",
            Priority.MUST,
            "## Du bist dran",
            role="user",
        ),
    ]


def test_budget_available_accounts_for_output_and_margin() -> None:
    b = ContextBudget(context_size=16384, max_output_tokens=320, safety_margin=256)
    assert b.available == 16384 - 320 - 256


def test_budget_clamps_below_minimum() -> None:
    b = ContextBudget(context_size=200, max_output_tokens=100)
    # Even with context_size=200 we floor to MIN_CONTEXT_SIZE (1024) to keep
    # the budget sane — otherwise every prompt would be trimmed to nothing.
    assert b.context_size == ContextBudget.MIN_CONTEXT_SIZE


def test_large_budget_does_not_trim() -> None:
    """At 16K there's plenty of room — no section should be dropped."""
    sections = _sections_demo()
    budget = ContextBudget(context_size=16384, max_output_tokens=320)
    fitted, log = budget.fit_sections(sections)
    names_with_text = {s.name for s in fitted if s.text}
    # All initially-populated sections still have text.
    assert "identity" in names_with_text
    assert "rules" in names_with_text
    assert "persona" in names_with_text
    assert "history" in names_with_text
    assert "user_message" in names_with_text
    assert "instruction" in names_with_text
    # Per-section caps (persona@1500, scenario@500, example@500) may still
    # fire at step 0 — that's expected, log entries should mention "capped".
    if log:
        for entry in log:
            assert "dropped" not in entry
            assert "reduced" not in entry


def test_tiny_budget_drops_low_sections_first() -> None:
    """At 2K, LOW priority sections (example_dialog) must go before MEDIUM."""
    sections = _sections_demo()
    budget = ContextBudget(context_size=2048, max_output_tokens=320)
    fitted, log = budget.fit_sections(sections)
    by_name = {s.name: s for s in fitted}
    # example_dialog is LOW → dropped first under pressure
    assert by_name["example_dialog"].text == ""
    # MUST sections are always there
    assert by_name["identity"].text.startswith("Du bist")
    assert by_name["rules"].text.startswith("## Regeln")
    assert by_name["instruction"].text.startswith("## Du bist dran")
    # user_message (HIGH) stays populated
    assert "Hi Luna" in by_name["user_message"].text
    # Log mentions the drop
    assert any("example_dialog" in entry and "dropped" in entry for entry in log)


def test_medium_section_gets_reduced_before_being_trimmed() -> None:
    """reduce_fn is called step-by-step so we shrink gracefully."""
    reduce_calls: list[int] = []

    def recording_reducer(step: int) -> str:
        reduce_calls.append(step)
        return "## Chat\n" + ("msg\n" * step)

    sections = [
        PromptSection("identity", Priority.MUST, "Id", role="system"),
        PromptSection(
            "history",
            Priority.MEDIUM,
            "## Chat\n" + ("a" * 5000),  # ~1428 tokens
            role="user",
            reduce_fn=recording_reducer,
            reduce_steps=[4, 2],
        ),
        PromptSection("instruction", Priority.MUST, "Now you", role="user"),
    ]
    budget = ContextBudget(context_size=1024, max_output_tokens=320)
    fitted, log = budget.fit_sections(sections)
    # Reduce function was invoked with at least one step before we gave up.
    assert reduce_calls, "reduce_fn should have been called"
    assert reduce_calls[0] == 4
    assert any("history" in entry and "reduced" in entry for entry in log)


def test_must_sections_never_dropped_even_under_extreme_pressure() -> None:
    """Even at absurdly tight budget, MUST sections survive."""
    sections = [
        PromptSection("identity", Priority.MUST, "Du bist X.", role="system"),
        PromptSection("rules", Priority.MUST, "## Regeln\nkurz.", role="system"),
        PromptSection(
            "persona",
            Priority.HIGH,
            "Persona text " * 500,
            role="system",
        ),
        PromptSection(
            "user_message",
            Priority.HIGH,
            "Hi",
            role="user",
            max_tokens=800,
        ),
        PromptSection("instruction", Priority.MUST, "## Du bist dran", role="user"),
    ]
    budget = ContextBudget(context_size=1024, max_output_tokens=320)
    fitted, _log = budget.fit_sections(sections)
    by_name = {s.name: s for s in fitted}
    assert by_name["identity"].text == "Du bist X."
    assert by_name["rules"].text == "## Regeln\nkurz."
    assert by_name["instruction"].text == "## Du bist dran"


def test_high_section_trimmed_only_as_last_resort() -> None:
    """persona gets emergency-trimmed if nothing else fits."""
    big_persona = "Persona " * 2000  # ~4600 tokens
    sections = [
        PromptSection("identity", Priority.MUST, "Id", role="system"),
        PromptSection(
            "persona", Priority.HIGH, big_persona, role="system", max_tokens=0
        ),
        PromptSection("instruction", Priority.MUST, "Du", role="user"),
    ]
    budget = ContextBudget(context_size=1024, max_output_tokens=320)
    fitted, log = budget.fit_sections(sections)
    persona = next(s for s in fitted if s.name == "persona")
    assert persona.tokens > 0  # not dropped
    assert persona.tokens <= budget.available  # trimmed
    # Emergency-trim was applied
    assert any("emergency" in entry for entry in log)


def test_fit_sections_is_idempotent_when_already_fits() -> None:
    sections = [
        PromptSection("identity", Priority.MUST, "hi", role="system"),
        PromptSection("instruction", Priority.MUST, "go", role="user"),
    ]
    budget = ContextBudget(context_size=16384, max_output_tokens=320)
    fitted1, log1 = budget.fit_sections(sections)
    fitted2, log2 = budget.fit_sections(fitted1)
    # Second pass produces no additional trims.
    assert log2 == []
    assert [s.text for s in fitted1] == [s.text for s in fitted2]


def test_budget_report_returns_diagnostic_fields() -> None:
    sections = _sections_demo()
    budget = ContextBudget(context_size=16384, max_output_tokens=320)
    report = budget.report(sections)
    assert report["context_size"] == 16384
    assert report["max_output"] == 320
    assert report["total_used"] > 0
    assert isinstance(report["sections"], list)
    # Sections are sorted/kept in input order in the report.
    section_names = [s["name"] for s in report["sections"]]
    assert "identity" in section_names
