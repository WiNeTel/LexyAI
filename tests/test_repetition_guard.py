"""
Phase 13.2 — pin the n-gram-based repetition detector that the
group-turn orchestrator runs after every char turn.

Mike's Castaway smoke-test had four characters writing wholesale
copies of each other's "Salz brennt / Sandkörner / Schläfen reiben"
boilerplate. The detector is the first line of defence: when a new
turn shares more than 40% of its (stopword-stripped) 4-grams with a
predecessor in the same round, the orchestrator re-prompts the LLM
once with an anti-repetition hint.
"""

from __future__ import annotations

from plugins.character_chat.group_turn import (
    _jaccard,
    _ngrams,
    detect_repetition,
)


class TestNgrams:
    def test_stopwords_dropped(self) -> None:
        grams = _ngrams("Ich gehe mit dem Hund in den Park", n=2)
        # All German stopwords removed, so we should get bi-grams of
        # the content words "gehe" / "hund" / "park".
        flat = [tok for g in grams for tok in g]
        assert "ich" not in flat
        assert "mit" not in flat
        assert "dem" not in flat
        assert "in" not in flat
        assert "den" not in flat
        assert "gehe" in flat
        assert "hund" in flat
        assert "park" in flat

    def test_short_input_falls_back_to_smaller_n(self) -> None:
        """Three content words, n=4 → falls back to bi-grams."""
        grams = _ngrams("Sand brennt heiss", n=4)
        # Without the fallback we'd get nothing. With it, we should
        # at least get bi-grams.
        assert len(grams) > 0

    def test_too_short_returns_empty(self) -> None:
        grams = _ngrams("ja", n=4)
        assert grams == set()


class TestJaccard:
    def test_identical_sets_score_1(self) -> None:
        s = {("a", "b"), ("b", "c")}
        assert _jaccard(s, s) == 1.0

    def test_disjoint_sets_score_0(self) -> None:
        a = {("a", "b")}
        b = {("c", "d")}
        assert _jaccard(a, b) == 0.0

    def test_half_overlap(self) -> None:
        a = {("a", "b"), ("b", "c")}
        b = {("a", "b"), ("x", "y")}
        # Intersection: 1 ({(a,b)}). Union: 3. Jaccard ~= 0.33.
        assert abs(_jaccard(a, b) - 1 / 3) < 0.01

    def test_empty_returns_0(self) -> None:
        assert _jaccard(set(), {("a", "b")}) == 0.0


class TestDetectRepetition:
    """The integration helper the orchestrator calls."""

    def test_high_overlap_triggers(self) -> None:
        """Two near-wholesale-copy narrative turns → repetition.

        Real Castaway-log pattern: every char emits the SAME boiler-
        plate ('Salz brennt Haut' / 'Kopf dröhnt schwer' / 'Brandung
        hämmert Schläfen'). With threshold=0.3 (sensible for German
        narrative RP — see _ngrams docstring), ~30% trigram overlap
        is the boundary."""
        prev = (
            "Salz brennt Haut Kopf dröhnt schwer Brandung hämmert "
            "Schläfen Salz brennt Haut Kopf dröhnt schwer Brandung "
            "hämmert Schläfen"
        )
        new = (
            "Salz brennt Haut Kopf dröhnt schwer Brandung hämmert "
            "Schläfen erneut salz brennt"
        )
        is_rep, jac, samples = detect_repetition(new, [prev], threshold=0.3)
        assert is_rep is True
        assert jac >= 0.3
        assert len(samples) > 0  # at least one repeated phrase identified

    def test_distinct_turns_do_not_trigger(self) -> None:
        prev = (
            "Sandra prüft die Wunde an Lenas Knie und holt frisches "
            "Wasser aus dem Bach"
        )
        new = (
            "Mira klettert auf die Palme und wirft drei Kokosnüsse "
            "herunter dann lacht sie laut"
        )
        is_rep, jac, _ = detect_repetition(new, [prev], threshold=0.4)
        assert is_rep is False
        assert jac < 0.4

    def test_no_predecessors_returns_false(self) -> None:
        is_rep, jac, _ = detect_repetition("Sandra spricht.", [], 0.4)
        assert is_rep is False
        assert jac == 0.0

    def test_empty_new_text_returns_false(self) -> None:
        is_rep, jac, _ = detect_repetition("", ["Sandra sagt was."], 0.4)
        assert is_rep is False
        assert jac == 0.0

    def test_threshold_tuning(self) -> None:
        """Same content, different thresholds → toggle behaviour."""
        a = "starre auf den Sand und reibe Schläfen während Salz brennt"
        b = "starre auf den Sand und reibe Schläfen während Salz juckt"
        # Lower threshold → triggers
        is_rep_low, _, _ = detect_repetition(b, [a], threshold=0.2)
        # Higher threshold → silent
        is_rep_high, _, _ = detect_repetition(b, [a], threshold=0.95)
        assert is_rep_low is True
        assert is_rep_high is False

    def test_multiple_predecessors_picks_max(self) -> None:
        """When checking against several predecessors, the max
        Jaccard wins — even if one is unrelated."""
        unrelated = "Mira klettert die Palme hoch."
        twin = "Sandra starrt auf den Sand und Salz brennt heiss."
        new = "Sandra starrt auf den Sand und Salz brennt heiss erneut."
        is_rep, jac, _ = detect_repetition(
            new, [unrelated, twin], threshold=0.4,
        )
        assert is_rep is True
        assert jac >= 0.4


# ─── Phase 13.5 (B+D): cross-round self-repetition ──────────────────────


import asyncio
import time

from plugins.character_chat.character_card import CharacterCard
from plugins.character_chat.group_turn import (
    GroupTurnOrchestrator,
    GroupTurnRequest,
)


class _SeqLLM:
    """Returns scripted responses in order, recording every call."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self.replies:
            return ""
        return self.replies.pop(0)


def _card(name: str = "Mira", char_id: str = "mira-id") -> CharacterCard:
    return CharacterCard(
        id=char_id,
        name=name,
        persona=f"{name} ist Surf-Lehrerin.",
        age_stage="adult",
        created_at=time.time(),
        updated_at=time.time(),
    )


class TestCrossRoundSelfRepetition:
    """Phase 13.5 (B+D) — the guard now compares the new turn
    against the SAME char's own recent turns from prior rounds, not
    only co-speakers in the current round. Mira's 'wische mir den
    Salzfilm von der Stirn' across three rounds was the trigger."""

    def test_self_repetition_across_rounds_triggers_reprompt(self) -> None:
        # First call: mostly the same boilerplate Mira already used.
        # Second call (the re-prompt): something distinct.
        new_text = (
            "wische mir den Salzfilm von der Stirn und schaue zur "
            "Brandung salz brennt schwer wische erneut den salzfilm"
        )
        prior = (
            "wische mir den Salzfilm von der Stirn und schaue zur "
            "Brandung salz brennt schwer schläfen dröhnen"
        )
        retry_text = (
            "Mira hört das Knirschen von Holz an den Felsen und "
            "richtet sich auf den Lärm aus"
        )
        llm = _SeqLLM([new_text, retry_text])
        orch = GroupTurnOrchestrator(
            llm_chat=llm, turn_selection="round_robin",
        )
        mira = _card()
        req = GroupTurnRequest(
            session_id="s1",
            history=[],
            characters=[mira],
            user_message="Was machst du jetzt?",
            prior_turns_by_char={mira.id: [prior]},
        )
        result = asyncio.run(orch.run_round(req))
        # Two LLM calls happened: original + re-prompt.
        assert len(llm.calls) == 2
        # The re-prompt's system prompt carries the Anti-Wiederholung block.
        retry_system = llm.calls[1]["messages"][0]["content"]
        assert "Anti-Wiederholung" in retry_system
        # Final visible turn is the retry (distinct, not the boilerplate).
        assert result.turns[0].content == retry_text

    def test_no_prior_turns_means_no_extra_reprompt(self) -> None:
        """First-ever turn for a char (no own history yet) — only the
        same-round predecessors guard fires (or nothing if alone)."""
        llm = _SeqLLM(["irgendein turn-text"])
        orch = GroupTurnOrchestrator(
            llm_chat=llm, turn_selection="round_robin",
        )
        mira = _card()
        req = GroupTurnRequest(
            session_id="s1",
            history=[],
            characters=[mira],
            user_message="Hi.",
            prior_turns_by_char={},  # no priors
        )
        result = asyncio.run(orch.run_round(req))
        # Single LLM call, no re-prompt.
        assert len(llm.calls) == 1
        assert result.turns[0].content == "irgendein turn-text"

    def test_distinct_prior_does_not_trigger(self) -> None:
        """When Mira's prior turn is about something completely
        different, the new turn passes the guard cleanly."""
        prior_about_palms = "klettere die Palme hoch und schüttle Kokosnüsse"
        new_about_water = (
            "wate vorsichtig in die Lagune und beobachte die Felsen "
            "im flachen Wasser"
        )
        llm = _SeqLLM([new_about_water])
        orch = GroupTurnOrchestrator(
            llm_chat=llm, turn_selection="round_robin",
        )
        mira = _card()
        req = GroupTurnRequest(
            session_id="s1",
            history=[],
            characters=[mira],
            user_message="Was tust du?",
            prior_turns_by_char={mira.id: [prior_about_palms]},
        )
        result = asyncio.run(orch.run_round(req))
        # No re-prompt; the prior is unrelated.
        assert len(llm.calls) == 1
        assert result.turns[0].content == new_about_water
