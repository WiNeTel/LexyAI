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
