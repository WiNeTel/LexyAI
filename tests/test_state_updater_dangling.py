"""
Phase 13.2 — pin the dangling-tag state-block stripper.

Mike's Castaway smoke-test produced ``<state>location=beach;
mood=anxious; last_action=looking at`` as visible text in the chat
because the LLM hit ``max_tokens=320`` mid-state-block — the closing
``</state>`` was never emitted, so the original regex (which requires
the closing tag) didn't strip anything. Phase 13.2 adds a fallback
that:
1. Strips the dangling open-tag fragment.
2. Best-effort parses the partial key=value pairs that are present
   before the cut, so we don't lose state info on cut-offs.
"""

from __future__ import annotations

from plugins.character_chat.state_updater import (
    merge_state,
    parse_state_block,
)


class TestDanglingStateBlock:
    def test_dangling_open_tag_is_stripped(self) -> None:
        """No closing ``</state>`` → fragment must be removed."""
        content = (
            "Sandra dreht sich langsam um und blickt zum Bach. "
            "<state>location=beach; mood=anxious; last_action=looking at"
        )
        cleaned, updates = parse_state_block(content)
        assert "<state>" not in cleaned
        assert "location=" not in cleaned
        # Real text survives.
        assert "Sandra dreht sich langsam um" in cleaned

    def test_dangling_block_partial_parse(self) -> None:
        """Even on cut-off, complete key=value pairs are extracted."""
        content = (
            "Bla bla. <state>location=beach; mood=calm; energy=tir"
        )
        cleaned, updates = parse_state_block(content)
        assert updates.get("location") == "beach"
        assert updates.get("mood") == "calm"
        # ``energy=tir`` is parseable (key+value present), so it's
        # extracted even though it's likely the cut-off truncation
        # of "tired". The split on ``;`` doesn't care about cut-off.
        assert updates.get("energy") == "tir"

    def test_complete_block_followed_by_dangling(self) -> None:
        """Both blocks should be removed AND merged into updates.
        Closed block wins on key collisions (more trustworthy)."""
        content = (
            "Hallo. <state>mood=calm; location=ufer</state> "
            "Spaeter: <state>mood=panic; energy=tired"
        )
        cleaned, updates = parse_state_block(content)
        assert "<state>" not in cleaned
        # Real text survives.
        assert "Hallo." in cleaned
        assert "Spaeter:" in cleaned
        # Closed block wins on collision (mood).
        assert updates["mood"] == "calm"
        assert updates["location"] == "ufer"
        # Dangling-only key (energy) is captured.
        assert updates["energy"] == "tired"

    def test_no_state_block_at_all(self) -> None:
        """Content without any state block stays untouched."""
        content = "Sandra sagt nichts. Sie schaut zum Meer."
        cleaned, updates = parse_state_block(content)
        assert cleaned == content
        assert updates == {}

    def test_dangling_with_no_pairs_just_strips(self) -> None:
        """Dangling-tag with garbage / no parseable pairs — strip + drop."""
        content = "Yara nickt. <state>"
        cleaned, updates = parse_state_block(content)
        assert "<state>" not in cleaned
        assert "Yara nickt." in cleaned
        assert updates == {}

    def test_case_insensitive_dangling_tag(self) -> None:
        """LLMs sometimes uppercase tags (``<STATE>``)."""
        content = "Mira lacht. <STATE>mood=happy"
        cleaned, updates = parse_state_block(content)
        assert "<STATE>" not in cleaned
        assert "<state>" not in cleaned
        assert updates.get("mood") == "happy"


class TestMergeStateUnchanged:
    """Phase 13.2 didn't touch merge_state — quick sanity that the
    no-anchor-whitelist Phase-13 contract holds."""

    def test_merge_accepts_arbitrary_snake_case(self) -> None:
        merged = merge_state({}, {"hunger": "satt", "durst": "trinken"})
        assert merged == {"hunger": "satt", "durst": "trinken"}

    def test_merge_drops_invalid_key_shape(self) -> None:
        merged = merge_state({}, {"Hunger Level": "satt"})  # space → invalid
        assert merged == {}

    def test_merge_empty_value_clears_key(self) -> None:
        merged = merge_state({"hunger": "satt"}, {"hunger": ""})
        assert merged == {}
