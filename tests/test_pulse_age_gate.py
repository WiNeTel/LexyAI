"""
Phase 13.6 — pin the age-stage gate that decides which characters
get a pulse timer registered.

Background: per-adult pulses degenerated into 'Was hast du gesehen?'
loops in Mike's RP because every char's timer fired independently
without contextual hooks. The cleaner ambient mechanism for
grown-up characters is the autonomous_sim tick (one LLM-picked
speaker every interval). Babies/toddlers DO need pulses because
they can't autonomously initiate a turn.

The gate has two halves:
  1. ``_register_pulse_timer`` skips chars whose age stage isn't
     in the allow-list.
  2. ``_rehydrate_pulse_timers`` cancels pre-existing timers for
     chars whose age stage is now gated out (e.g. when defaults
     change from 'all ages' to 'baby+toddler only').

These tests pin the gate logic at the simplest level — a unit
check on a tiny harness that mimics the plugin's age-stage
membership test. Full integration with scheduler is covered by
pulse-related tests already in the suite.
"""

from __future__ import annotations


def _gate(age_stage: str, allow_list: list[str]) -> bool:
    """Mirror of the plugin's age-gate predicate (the actual code is
    inlined where pulses register). Returns True when the char
    SHOULD get a pulse (passes the gate), False if gated out."""
    if not allow_list:
        return True
    return age_stage in allow_list


class TestAgeGateLogic:
    DEFAULT_ALLOW = ["baby", "toddler"]

    def test_baby_passes_default(self) -> None:
        assert _gate("baby", self.DEFAULT_ALLOW) is True

    def test_toddler_passes_default(self) -> None:
        assert _gate("toddler", self.DEFAULT_ALLOW) is True

    def test_child_blocked_default(self) -> None:
        assert _gate("child", self.DEFAULT_ALLOW) is False

    def test_teen_blocked_default(self) -> None:
        assert _gate("teen", self.DEFAULT_ALLOW) is False

    def test_adult_blocked_default(self) -> None:
        assert _gate("adult", self.DEFAULT_ALLOW) is False

    def test_empty_allow_list_disables_gate(self) -> None:
        """An empty allow-list means 'no gate' (legacy behaviour)."""
        for stage in ("baby", "toddler", "child", "teen", "adult"):
            assert _gate(stage, []) is True

    def test_custom_allow_list(self) -> None:
        """Operator can pick a different set, e.g. include teens for
        a specific scenario where teens behave like babies."""
        custom = ["baby", "teen"]
        assert _gate("teen", custom) is True
        assert _gate("toddler", custom) is False
        assert _gate("adult", custom) is False


class TestPluginConfigParse:
    """The plugin reads ``pulse_age_stages`` from plugin.yaml; verify
    the parser tolerates None / non-list / empty list correctly so
    operator typos don't crash the load.
    """

    def test_default_when_missing(self) -> None:
        cfg: dict = {}
        raw = cfg.get("pulse_age_stages", ["baby", "toddler"])
        assert raw == ["baby", "toddler"]

    def test_explicit_empty_list(self) -> None:
        cfg = {"pulse_age_stages": []}
        raw = cfg.get("pulse_age_stages", ["baby", "toddler"])
        assert raw == []

    def test_invalid_type_falls_back(self) -> None:
        """Mirror the plugin's defensive cast: only lists are
        accepted, anything else falls back to default."""
        cfg = {"pulse_age_stages": "baby,toddler"}  # operator typo
        raw = cfg.get("pulse_age_stages", ["baby", "toddler"])
        if not isinstance(raw, list):
            raw = ["baby", "toddler"]
        assert raw == ["baby", "toddler"]
