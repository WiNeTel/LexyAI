"""Tests for the attach/spawn → Scene Director gate (pure decision).

The scheduling wrapper (`_maybe_trigger_scene_director`) is a thin
fire-and-forget over this pure predicate; the predicate is where the logic
lives, so that's what we pin.
"""

from __future__ import annotations

from plugins.character_chat.character_chat_plugin import (
    _should_trigger_scene_director,
)

_BABY = "Shani hat ein 2 Monate altes Baby dabei"
_NO_DEP = "Ein einsamer Söldner ohne Bindungen"


def test_off_never_triggers() -> None:
    # Even an obvious dependent must not fire when the director is off.
    assert _should_trigger_scene_director("off", _BABY, {}) is False


def test_confirm_triggers_on_dependent() -> None:
    assert _should_trigger_scene_director("confirm", _BABY, {}) is True


def test_auto_triggers_on_dependent() -> None:
    assert _should_trigger_scene_director("auto", _BABY, {}) is True


def test_no_dependent_does_not_trigger() -> None:
    assert _should_trigger_scene_director("confirm", _NO_DEP, {}) is False
    assert _should_trigger_scene_director("auto", _NO_DEP, {}) is False


def test_dependent_can_come_from_relationships() -> None:
    # The hint may live in relationships, not just the persona text.
    assert _should_trigger_scene_director(
        "confirm", "Programmiererin", {"c_baby": "ihr Baby"}
    ) is True
