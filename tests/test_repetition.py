"""Smoke tests for the RepetitionDetector."""

from __future__ import annotations

from lexy_core.llm import RepetitionDetector


def test_detects_simple_repetition() -> None:
    detector = RepetitionDetector(window_size=100, min_pattern_len=10, max_repeats=3)
    # Feed a repeating pattern that should trigger detection
    pattern = "The answer is 42. " * 6
    triggered = False
    for char in pattern:
        if detector.check(char):
            triggered = True
            break
    assert triggered


def test_non_repetition_passes() -> None:
    detector = RepetitionDetector(window_size=100, min_pattern_len=10, max_repeats=3)
    text = "This is a perfectly normal sentence without any suspicious repeats whatsoever."
    for char in text:
        assert detector.check(char) is False
