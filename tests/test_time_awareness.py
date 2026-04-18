"""Tests for Lexy's Zeitgefühl module.

Covers only the new module ``lexy_core.agent.time_awareness`` — the gap
classification, time-of-day classification, and the assembly rules for
the prompt block. The existing agent tests cover the surrounding plumbing.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from lexy_core.agent.time_awareness import (
    build_time_awareness_block,
    compute_gap,
    describe_time_of_day,
)


# ─── compute_gap ─────────────────────────────────────────────────────────────


def _at(hour: int, minute: int = 0, day: int = 16) -> float:
    return datetime(2026, 4, day, hour, minute).timestamp()


def test_gap_fresh_when_no_previous_ts() -> None:
    cat, text = compute_gap(0.0, _at(14))
    assert cat == "fresh"
    assert text == ""


def test_gap_fresh_when_previous_is_in_the_future() -> None:
    """Defensive: clock skew must not explode the block."""
    cat, _ = compute_gap(_at(14), _at(12))
    assert cat == "fresh"


def test_gap_moments_under_two_minutes() -> None:
    now = _at(14, 0)
    cat, text = compute_gap(now - 30, now)
    assert cat == "moments"
    assert text == "gerade eben"


def test_gap_short_minutes_bucket() -> None:
    now = _at(14, 0)
    cat, text = compute_gap(now - 5 * 60, now)
    assert cat == "short"
    assert "5 Minuten" in text


def test_gap_hour_bucket_snaps_to_natural_rounding() -> None:
    now = _at(14, 0)
    # 25 minutes → half hour
    cat, text = compute_gap(now - 25 * 60, now)
    assert cat == "hour"
    assert "halben Stunde" in text
    # 55 minutes → knapp einer Stunde
    cat, text = compute_gap(now - 55 * 60, now)
    assert cat == "hour"
    assert "knapp" in text


def test_gap_few_hours_bucket() -> None:
    now = _at(14, 0)
    cat, text = compute_gap(now - 3 * 3600, now)
    assert cat == "few_hours"
    assert "3 Stunden" in text


def test_gap_today_bucket() -> None:
    now = _at(18, 0)
    cat, text = compute_gap(now - 6 * 3600, now)
    assert cat == "today"
    assert "heute" in text


def test_gap_yesterday_bucket() -> None:
    now = _at(14, 0)
    cat, _ = compute_gap(now - 18 * 3600, now)
    assert cat == "yesterday"


def test_gap_days_bucket() -> None:
    now = _at(14, 0)
    cat, text = compute_gap(now - 2 * 86400, now)
    assert cat == "days"
    assert "2 Tagen" in text


def test_gap_week_bucket() -> None:
    now = _at(14, 0)
    cat, _ = compute_gap(now - 5 * 86400, now)
    assert cat == "week"


def test_gap_long_bucket_weeks() -> None:
    now = _at(14, 0)
    cat, text = compute_gap(now - 20 * 86400, now)
    assert cat == "long"
    assert "Wochen" in text


def test_gap_long_bucket_months() -> None:
    now = _at(14, 0)
    cat, text = compute_gap(now - 90 * 86400, now)
    assert cat == "long"
    assert "Monaten" in text


# ─── describe_time_of_day ────────────────────────────────────────────────────


def test_time_of_day_all_buckets() -> None:
    cases = [
        (3, "deep_night"),
        (6, "early_morning"),
        (10, "morning"),
        (13, "noon"),
        (16, "afternoon"),
        (20, "evening"),
        (23, "late_evening"),
        (0, "late_evening"),
    ]
    for hour, expected in cases:
        key, label = describe_time_of_day(datetime(2026, 4, 16, hour, 0))
        assert key == expected, f"hour={hour} got {key}, want {expected}"
        assert label  # non-empty German label


# ─── build_time_awareness_block ──────────────────────────────────────────────


def test_block_is_empty_on_fresh_session_during_daytime() -> None:
    """First message ever, ordinary daytime → the block stays silent."""
    block = build_time_awareness_block(
        previous_ts=0.0,
        now_dt=datetime(2026, 4, 16, 14, 30),
        weekday_de="Donnerstag",
    )
    assert block == ""


def test_block_is_silent_on_moments_gap_in_ordinary_daytime() -> None:
    """Mid-conversation at noon → no noise."""
    now = datetime(2026, 4, 16, 12, 30)
    prev = now.timestamp() - 30  # 30 seconds ago
    block = build_time_awareness_block(
        previous_ts=prev, now_dt=now, weekday_de="Donnerstag",
    )
    assert block == ""


def test_block_fires_on_deep_night_even_without_gap() -> None:
    """Unusual time of day is enough reason to surface the block."""
    now = datetime(2026, 4, 16, 3, 15)
    block = build_time_awareness_block(
        previous_ts=now.timestamp() - 30,
        now_dt=now,
        weekday_de="Donnerstag",
    )
    assert block != ""
    assert "Zeitgefühl" in block
    assert "tiefe Nacht" in block
    # Night hint should fire
    assert "noch wach" in block


def test_block_fires_on_three_hour_gap_evening() -> None:
    now = datetime(2026, 4, 16, 22, 30)
    prev = now.timestamp() - 3 * 3600
    block = build_time_awareness_block(
        previous_ts=prev, now_dt=now, weekday_de="Donnerstag",
    )
    assert "Zeitgefühl" in block
    assert "Stunden" in block
    # Previous clock reference included
    assert "19:30" in block
    # Weekday rendered
    assert "Donnerstag" in block


def test_block_long_gap_adds_reconnection_hint() -> None:
    """Multi-day gaps earn the warmer greeting hint."""
    now = datetime(2026, 4, 16, 14, 30)
    prev = (now - timedelta(days=5)).timestamp()
    block = build_time_awareness_block(
        previous_ts=prev, now_dt=now, weekday_de="Donnerstag",
    )
    assert "schön dich wieder zu hören" in block


def test_block_always_closes_with_nonforcing_footer() -> None:
    """The footer reminds Lexy not to force a mention — critical UX."""
    now = datetime(2026, 4, 16, 22, 30)
    prev = now.timestamp() - 3 * 3600
    block = build_time_awareness_block(
        previous_ts=prev, now_dt=now, weekday_de="Donnerstag",
    )
    assert "Kein Zwang" in block


def test_block_does_not_include_gap_line_when_only_late_evening() -> None:
    """Late-evening-alone case surfaces tageszeit but no phantom gap line."""
    now = datetime(2026, 4, 16, 23, 0)
    prev = now.timestamp() - 30  # moments gap
    block = build_time_awareness_block(
        previous_ts=prev, now_dt=now, weekday_de="Donnerstag",
    )
    assert "Zeitgefühl" in block
    assert "Tageszeit jetzt" in block
    assert "Letztes Gespräch" not in block


def test_block_uses_yesterday_phrasing_across_date_boundary() -> None:
    now = datetime(2026, 4, 16, 9, 0)
    prev = datetime(2026, 4, 15, 22, 0).timestamp()
    block = build_time_awareness_block(
        previous_ts=prev, now_dt=now, weekday_de="Donnerstag",
    )
    # 11h gap crosses a date boundary — render "gestern um 22:00"
    assert "gestern um 22:00" in block
