"""
Unit tests for the scheduler recurring-pattern parser and next-fire
computation.

The parser is a plain, side-effect-free module — these tests do not need
a running LexyApp. They cover:

* Pattern-kind happy paths (interval / daily / weekdays / monthly)
* Weekday range wrap-around ("fr-mo" → Fr/Sa/So/Mo)
* Error paths (empty, unknown head, bad units, bad times, out-of-range days)
* ``next_fire_at`` strictly after reference for every pattern kind
"""

from __future__ import annotations

from datetime import datetime, time

import pytest

from plugins.scheduler.recurring_parser import (
    RecurringSpec,
    next_fire_at,
    parse_recurring,
)


# ─── Parsing ────────────────────────────────────────────────────────────────


class TestParseEvery:
    def test_every_seconds(self) -> None:
        spec = parse_recurring("every 30s")
        assert spec.kind == "interval"
        assert spec.interval_seconds == 30

    def test_every_minutes(self) -> None:
        spec = parse_recurring("every 45m")
        assert spec.interval_seconds == 45 * 60

    def test_every_hours(self) -> None:
        spec = parse_recurring("every 3h")
        assert spec.interval_seconds == 3 * 3600

    def test_every_case_insensitive(self) -> None:
        assert parse_recurring("Every 10M").interval_seconds == 600

    def test_every_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            parse_recurring("every 0m")

    def test_every_rejects_over_7_days(self) -> None:
        with pytest.raises(ValueError):
            parse_recurring("every 200h")  # 200h > 7d

    def test_every_rejects_bad_unit(self) -> None:
        with pytest.raises(ValueError):
            parse_recurring("every 10x")

    def test_every_rejects_missing_value(self) -> None:
        with pytest.raises(ValueError):
            parse_recurring("every m")


class TestParseDaily:
    def test_daily_basic(self) -> None:
        spec = parse_recurring("daily 09:00")
        assert spec.kind == "daily"
        assert spec.at_time == time(9, 0)

    def test_daily_late(self) -> None:
        spec = parse_recurring("daily 23:59")
        assert spec.at_time == time(23, 59)

    def test_daily_rejects_bad_time(self) -> None:
        with pytest.raises(ValueError):
            parse_recurring("daily 25:00")

    def test_daily_rejects_missing_time(self) -> None:
        with pytest.raises(ValueError):
            parse_recurring("daily")


class TestParseWeekly:
    def test_weekly_english(self) -> None:
        spec = parse_recurring("weekly mo 14:00")
        assert spec.kind == "weekdays"
        assert spec.weekdays == frozenset({0})
        assert spec.at_time == time(14, 0)

    def test_weekly_german(self) -> None:
        spec = parse_recurring("weekly montag 14:00")
        assert spec.weekdays == frozenset({0})

    def test_weekly_short_tokens(self) -> None:
        assert parse_recurring("weekly fr 12:00").weekdays == frozenset({4})

    def test_weekly_unknown_day(self) -> None:
        with pytest.raises(ValueError):
            parse_recurring("weekly xx 09:00")


class TestParseWeekdayRange:
    def test_mo_fr(self) -> None:
        spec = parse_recurring("mo-fr 18:00")
        assert spec.kind == "weekdays"
        assert spec.weekdays == frozenset({0, 1, 2, 3, 4})

    def test_sa_su_weekend(self) -> None:
        spec = parse_recurring("sa-su 10:00")
        assert spec.weekdays == frozenset({5, 6})

    def test_fr_mo_wraps_around(self) -> None:
        spec = parse_recurring("fr-mo 10:00")
        assert spec.weekdays == frozenset({4, 5, 6, 0})

    def test_mo_mo_is_just_monday(self) -> None:
        spec = parse_recurring("mo-mo 10:00")
        assert spec.weekdays == frozenset({0})


class TestParseMonthly:
    def test_monthly_basic(self) -> None:
        spec = parse_recurring("monthly 1 09:00")
        assert spec.kind == "monthly"
        assert spec.day_of_month == 1
        assert spec.at_time == time(9, 0)

    def test_monthly_rejects_day_29(self) -> None:
        with pytest.raises(ValueError):
            parse_recurring("monthly 29 09:00")

    def test_monthly_rejects_day_0(self) -> None:
        with pytest.raises(ValueError):
            parse_recurring("monthly 0 09:00")


class TestParseErrors:
    def test_empty(self) -> None:
        with pytest.raises(ValueError):
            parse_recurring("")

    def test_whitespace(self) -> None:
        with pytest.raises(ValueError):
            parse_recurring("   \t\n ")

    def test_unknown_head(self) -> None:
        with pytest.raises(ValueError):
            parse_recurring("kebab 09:00")


# ─── next_fire_at ───────────────────────────────────────────────────────────


class TestNextFireInterval:
    def test_interval_adds_seconds(self) -> None:
        spec = parse_recurring("every 30m")
        ref = datetime(2026, 4, 14, 12, 0, 0)
        nxt = next_fire_at(spec, ref)
        assert (nxt - ref).total_seconds() == 30 * 60


class TestNextFireDaily:
    def test_daily_before_today(self) -> None:
        spec = parse_recurring("daily 09:00")
        ref = datetime(2026, 4, 14, 6, 0, 0)  # 06:00 → today 09:00
        nxt = next_fire_at(spec, ref)
        assert nxt == datetime(2026, 4, 14, 9, 0, 0)

    def test_daily_after_today(self) -> None:
        spec = parse_recurring("daily 09:00")
        ref = datetime(2026, 4, 14, 12, 0, 0)  # 12:00 → tomorrow 09:00
        nxt = next_fire_at(spec, ref)
        assert nxt == datetime(2026, 4, 15, 9, 0, 0)

    def test_daily_exact_match_jumps_next_day(self) -> None:
        """Reference equal to fire time → must return the NEXT day."""
        spec = parse_recurring("daily 09:00")
        ref = datetime(2026, 4, 14, 9, 0, 0)
        nxt = next_fire_at(spec, ref)
        assert nxt == datetime(2026, 4, 15, 9, 0, 0)


class TestNextFireWeekdays:
    def test_mo_fr_on_monday_morning(self) -> None:
        spec = parse_recurring("mo-fr 18:00")
        ref = datetime(2026, 4, 13, 10, 0, 0)  # Monday 10:00 → Monday 18:00
        nxt = next_fire_at(spec, ref)
        assert nxt == datetime(2026, 4, 13, 18, 0, 0)

    def test_mo_fr_on_friday_evening_jumps_monday(self) -> None:
        spec = parse_recurring("mo-fr 18:00")
        ref = datetime(2026, 4, 17, 19, 0, 0)  # Friday 19:00 → next Monday
        nxt = next_fire_at(spec, ref)
        # 2026-04-17 is Friday; next Monday is 2026-04-20
        assert nxt == datetime(2026, 4, 20, 18, 0, 0)
        assert nxt.weekday() == 0

    def test_weekly_mo_from_tuesday(self) -> None:
        spec = parse_recurring("weekly mo 14:00")
        ref = datetime(2026, 4, 14, 12, 0, 0)  # Tuesday
        nxt = next_fire_at(spec, ref)
        assert nxt.weekday() == 0
        assert nxt.time() == time(14, 0)
        assert nxt >= ref


class TestNextFireMonthly:
    def test_monthly_mid_month_jumps_next_month(self) -> None:
        spec = parse_recurring("monthly 1 09:00")
        ref = datetime(2026, 4, 14, 12, 0, 0)  # Apr 14 → May 1
        nxt = next_fire_at(spec, ref)
        assert nxt == datetime(2026, 5, 1, 9, 0, 0)

    def test_monthly_today_earlier_stays(self) -> None:
        spec = parse_recurring("monthly 14 15:00")
        ref = datetime(2026, 4, 14, 10, 0, 0)  # Apr 14 10:00 → Apr 14 15:00
        nxt = next_fire_at(spec, ref)
        assert nxt == datetime(2026, 4, 14, 15, 0, 0)

    def test_monthly_december_wraps_to_january(self) -> None:
        spec = parse_recurring("monthly 1 09:00")
        ref = datetime(2026, 12, 15, 12, 0, 0)
        nxt = next_fire_at(spec, ref)
        assert nxt == datetime(2027, 1, 1, 9, 0, 0)
