"""
Lexy AI - Scheduler Recurring Pattern Parser.

Lightweight parser for the human-readable recurring patterns the Scheduler
accepts. Returns a :class:`RecurringSpec` describing when to fire next,
plus a helper that computes the next absolute ``fire_at`` timestamp from a
reference ``datetime``.

Supported pattern syntax
------------------------

* ``daily HH:MM``           — every day at ``HH:MM``
* ``every Nm`` / ``every Nh`` / ``every Ns``
                            — fixed interval in minutes / hours / seconds
* ``mo-fr HH:MM`` /
  ``sa-su HH:MM``           — weekday range, daily at ``HH:MM``
* ``weekly <day> HH:MM``    — a specific weekday (``mo``..``su``) at ``HH:MM``
* ``monthly DD HH:MM``      — the Nth day of the month at ``HH:MM``

Weekday tokens: ``mo mon``, ``di tue``, ``mi wed``, ``do thu``, ``fr fri``,
``sa sat``, ``so sun``. German and English are both accepted.

Design notes
------------
* The parser is intentionally **strict** — anything it does not recognise
  raises :class:`ValueError`. The scheduler UI catches that and surfaces
  the error to the user instead of silently producing odd schedules.
* The parser does not know anything about timezones. It operates in
  the server's local time (``datetime.now()``), which matches the rest
  of the scheduler plugin.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta


# ─── Weekday tokens ─────────────────────────────────────────────────────────

_WEEKDAY_TOKENS: dict[str, int] = {
    # Monday = 0 to match datetime.weekday()
    "mo": 0, "mon": 0, "monday": 0, "montag": 0,
    "di": 1, "tu": 1, "tue": 1, "tuesday": 1, "dienstag": 1,
    "mi": 2, "we": 2, "wed": 2, "wednesday": 2, "mittwoch": 2,
    "do": 3, "th": 3, "thu": 3, "thursday": 3, "donnerstag": 3,
    "fr": 4, "fri": 4, "friday": 4, "freitag": 4,
    "sa": 5, "sat": 5, "saturday": 5, "samstag": 5,
    "so": 6, "su": 6, "sun": 6, "sunday": 6, "sonntag": 6,
}


# ─── Spec dataclass ─────────────────────────────────────────────────────────


@dataclass
class RecurringSpec:
    """Normalised representation of a recurring schedule.

    Exactly one of ``interval_seconds`` or ``at_time`` is set (for
    ``every``-patterns the former, for time-of-day patterns the latter).
    ``weekdays`` is ``None`` for daily/monthly patterns.
    ``day_of_month`` is only set for monthly patterns.
    """

    kind: str                           # "interval" | "daily" | "weekdays" | "monthly"
    interval_seconds: int = 0
    at_time: time | None = None
    weekdays: frozenset[int] | None = None
    day_of_month: int | None = None

    @property
    def is_interval(self) -> bool:
        return self.kind == "interval"


# ─── Parser ─────────────────────────────────────────────────────────────────


def parse_recurring(pattern: str) -> RecurringSpec:
    """Parse a recurring pattern string into a :class:`RecurringSpec`.

    Raises:
        ValueError: if the pattern is empty, malformed, or uses unknown
            tokens.
    """
    if not isinstance(pattern, str):
        raise ValueError(f"pattern must be a string, got {type(pattern).__name__}")
    raw = pattern.strip().lower()
    if not raw:
        raise ValueError("empty recurring pattern")

    tokens = raw.split()
    head = tokens[0]

    if head == "every":
        return _parse_every(tokens)
    if head == "daily":
        return _parse_daily(tokens)
    if head == "weekly":
        return _parse_weekly(tokens)
    if head == "monthly":
        return _parse_monthly(tokens)
    if "-" in head and head.split("-", 1)[0] in _WEEKDAY_TOKENS:
        return _parse_weekday_range(tokens)

    raise ValueError(f"unknown recurring pattern: {pattern!r}")


def _parse_every(tokens: list[str]) -> RecurringSpec:
    if len(tokens) != 2:
        raise ValueError("'every' expects one argument, e.g. 'every 30m'")
    token = tokens[1]
    if len(token) < 2 or not token[:-1].isdigit():
        raise ValueError(f"invalid interval token: {token!r}")
    value = int(token[:-1])
    unit = token[-1]
    if value <= 0:
        raise ValueError("interval must be positive")
    if unit == "s":
        seconds = value
    elif unit == "m":
        seconds = value * 60
    elif unit == "h":
        seconds = value * 3600
    else:
        raise ValueError(f"unknown interval unit: {unit!r} (expected s/m/h)")
    if seconds > 7 * 24 * 3600:
        raise ValueError("interval must not exceed 7 days")
    return RecurringSpec(kind="interval", interval_seconds=seconds)


def _parse_daily(tokens: list[str]) -> RecurringSpec:
    if len(tokens) != 2:
        raise ValueError("'daily' expects a time, e.g. 'daily 09:00'")
    return RecurringSpec(kind="daily", at_time=_parse_hhmm(tokens[1]))


def _parse_weekly(tokens: list[str]) -> RecurringSpec:
    if len(tokens) != 3:
        raise ValueError("'weekly' expects <day> <HH:MM>, e.g. 'weekly mo 14:00'")
    day_token = tokens[1]
    if day_token not in _WEEKDAY_TOKENS:
        raise ValueError(f"unknown weekday token: {day_token!r}")
    weekday = _WEEKDAY_TOKENS[day_token]
    at = _parse_hhmm(tokens[2])
    return RecurringSpec(kind="weekdays", at_time=at, weekdays=frozenset({weekday}))


def _parse_monthly(tokens: list[str]) -> RecurringSpec:
    if len(tokens) != 3:
        raise ValueError(
            "'monthly' expects <day-of-month> <HH:MM>, e.g. 'monthly 1 09:00'"
        )
    try:
        day = int(tokens[1])
    except ValueError as exc:
        raise ValueError(f"invalid day-of-month: {tokens[1]!r}") from exc
    if not 1 <= day <= 28:
        # Days 29-31 don't exist in every month; keep it simple by
        # requiring users to pick a day that occurs in every month.
        raise ValueError(
            "monthly day must be between 1 and 28 (days 29-31 are rejected "
            "because they do not occur every month)"
        )
    at = _parse_hhmm(tokens[2])
    return RecurringSpec(kind="monthly", at_time=at, day_of_month=day)


def _parse_weekday_range(tokens: list[str]) -> RecurringSpec:
    if len(tokens) != 2:
        raise ValueError(
            "weekday range expects <range> <HH:MM>, e.g. 'mo-fr 18:00'"
        )
    range_token = tokens[0]
    start_token, end_token = range_token.split("-", 1)
    if start_token not in _WEEKDAY_TOKENS:
        raise ValueError(f"unknown weekday token: {start_token!r}")
    if end_token not in _WEEKDAY_TOKENS:
        raise ValueError(f"unknown weekday token: {end_token!r}")
    start = _WEEKDAY_TOKENS[start_token]
    end = _WEEKDAY_TOKENS[end_token]
    days: set[int] = set()
    # Walk the range modulo 7 so "fr-mo" means Fr, Sa, So, Mo.
    i = start
    while True:
        days.add(i)
        if i == end:
            break
        i = (i + 1) % 7
        if len(days) >= 7:
            break
    return RecurringSpec(
        kind="weekdays",
        at_time=_parse_hhmm(tokens[1]),
        weekdays=frozenset(days),
    )


def _parse_hhmm(token: str) -> time:
    if ":" not in token:
        raise ValueError(f"invalid HH:MM token: {token!r}")
    parts = token.split(":")
    if len(parts) != 2:
        raise ValueError(f"invalid HH:MM token: {token!r}")
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"invalid HH:MM token: {token!r}") from exc
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"HH:MM out of range: {token!r}")
    return time(hour=hour, minute=minute)


# ─── Next-fire computation ──────────────────────────────────────────────────


def next_fire_at(spec: RecurringSpec, reference: datetime) -> datetime:
    """Given a spec and a reference ``datetime``, return the next firing moment.

    The returned ``datetime`` is always strictly **after** ``reference``.
    For ``weekdays`` patterns, we iterate day-by-day (at most 7 hops) to
    find the next matching weekday whose at_time is in the future.
    """
    if spec.kind == "interval":
        return reference + timedelta(seconds=spec.interval_seconds)

    at = spec.at_time
    if at is None:
        raise ValueError("time-of-day pattern missing at_time")

    if spec.kind == "daily":
        candidate = reference.replace(
            hour=at.hour, minute=at.minute, second=0, microsecond=0
        )
        if candidate <= reference:
            candidate = candidate + timedelta(days=1)
        return candidate

    if spec.kind == "weekdays":
        if not spec.weekdays:
            raise ValueError("weekdays pattern missing weekday set")
        for offset in range(0, 8):
            day = reference + timedelta(days=offset)
            candidate = day.replace(
                hour=at.hour, minute=at.minute, second=0, microsecond=0
            )
            if candidate <= reference:
                continue
            if candidate.weekday() in spec.weekdays:
                return candidate
        # Should be unreachable: an 8-day horizon always hits every weekday.
        raise RuntimeError("weekdays pattern could not resolve next fire")

    if spec.kind == "monthly":
        if spec.day_of_month is None:
            raise ValueError("monthly pattern missing day_of_month")
        year = reference.year
        month = reference.month
        candidate = reference.replace(
            day=spec.day_of_month,
            hour=at.hour,
            minute=at.minute,
            second=0,
            microsecond=0,
        )
        if candidate <= reference:
            month += 1
            if month == 13:
                month = 1
                year += 1
            candidate = candidate.replace(year=year, month=month)
        return candidate

    raise ValueError(f"unsupported spec kind: {spec.kind!r}")
