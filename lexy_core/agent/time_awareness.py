"""
Zeitgefühl — a natural sense of elapsed time for Lexy's system prompt.

The user's insight: Lexy already knows the *current* date+time, but she has
no sense of the **pause** between conversations. So she can tell you what
time it is when asked, but she never naturally reacts to the fact that
your last chat was four hours ago.

This module produces a tiny prompt-block the agent injects into the system
prompt. The framing is deliberately passive — she has the info, but is
told to only reference it when it *naturally* fits. No forced "We
haven't spoken in 2 minutes!" outbursts.

The block only appears when there IS a prior interaction; for fresh
sessions we stay silent to avoid phantom "our first chat!" commentary.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal


# ─── Gap categories ──────────────────────────────────────────────────────────

# Keys are used internally for tests / analytics; labels are the German
# natural-language strings fed into the prompt.
GapCategory = Literal[
    "fresh",       # no prior interaction (new session)
    "moments",     # < 2 min — mid-conversation, no gap-comment warranted
    "short",       # 2-15 min
    "hour",        # 15-60 min
    "few_hours",   # 1-4 h
    "today",       # 4-12 h
    "yesterday",   # 12-24 h
    "days",        # 1-3 days
    "week",        # 3-7 days
    "long",        # > 7 days
]


def compute_gap(
    previous_ts: float, now_ts: float
) -> tuple[GapCategory, str]:
    """Classify the gap between ``previous_ts`` and ``now_ts``.

    Returns ``(category, human_text)`` where ``human_text`` is a short
    German phrase like ``"vor ca. 3 Stunden"`` or ``"vor 2 Tagen"``.

    ``previous_ts == 0`` (or any non-positive value) is treated as
    "fresh session" — category ``"fresh"``, empty human text.
    """
    if previous_ts <= 0 or now_ts <= previous_ts:
        return "fresh", ""

    delta = now_ts - previous_ts
    minutes = delta / 60.0
    hours = delta / 3600.0
    days = delta / 86400.0

    if minutes < 2:
        return "moments", "gerade eben"
    if minutes < 15:
        mins_int = max(2, int(minutes))
        return "short", f"vor ca. {mins_int} Minuten"
    if minutes < 60:
        mins_int = int(minutes)
        # Snap to 15/30/45-minute buckets so it sounds natural
        if mins_int >= 50:
            return "hour", "vor knapp einer Stunde"
        if mins_int >= 35:
            return "hour", "vor ca. 45 Minuten"
        if mins_int >= 20:
            return "hour", "vor ca. einer halben Stunde"
        return "hour", f"vor ca. {mins_int} Minuten"
    if hours < 4:
        h = round(hours)
        if h <= 1:
            return "few_hours", "vor etwa einer Stunde"
        return "few_hours", f"vor ca. {h} Stunden"
    if hours < 12:
        h = round(hours)
        return "today", f"heute früher (vor ca. {h} Stunden)"
    if hours < 24:
        return "yesterday", "gestern Abend / heute früh"
    if days < 3:
        d = int(days)
        if d <= 1:
            return "days", "gestern"
        return "days", f"vor {d} Tagen"
    if days < 7:
        d = int(days)
        return "week", f"vor {d} Tagen"
    if days < 14:
        return "long", "vor über einer Woche"
    if days < 31:
        weeks = int(days / 7)
        return "long", f"vor {weeks} Wochen"
    months = int(days / 30)
    if months < 12:
        return "long", f"vor {months} Monaten"
    years = max(1, int(days / 365))
    return "long", f"vor {years} {'Jahr' if years == 1 else 'Jahren'}"


# ─── Time of day ─────────────────────────────────────────────────────────────

TimeOfDay = Literal[
    "deep_night",     # 01:00-05:00
    "early_morning",  # 05:00-08:00
    "morning",        # 08:00-12:00
    "noon",           # 12:00-14:00
    "afternoon",      # 14:00-18:00
    "evening",        # 18:00-22:00
    "late_evening",   # 22:00-01:00
]


def describe_time_of_day(now_dt: datetime) -> tuple[TimeOfDay, str]:
    """Return ``(key, german_label)`` for the hour in ``now_dt``."""
    h = now_dt.hour
    if h < 1:
        return "late_evening", "später Abend"
    if h < 5:
        return "deep_night", "tiefe Nacht"
    if h < 8:
        return "early_morning", "früher Morgen"
    if h < 12:
        return "morning", "Vormittag"
    if h < 14:
        return "noon", "Mittagszeit"
    if h < 18:
        return "afternoon", "Nachmittag"
    if h < 22:
        return "evening", "Abend"
    return "late_evening", "später Abend"


# ─── Assembled block ─────────────────────────────────────────────────────────

# Bigger gaps earn a stronger cue ("lange nicht gehört"); mid-conversation
# snippets stay silent so we don't interrupt flow with phantom commentary.
# The agent gets the category key so she can decide on tone; the footer
# reminds her NOT to force a mention.

_FOOTER = (
    "Erwähne das nur, wenn es natürlich passt — bei Begrüßung nach "
    "längerer Pause, beim späten Abend, wenn die Pause zum Kontext passt. "
    "Kein Zwang. Bei nahtlosem Gesprächsverlauf ignorieren."
)

_NIGHT_HINT = (
    "Hinweis: Es ist tiefe Nacht. Wenn Mike jetzt schreibt, ist das "
    "ungewöhnlich — eine kurze, warme Reaktion darauf kann passen (z.B. "
    "\"Du bist noch wach?\"). Aber nur wenn es natürlich kommt."
)

_LONG_GAP_HINT = (
    "Hinweis: Zwischen eurem letzten Austausch und jetzt liegt eine "
    "deutliche Pause. Eine kleine Begrüßung oder ein Kontext-Hinweis "
    "(\"schön dich wieder zu hören\") kann am Anfang passen."
)


def build_time_awareness_block(
    previous_ts: float,
    now_dt: datetime,
    weekday_de: str | None = None,
) -> str:
    """Produce the full prompt block for the system prompt.

    Returns an empty string when there is nothing worth saying — i.e.
    fresh session AND no unusual time of day. That keeps the system
    prompt clean for first-message conversations.

    ``weekday_de`` — optional pre-rendered German weekday name. We fall
    back to ``now_dt.strftime("%A")`` if not provided (English), which
    isn't great but never blocks production. The agent passes the same
    ``_WEEKDAYS_DE`` array it already uses for the date line.
    """
    gap_key, gap_text = compute_gap(
        previous_ts=previous_ts, now_ts=now_dt.timestamp()
    )
    tod_key, tod_label = describe_time_of_day(now_dt)

    # Keep the block silent when there's nothing interesting to tell her:
    # * fresh or mid-conversation ("moments") gap, AND
    # * ordinary daytime (not deep-night / late-evening)
    # In either of those states the existing date-time line already
    # suffices; a Zeitgefühl block would just be noise mid-flow.
    unusual_time = tod_key in ("deep_night", "late_evening")
    meaningful_gap = gap_key not in ("fresh", "moments")
    if not meaningful_gap and not unusual_time:
        return ""

    lines: list[str] = ["## Zeitgefühl"]

    if gap_key != "fresh" and gap_key != "moments":
        # Include the actual previous wall-clock time too so she can
        # reference it naturally ("heute Mittag" vs "gestern Abend").
        try:
            prev_dt = datetime.fromtimestamp(previous_ts)
            prev_clock = prev_dt.strftime("%H:%M")
            same_day = prev_dt.date() == now_dt.date()
            if same_day:
                when = f"heute um {prev_clock}"
            elif (now_dt.date() - prev_dt.date()) == timedelta(days=1):
                when = f"gestern um {prev_clock}"
            else:
                when = prev_dt.strftime("%d.%m. um %H:%M")
            lines.append(
                f"- Letztes Gespräch in dieser Session: {gap_text} ({when})."
            )
        except (OSError, OverflowError, ValueError):
            lines.append(f"- Letztes Gespräch in dieser Session: {gap_text}.")

    weekday_text = weekday_de or now_dt.strftime("%A")
    lines.append(
        f"- Tageszeit jetzt: {tod_label} "
        f"({now_dt.strftime('%H:%M')} am {weekday_text})."
    )

    if tod_key == "deep_night":
        lines.append(_NIGHT_HINT)
    elif gap_key in ("days", "week", "long"):
        lines.append(_LONG_GAP_HINT)

    lines.append("")  # blank line before footer for readability
    lines.append(_FOOTER)

    return "\n".join(lines)


__all__ = [
    "GapCategory",
    "TimeOfDay",
    "compute_gap",
    "describe_time_of_day",
    "build_time_awareness_block",
]
