"""
Avatar World — TimeOfDay helper.

Maps the current wall-clock hour onto a 5-bucket day schedule:

    06:00 - 09:00  morning
    09:00 - 12:00  midday
    12:00 - 18:00  afternoon
    18:00 - 22:00  evening
    22:00 - 06:00  night

Each bucket also has a recommended background id from the plugin's
configured ``available_backgrounds``. The frontend's skybox manager
crossfades when the bucket changes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from plugins.avatar_world.state import TimeOfDay


# Centralised hour cutoffs so emotion/outfit routers can reason about
# "is it evening?" without recomputing thresholds in three places.
_BUCKETS: tuple[tuple[int, int, TimeOfDay], ...] = (
    (6, 9, "morning"),
    (9, 12, "midday"),
    (12, 18, "afternoon"),
    (18, 22, "evening"),
)


def bucket_for_hour(hour: int) -> TimeOfDay:
    """Return the TimeOfDay bucket that the given 24h hour falls into."""
    h = max(0, min(23, int(hour)))
    for lo, hi, name in _BUCKETS:
        if lo <= h < hi:
            return name
    # Anything not caught above is the night band (22:00-06:00).
    return "night"


def current_bucket(now: datetime | None = None) -> TimeOfDay:
    ts = now or datetime.now()
    return bucket_for_hour(ts.hour)


# Default background ids per bucket. ``pick_background`` only returns one
# from the plugin's whitelist so a user can prune the list without
# breaking anything — we fall back to the first allowed id.
_DEFAULT_BACKGROUND_PER_BUCKET: dict[TimeOfDay, str] = {
    "morning": "city_morning",
    "midday": "city_day",
    "afternoon": "city_day",
    "evening": "city_evening",
    "night": "city_night",
}


def pick_background(
    bucket: TimeOfDay, available: Sequence[str]
) -> str:
    """Pick a background id for a bucket from the plugin's whitelist.

    If the bucket's preferred id is in the whitelist we use it; otherwise
    we fall back to the first whitelisted id (or empty string when the
    list is empty, which a sane plugin config shouldn't produce).
    """
    preferred = _DEFAULT_BACKGROUND_PER_BUCKET.get(bucket, "")
    if preferred and preferred in available:
        return preferred
    return available[0] if available else ""
