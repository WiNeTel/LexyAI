"""
Lexy AI - Dashboard Clock Widget.

Provides current time, date, and weekday (in German) for the dashboard.
The backend sends a time reference every 60 seconds; the frontend handles
second-level updates client-side using the unix timestamp as anchor.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from lexy_core.utils.logging import get_logger

from .base_widget import BaseWidget

log = get_logger(module="widget.clock")

# Deutsch-lokalisierte Wochentage (Montag = 0)
_WEEKDAY_DE: dict[int, str] = {
    0: "Montag",
    1: "Dienstag",
    2: "Mittwoch",
    3: "Donnerstag",
    4: "Freitag",
    5: "Samstag",
    6: "Sonntag",
}


class ClockWidget(BaseWidget):
    """Live clock widget (time, date, weekday in German)."""

    widget_id: str = "clock"
    title: str = "Uhr"
    default_size: tuple[int, int] = (2, 1)
    refresh_interval: float = 60.0  # Frontend handles sub-second ticking

    def __init__(self, api: Any) -> None:
        super().__init__(api)
        self._timezone_name: str = "Europe/Berlin"

    async def get_data(self) -> dict[str, Any]:
        """Return current time snapshot for the dashboard."""
        try:
            tz = ZoneInfo(self._timezone_name)
        except (KeyError, Exception):  # noqa: BLE001
            tz = ZoneInfo("Europe/Berlin")

        now = datetime.now(tz=tz)
        weekday = _WEEKDAY_DE.get(now.weekday(), "Unbekannt")

        return {
            "time": now.strftime("%H:%M"),
            "date": now.strftime("%d.%m.%Y"),
            "weekday": weekday,
            "timezone": self._timezone_name,
            "unix": time.time(),
        }
