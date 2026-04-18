"""
Lexy AI - Dashboard BaseWidget.

Abstract base class for all dashboard widgets. Each widget provides:

* ``widget_id``         – unique slug used as the WS message key
* ``title``             – display name in the dashboard
* ``default_size``      – (columns, rows) in the CSS grid
* ``refresh_interval``  – seconds between automatic ``get_data()`` calls
                          (0 = no auto-refresh; push / on-demand only)

Concrete widgets are discovered by the DashboardPlugin on startup and
registered into the widget registry automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseWidget(ABC):
    """Abstract base class for dashboard widgets."""

    widget_id: str
    title: str
    default_size: tuple[int, int]  # (width, height) in grid units
    refresh_interval: float  # seconds between updates (0 = no auto-refresh)

    def __init__(self, api: Any) -> None:
        self._api = api

    @abstractmethod
    async def get_data(self) -> dict[str, Any]:
        """Return current widget data. Called periodically by dashboard."""
        ...

    def to_manifest(self) -> dict[str, Any]:
        """Serialise widget metadata for the frontend registry."""
        return {
            "widget_id": self.widget_id,
            "title": self.title,
            "default_size": list(self.default_size),
            "refresh_interval": self.refresh_interval,
        }
