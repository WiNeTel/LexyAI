"""
Lexy AI - Dashboard built-in widgets.

Each widget module exposes a concrete ``BaseWidget`` subclass. The
DashboardPlugin discovers them via the ``ALL_WIDGET_CLASSES`` list
and instantiates + registers them during ``on_load()``.
"""

from __future__ import annotations

from .base_widget import BaseWidget
from .clock_widget import ClockWidget
from .memory_stats_widget import MemoryStatsWidget
from .notes_widget import NotesWidget
from .search_widget import SearchWidget
from .sessions_widget import SessionsWidget
from .system_status_widget import SystemStatusWidget
from .thoughts_widget import ThoughtsWidget
from .weather_widget import WeatherWidget

ALL_WIDGET_CLASSES: list[type[BaseWidget]] = [
    ClockWidget,
    WeatherWidget,
    MemoryStatsWidget,
    SystemStatusWidget,
    SessionsWidget,
    ThoughtsWidget,
    SearchWidget,
    NotesWidget,
]

__all__ = [
    "ALL_WIDGET_CLASSES",
    "BaseWidget",
    "ClockWidget",
    "MemoryStatsWidget",
    "NotesWidget",
    "SearchWidget",
    "SessionsWidget",
    "SystemStatusWidget",
    "ThoughtsWidget",
    "WeatherWidget",
]
