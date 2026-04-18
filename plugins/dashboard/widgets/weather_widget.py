"""
Lexy AI - Dashboard Weather Widget.

Displays current weather data by delegating to the weather plugin.  If the
weather plugin is not loaded (or has no cached data yet), the widget returns
``{available: false}`` so the frontend can show a placeholder.

The weather plugin exposes its tool handler ``_handle`` directly.  We call it
with the default location (no arguments) to fetch fresh data.
"""

from __future__ import annotations

from typing import Any

from lexy_core.utils.logging import get_logger

from .base_widget import BaseWidget

log = get_logger(module="widget.weather")

# Map Open-Meteo condition strings to icon keys the frontend understands.
_ICON_MAP: dict[str, str] = {
    "clear sky": "clear",
    "mainly clear": "clear",
    "partly cloudy": "partly_cloudy",
    "overcast": "overcast",
    "fog": "fog",
    "depositing rime fog": "fog",
    "light drizzle": "drizzle",
    "moderate drizzle": "drizzle",
    "dense drizzle": "drizzle",
    "light rain": "rain",
    "moderate rain": "rain",
    "heavy rain": "rain",
    "rain showers": "rain",
    "heavy rain showers": "rain",
    "violent rain showers": "rain",
    "light snow": "snow",
    "moderate snow": "snow",
    "heavy snow": "snow",
    "thunderstorm": "thunderstorm",
    "thunderstorm with hail": "thunderstorm",
    "thunderstorm with heavy hail": "thunderstorm",
}


class WeatherWidget(BaseWidget):
    """Current weather from the weather plugin."""

    widget_id: str = "weather"
    title: str = "Wetter"
    default_size: tuple[int, int] = (2, 2)
    refresh_interval: float = 300.0  # 5 Minuten

    def __init__(self, api: Any) -> None:
        super().__init__(api)

    async def get_data(self) -> dict[str, Any]:
        """Fetch weather via the weather plugin's tool handler."""
        weather_plugin = self._api.get_plugin("weather")
        if weather_plugin is None:
            log.debug("widget.weather.no_plugin")
            return {"available": False}

        try:
            # WeatherPlugin._handle() accepts optional kwargs and uses defaults
            result: dict[str, Any] = await weather_plugin._handle()
        except Exception as exc:  # noqa: BLE001
            log.warning("widget.weather.fetch_failed", error=str(exc))
            return {"available": False, "error": str(exc)}

        if "error" in result:
            return {"available": False, "error": result["error"]}

        condition = str(result.get("conditions", "unknown"))
        icon = _ICON_MAP.get(condition, "unknown")

        return {
            "available": True,
            "location": result.get("location", ""),
            "temperature": result.get("temperature"),
            "condition": condition,
            "humidity": result.get("humidity"),
            "wind_speed": result.get("wind_speed"),
            "icon": icon,
        }
