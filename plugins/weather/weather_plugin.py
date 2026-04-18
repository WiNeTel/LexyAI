"""
Lexy AI - Weather Plugin.

Exposes a single ``get_weather`` tool backed by Open-Meteo
(https://open-meteo.com) – free, no API key, no rate limit for personal use.
"""

from __future__ import annotations

from typing import Any

import httpx

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

log = get_logger(module="weather_plugin")


WEATHER_CODE_MAP: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "rain showers",
    81: "heavy rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}

TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "City name (e.g. 'Berlin', 'Munich', 'New York')",
        },
        "units": {
            "type": "string",
            "enum": ["metric", "imperial"],
            "description": "Temperature units (default metric = °C)",
        },
        "forecast_hours": {
            "type": "integer",
            "description": (
                "Number of upcoming hours to include in the forecast "
                "(0 = current conditions only, max 48). Use this when the "
                "user asks about 'heute Abend', 'heute Nacht', 'morgen "
                "früh', 'the next few hours', etc."
            ),
        },
    },
    "required": ["location"],
}


class WeatherPlugin(BasePlugin):
    """Open-Meteo weather tool."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._client: httpx.AsyncClient | None = None
        self._default_location: str = "Berlin"
        self._default_lat: float = 52.52
        self._default_lon: float = 13.41
        self._default_units: str = "metric"
        self._timezone: str = "Europe/Berlin"

    async def on_load(self) -> None:
        config = self.api.get_config()
        self._default_location = str(config.get("location", "Berlin"))
        self._default_lat = float(config.get("latitude", 52.52))
        self._default_lon = float(config.get("longitude", 13.41))
        self._default_units = str(config.get("units", "metric"))
        self._timezone = str(config.get("timezone", "Europe/Berlin"))
        self._client = httpx.AsyncClient(timeout=10.0)

    async def on_enable(self) -> None:
        self.api.register_tool(
            name="get_weather",
            handler=self._handle,
            description=(
                "Get the current weather for a given city. Uses Open-Meteo. "
                "Returns temperature, wind, and a human-readable description."
            ),
            schema=TOOL_SCHEMA,
        )

    async def on_disable(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ─── Tool handler ───────────────────────────────────────────

    async def _handle(
        self,
        location: str | None = None,
        units: str | None = None,
        forecast_hours: int | None = None,
    ) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("Weather plugin not loaded")

        target_location = location or self._default_location
        unit_system = units or self._default_units
        temp_unit = "celsius" if unit_system == "metric" else "fahrenheit"

        try:
            hours = max(0, min(48, int(forecast_hours or 0)))
        except (TypeError, ValueError):
            hours = 0

        lat, lon = await self._geocode(target_location)
        if lat is None or lon is None:
            return {
                "error": f"Could not geocode location: {target_location}",
                "location": target_location,
            }

        params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
            "temperature_unit": temp_unit,
            "timezone": self._timezone,
        }
        if hours > 0:
            import datetime as _dt

            params["hourly"] = (
                "temperature_2m,weather_code,precipitation_probability,wind_speed_10m"
            )
            # Open-Meteo returns hourly data in full-day blocks starting at
            # 00:00 local time. To cover ``hours`` hours from the current
            # moment we need enough days so that (current_hour + hours)
            # still fits inside the returned window. Always request at
            # least 2 days — cheap and avoids "morgen früh" gaps.
            current_hour = _dt.datetime.now().hour
            total_hours_needed = current_hour + hours
            params["forecast_days"] = min(3, max(2, (total_hours_needed // 24) + 1))

        try:
            resp = await self._client.get(
                "https://api.open-meteo.com/v1/forecast",
                params=params,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.error("weather.forecast_failed", error=str(exc))
            return {"error": str(exc), "location": target_location}

        data = resp.json()
        current = data.get("current", {}) or {}
        code = int(current.get("weather_code", 0) or 0)
        description = WEATHER_CODE_MAP.get(code, "unknown conditions")
        unit_label = "°C" if unit_system == "metric" else "°F"

        result: dict[str, Any] = {
            "location": target_location,
            "latitude": lat,
            "longitude": lon,
            "temperature": current.get("temperature_2m"),
            "unit": unit_label,
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "conditions": description,
            "weather_code": code,
        }

        if hours > 0:
            hourly = data.get("hourly", {}) or {}
            times: list[str] = list(hourly.get("time", []))
            temps: list[float] = list(hourly.get("temperature_2m", []))
            codes: list[int] = list(hourly.get("weather_code", []))
            precip: list[int] = list(hourly.get("precipitation_probability", []) or [])
            winds: list[float] = list(hourly.get("wind_speed_10m", []) or [])

            current_iso = str(current.get("time", "") or "")
            start_idx = 0
            if current_iso and current_iso in times:
                start_idx = times.index(current_iso)
            elif times:
                # Fall back: drop past entries by matching "YYYY-MM-DDTHH"
                try:
                    import datetime as _dt

                    now_local = _dt.datetime.now().strftime("%Y-%m-%dT%H")
                    start_idx = next(
                        (
                            i
                            for i, t in enumerate(times)
                            if str(t).startswith(now_local)
                        ),
                        0,
                    )
                except Exception:  # noqa: BLE001
                    start_idx = 0

            end_idx = min(start_idx + hours, len(times))
            forecast: list[dict[str, Any]] = []
            for idx in range(start_idx, end_idx):
                code_f = int(codes[idx]) if idx < len(codes) else 0
                forecast.append(
                    {
                        "time": times[idx][-5:] if len(times[idx]) >= 5 else times[idx],
                        "temperature": temps[idx] if idx < len(temps) else None,
                        "conditions": WEATHER_CODE_MAP.get(code_f, "unknown"),
                        "precipitation_probability": (
                            precip[idx] if idx < len(precip) else None
                        ),
                        "wind_speed": winds[idx] if idx < len(winds) else None,
                    }
                )
            result["forecast"] = forecast
            result["forecast_hours"] = len(forecast)

        return result

    async def _geocode(self, location: str) -> tuple[float | None, float | None]:
        if location.strip().lower() == self._default_location.strip().lower():
            return self._default_lat, self._default_lon
        assert self._client is not None
        try:
            resp = await self._client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1, "language": "en"},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.error("weather.geocode_failed", error=str(exc))
            return None, None

        results = resp.json().get("results") or []
        if not results:
            return None, None
        first = results[0]
        return float(first["latitude"]), float(first["longitude"])
