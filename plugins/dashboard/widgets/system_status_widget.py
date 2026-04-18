"""
Lexy AI - Dashboard System Status Widget.

Checks the health of all local services (LLM brains, ChromaDB, SearXNG,
CosyVoice) via quick HTTP pings, and reports loaded plugin count + uptime.

Each entry's ``paths`` is a list of candidate health-check endpoints tried
in order — the first one returning a 2xx/3xx marks the service as up. This
lets us stay robust across service-version API changes (e.g. ChromaDB
dropped ``/api/v1/heartbeat`` → 410 Gone in ChromaDB 1.0 in favour of
``/api/v2/heartbeat``).
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from lexy_core.utils.logging import get_logger

from .base_widget import BaseWidget

log = get_logger(module="widget.system_status")


# Services to monitor. ``paths`` is a list of candidate health-check
# endpoints — each one tried in order until one answers with 2xx/3xx.
_SERVICES: list[dict[str, Any]] = [
    {
        "name": "E4B Brain",
        "host": "127.0.0.1",
        "port": "5005",
        "paths": ["/health"],
    },
    {
        "name": "A4B Brain",
        "host": "127.0.0.1",
        "port": "5006",
        "paths": ["/health"],
    },
    {
        "name": "ChromaDB",
        "host": "127.0.0.1",
        "port": "8000",
        # ChromaDB 1.0+ answers on v2; older 0.4.x still on v1. Try both
        # and fall back to a bare-root probe as last resort.
        "paths": ["/api/v2/heartbeat", "/api/v1/heartbeat", "/"],
    },
    {
        "name": "SearXNG",
        "host": "127.0.0.1",
        "port": "7899",
        "paths": ["/"],
    },
    {
        "name": "CosyVoice",
        "host": "172.20.0.245",
        "port": "5500",
        "paths": ["/"],
    },
]


class SystemStatusWidget(BaseWidget):
    """Service health, loaded plugins, uptime."""

    widget_id: str = "system_status"
    title: str = "Systemstatus"
    default_size: tuple[int, int] = (3, 2)
    refresh_interval: float = 30.0

    def __init__(self, api: Any) -> None:
        super().__init__(api)
        self._start_time: float = time.monotonic()

    async def get_data(self) -> dict[str, Any]:
        """Ping every service and collect plugin / uptime info."""
        services: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=2.0) as client:
            for svc in _SERVICES:
                entry = await self._probe_service(client, svc)
                services.append(entry)

        # Plugin count
        plugins_loaded: int = 0
        if self._api._app.plugin_loader is not None:
            plugins_loaded = self._api._app.plugin_loader.loaded_count

        uptime_seconds = time.monotonic() - self._start_time

        return {
            "services": services,
            "plugins_loaded": plugins_loaded,
            "uptime_seconds": round(uptime_seconds, 1),
        }

    async def _probe_service(
        self,
        client: httpx.AsyncClient,
        svc: dict[str, Any],
    ) -> dict[str, Any]:
        """Try each candidate path until one answers with 2xx/3xx.

        The very first path is also what gets blamed in the error / status
        code field so users see the "primary" endpoint that failed.
        """
        entry: dict[str, Any] = {
            "name": svc["name"],
            "host": f"{svc['host']}:{svc['port']}",
            "status": "down",
        }

        # Accept both legacy 'path' (str) and new 'paths' (list) shapes.
        raw_paths = svc.get("paths") or svc.get("path") or ["/"]
        if isinstance(raw_paths, str):
            paths: list[str] = [raw_paths]
        else:
            paths = list(raw_paths)

        last_error: dict[str, Any] = {}
        for path in paths:
            url = f"http://{svc['host']}:{svc['port']}{path}"
            try:
                resp = await client.get(url)
            except httpx.TimeoutException:
                last_error = {"status": "timeout", "path": path}
                continue
            except httpx.ConnectError:
                # No listener at all on this host:port — no point trying
                # further paths.
                return {**entry, "status": "down"}
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "widget.system_status.ping_error",
                    service=svc["name"],
                    path=path,
                    error=str(exc),
                )
                last_error = {"status": "down", "path": path}
                continue

            if resp.status_code < 400:
                entry["status"] = "up"
                entry["path"] = path
                return entry
            # 4xx/5xx — note it but try the next path; a service may
            # have migrated its API (e.g. ChromaDB v1 → v2 → 410 Gone).
            last_error = {
                "status": "error",
                "status_code": resp.status_code,
                "path": path,
            }

        # Nothing answered 2xx/3xx — return the last recorded failure
        # (or the default "down" if we never got that far).
        entry.update(last_error)
        return entry
