"""
Lexy AI - BasePlugin ABC.

Every plugin extends this class and implements the three lifecycle hooks:

* ``on_load``    – initialise resources (DBs, models, config).
* ``on_enable``  – register handlers (events, hooks, tools, WS).
* ``on_disable`` – release private resources (PluginAPI handles registration cleanup).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lexy_core.plugin_system.plugin_api import PluginAPI
    from lexy_core.plugin_system.plugin_manifest import PluginManifest


class BasePlugin(ABC):
    """
    Abstract base class for Lexy plugins.

    Lifecycle
    ---------
    1. ``__init__(api, manifest)`` – plugin instantiated.
    2. ``on_load()``               – DB connections, models, config.
    3. ``on_enable()``             – register handlers / hooks / tools / WS.
    4. ``on_disable()``            – plugin-owned cleanup; PluginAPI removes registrations.
    """

    def __init__(self, api: "PluginAPI", manifest: "PluginManifest") -> None:
        self.api = api
        self.manifest = manifest
        self.name: str = manifest.name
        self._enabled: bool = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @abstractmethod
    async def on_load(self) -> None:
        """Initialise plugin resources (DB, models, config)."""

    @abstractmethod
    async def on_enable(self) -> None:
        """Register handlers via the PluginAPI facade."""

    @abstractmethod
    async def on_disable(self) -> None:
        """Release plugin-owned resources. PluginAPI cleans up registrations."""

    async def on_config_changed(self, cfg: dict) -> None:  # noqa: D401 — optional
        """Called when the plugin's config is patched at runtime.

        Default implementation is a no-op; plugins that hold derived state
        (intervals, rate limits, whitelists, …) should override this to apply
        the new config live without restart. Receives the fully merged config
        dict (defaults + overrides + patch).
        """
        return None
