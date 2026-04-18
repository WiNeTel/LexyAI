"""Lexy AI – Plugin System."""

from lexy_core.plugin_system.base_plugin import BasePlugin
from lexy_core.plugin_system.plugin_api import PluginAPI
from lexy_core.plugin_system.plugin_loader import PluginLoader
from lexy_core.plugin_system.plugin_manifest import FrontendModule, PluginManifest

__all__ = [
    "BasePlugin",
    "FrontendModule",
    "PluginAPI",
    "PluginLoader",
    "PluginManifest",
]
