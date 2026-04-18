"""
Lexy AI - Plugin Manifest Parser.

Parses plugin.yaml files into typed PluginManifest objects.

Example plugin.yaml:
    name: weather
    version: 1.0.0
    description: "Weather via Open-Meteo"
    entry: weather_plugin.WeatherPlugin
    requires: []
    optional: [scheduler]
    capabilities: [tool, event]

    config_defaults:
      location: "Berlin"

    frontend:
      - id: weather-widget
        js: weather.js
        css: weather.css
        mount: sidebar
        icon: cloud
        label: "Weather"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lexy_core.utils.logging import get_logger

log = get_logger(module="plugin_manifest")


@dataclass
class FrontendModule:
    """Frontend module declared in a plugin manifest."""

    id: str
    js: str
    css: str = ""
    mount: str = "sidebar"  # sidebar | main | overlay
    icon: str = ""
    label: str = ""


@dataclass
class PluginManifest:
    """Parsed plugin manifest."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    entry: str = ""  # "module.ClassName"
    requires: list[str] = field(default_factory=list)
    optional: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    frontend: list[FrontendModule] = field(default_factory=list)
    config_defaults: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "PluginManifest":
        """
        Load a manifest from disk.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the manifest is missing the required ``name`` field.
        """
        if not yaml_path.exists():
            raise FileNotFoundError(f"Plugin manifest not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        if "name" not in data:
            raise ValueError(f"Plugin manifest missing 'name' field: {yaml_path}")

        frontend_modules: list[FrontendModule] = []
        for fm_data in data.get("frontend", []) or []:
            frontend_modules.append(
                FrontendModule(
                    id=fm_data.get("id", ""),
                    js=fm_data.get("js", ""),
                    css=fm_data.get("css", ""),
                    mount=fm_data.get("mount", "sidebar"),
                    icon=fm_data.get("icon", ""),
                    label=fm_data.get("label", ""),
                )
            )

        manifest = cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            entry=data.get("entry", ""),
            requires=list(data.get("requires", []) or []),
            optional=list(data.get("optional", []) or []),
            capabilities=list(data.get("capabilities", []) or []),
            frontend=frontend_modules,
            config_defaults=dict(data.get("config_defaults", {}) or {}),
            path=yaml_path.parent,
        )
        log.debug("plugin_manifest.parsed", name=manifest.name, version=manifest.version)
        return manifest
