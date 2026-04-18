"""Smoke tests for plugin manifest loading (scans plugins/)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lexy_core.plugin_system import PluginManifest


PLUGINS_DIR = Path(__file__).resolve().parent.parent / "plugins"


@pytest.mark.parametrize(
    "plugin_dir",
    [d for d in sorted(PLUGINS_DIR.iterdir()) if (d / "plugin.yaml").exists()],
    ids=lambda d: d.name,
)
def test_every_plugin_has_valid_manifest(plugin_dir: Path) -> None:
    manifest = PluginManifest.from_yaml(plugin_dir / "plugin.yaml")
    assert manifest.name == plugin_dir.name
    assert manifest.entry  # must have entry module.ClassName
    assert "." in manifest.entry
    assert manifest.version
