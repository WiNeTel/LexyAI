"""
Lexy AI - PluginLoader.

Discovery → topo-sort → load → enable → reverse-order shutdown.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lexy_core.plugin_system.base_plugin import BasePlugin
from lexy_core.plugin_system.plugin_api import PluginAPI
from lexy_core.plugin_system.plugin_manifest import PluginManifest
from lexy_core.utils.logging import get_logger

if TYPE_CHECKING:
    from lexy_core.app import LexyApp

log = get_logger(module="plugin_loader")


class PluginLoader:
    """
    Discovers, loads, enables, and shuts down plugins.

    Workflow
    --------
    1. ``discover_and_load()``  – scan ``plugins_path`` for ``plugin.yaml`` files,
       topo-sort by ``requires``, then for each plugin: import → instantiate →
       ``on_load()`` → ``on_enable()``.
    2. ``disable_plugin(name)`` – call ``on_disable()`` then ``PluginAPI.cleanup()``.
    3. ``unload_all()``         – disable in reverse load order.
    """

    def __init__(self, plugins_path: Path, app: "LexyApp") -> None:
        self._plugins_path = plugins_path
        self._app = app
        self._manifests: dict[str, PluginManifest] = {}
        self._plugins: dict[str, BasePlugin] = {}
        self._apis: dict[str, PluginAPI] = {}
        self._load_order: list[str] = []

    @property
    def loaded_count(self) -> int:
        return len(self._plugins)

    def is_loaded(self, name: str) -> bool:
        return name in self._plugins

    def get_plugin(self, name: str) -> BasePlugin | None:
        return self._plugins.get(name)

    def get_all_plugins(self) -> dict[str, BasePlugin]:
        return dict(self._plugins)

    def get_manifests(self) -> dict[str, PluginManifest]:
        return dict(self._manifests)

    # ─── Discovery + Lifecycle ───────────────────────────────────────

    async def discover_and_load(self) -> None:
        """Discover, sort, load, and enable every plugin."""
        if not self._plugins_path.exists():
            log.warning("plugin_loader.path_missing", path=str(self._plugins_path))
            return

        self._discover()
        if not self._manifests:
            log.info("plugin_loader.no_plugins")
            return

        self._load_order = self._topological_sort()

        enabled_list = self._app.config.plugins.enabled
        disabled_list = self._app.config.plugins.disabled

        if enabled_list:
            plugins_to_load = [
                name for name in self._load_order
                if name in enabled_list and name not in disabled_list
            ]
        else:
            plugins_to_load = [
                name for name in self._load_order if name not in disabled_list
            ]

        # Profile-based exclusions (e.g. "chat" profile drops voice_gemma4)
        plugins_to_load = [
            name for name in plugins_to_load
            if not self._app.config.profile_excludes_plugin(name)
        ]
        for name in self._load_order:
            if self._app.config.profile_excludes_plugin(name):
                log.info(
                    "plugin_loader.skipped_by_profile",
                    plugin=name,
                    profile=self._app.config.system.profile,
                )

        for name in plugins_to_load:
            try:
                await self._load_plugin(name)
                await self._enable_plugin(name)
            except Exception as exc:  # noqa: BLE001 — keep loading the rest
                log.error("plugin_loader.failed", plugin=name, error=str(exc))

    def _discover(self) -> None:
        """Scan ``plugins/`` for ``plugin.yaml`` files."""
        for plugin_dir in self._plugins_path.iterdir():
            if not plugin_dir.is_dir():
                continue
            manifest_path = plugin_dir / "plugin.yaml"
            if not manifest_path.exists():
                continue
            try:
                manifest = PluginManifest.from_yaml(manifest_path)
                self._manifests[manifest.name] = manifest
                log.info(
                    "plugin_loader.discovered",
                    plugin=manifest.name,
                    version=manifest.version,
                )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "plugin_loader.manifest_error",
                    path=str(manifest_path),
                    error=str(exc),
                )

    def _topological_sort(self) -> list[str]:
        """Kahn's algorithm with deterministic ordering."""
        in_degree: dict[str, int] = {name: 0 for name in self._manifests}
        graph: dict[str, list[str]] = {name: [] for name in self._manifests}

        for name, manifest in self._manifests.items():
            for required in manifest.requires:
                if required in self._manifests:
                    graph[required].append(name)
                    in_degree[name] += 1

        queue = sorted([name for name, degree in in_degree.items() if degree == 0])
        result: list[str] = []

        while queue:
            queue.sort()  # deterministic order
            node = queue.pop(0)
            result.append(node)
            for dependent in graph[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._manifests):
            missing = set(self._manifests) - set(result)
            log.warning("plugin_loader.cyclic_dependencies", plugins=sorted(missing))
            result.extend(sorted(missing))

        return result

    async def _load_plugin(self, name: str) -> None:
        """Load one plugin: import, instantiate, ``on_load``."""
        manifest = self._manifests[name]

        for required in manifest.requires:
            if required not in self._plugins:
                raise RuntimeError(
                    f"Plugin '{name}' requires '{required}' which is not loaded"
                )

        plugin_dir = manifest.path
        if plugin_dir is None:
            raise RuntimeError(f"Plugin '{name}' has no path")

        # Add the *parent* of the plugin directory (i.e. ``plugins/``) so
        # that the plugin folder is treated as a proper Python package.
        # This allows relative imports (``from .foo import Bar``) inside
        # plugins to work correctly.
        parent_dir = str(Path(plugin_dir).parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        if "." not in manifest.entry:
            raise ValueError(
                f"Invalid entry '{manifest.entry}' (expected 'module.ClassName')"
            )
        module_name, class_name = manifest.entry.rsplit(".", 1)

        # Import as ``<plugin_folder>.<module>`` so the folder's
        # ``__init__.py`` is loaded first and relative imports resolve.
        package_module = f"{name}.{module_name}"
        module = importlib.import_module(package_module)
        plugin_class: type[Any] = getattr(module, class_name)

        if not issubclass(plugin_class, BasePlugin):
            raise TypeError(
                f"Class '{class_name}' in plugin '{name}' must subclass BasePlugin"
            )

        api = PluginAPI(name, self._app, manifest)
        self._apis[name] = api

        plugin = plugin_class(api, manifest)
        self._plugins[name] = plugin

        await plugin.on_load()
        log.info("plugin_loader.loaded", plugin=name)

    async def _enable_plugin(self, name: str) -> None:
        plugin = self._plugins.get(name)
        if plugin is None or plugin.enabled:
            return
        await plugin.on_enable()
        plugin._enabled = True
        log.info("plugin_loader.enabled", plugin=name)

    async def disable_plugin(self, name: str) -> None:
        plugin = self._plugins.get(name)
        if plugin is None or not plugin.enabled:
            return

        try:
            await plugin.on_disable()
        except Exception as exc:  # noqa: BLE001
            log.error("plugin_loader.disable_error", plugin=name, error=str(exc))

        plugin._enabled = False

        api = self._apis.get(name)
        if api is not None:
            await api.cleanup()

        log.info("plugin_loader.disabled", plugin=name)

    async def unload_plugin(self, name: str) -> None:
        await self.disable_plugin(name)
        self._plugins.pop(name, None)
        self._apis.pop(name, None)

    async def load_plugin(self, name: str) -> bool:
        """Discover + load + enable a single plugin (Phase 11 hot-load).

        Re-scans the plugins directory for the named plugin's manifest.
        Use this for plugins that just appeared on disk (e.g. published
        from ``workspace/extensions/`` by the coder plugin). Returns
        True on success, False if the plugin is already loaded or no
        matching manifest was found.
        """
        if name in self._plugins:
            return False
        # Re-scan only the named directory.
        target = self._plugins_path / name
        manifest_path = target / "plugin.yaml"
        if not manifest_path.exists():
            log.warning(
                "plugin_loader.load_missing_manifest",
                plugin=name, path=str(manifest_path),
            )
            return False
        try:
            manifest = PluginManifest.from_yaml(manifest_path)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "plugin_loader.load_manifest_error",
                plugin=name, error=str(exc),
            )
            return False
        if manifest.name != name:
            log.warning(
                "plugin_loader.load_name_mismatch",
                expected=name, got=manifest.name,
            )
            return False
        self._manifests[manifest.name] = manifest
        if name not in self._load_order:
            self._load_order.append(name)
        try:
            await self._load_plugin(name)
            await self._enable_plugin(name)
        except Exception as exc:  # noqa: BLE001
            log.error("plugin_loader.load_failed", plugin=name, error=str(exc))
            return False
        return True

    async def reload_plugin(self, name: str) -> bool:
        """Disable + reload + re-enable a plugin (Phase 11 hot-reload).

        Re-imports the plugin's Python modules so source changes take
        effect without a full backend restart. The order is:

        1. ``disable_plugin`` — runs the plugin's ``on_disable`` and
           clears its API registrations (tools, hooks, WS handlers).
        2. Drop cached imports of every ``<plugin_name>.*`` module so
           the next ``importlib.import_module`` reads from disk.
        3. Reload manifest from disk (``plugin.yaml`` may have changed).
        4. ``_load_plugin`` + ``_enable_plugin`` again.

        Returns True on success. If anything fails the plugin is left
        in whatever in-between state we got to — the caller should log
        and consider a full restart.
        """
        import importlib
        if name not in self._plugins and name not in self._manifests:
            # Maybe never loaded — just try a fresh load.
            return await self.load_plugin(name)
        await self.unload_plugin(name)

        # Drop cached imports for this plugin's package + submodules so
        # the next import_module re-reads from disk. Without this,
        # importlib.import_module returns the cached (old) module object.
        prefix = f"{name}."
        cached = [
            mod_name for mod_name in list(sys.modules)
            if mod_name == name or mod_name.startswith(prefix)
        ]
        for mod_name in cached:
            sys.modules.pop(mod_name, None)
        # Wipe __pycache__/ inside the plugin's directory so Python
        # can't serve a stale .pyc whose embedded source-mtime collides
        # with the freshly-edited .py (Windows mtime resolution is coarse).
        plugin_dir = self._plugins_path / name
        for cache_dir in plugin_dir.rglob("__pycache__"):
            try:
                import shutil
                shutil.rmtree(cache_dir, ignore_errors=True)
            except Exception:  # noqa: BLE001
                pass
        # Force the import system to re-stat directories.
        importlib.invalidate_caches()

        # Re-read manifest in case plugin.yaml changed.
        manifest_path = self._plugins_path / name / "plugin.yaml"
        if not manifest_path.exists():
            log.warning(
                "plugin_loader.reload_missing_manifest",
                plugin=name, path=str(manifest_path),
            )
            return False
        try:
            manifest = PluginManifest.from_yaml(manifest_path)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "plugin_loader.reload_manifest_error",
                plugin=name, error=str(exc),
            )
            return False
        self._manifests[manifest.name] = manifest
        if name not in self._load_order:
            self._load_order.append(name)
        try:
            await self._load_plugin(name)
            await self._enable_plugin(name)
        except Exception as exc:  # noqa: BLE001
            log.error("plugin_loader.reload_failed", plugin=name, error=str(exc))
            return False
        log.info("plugin_loader.reloaded", plugin=name)
        return True

    async def unload_all(self) -> None:
        """Reverse-order shutdown of all plugins."""
        for name in reversed(self._load_order):
            if name in self._plugins:
                try:
                    await self.unload_plugin(name)
                except Exception as exc:  # noqa: BLE001
                    log.error("plugin_loader.unload_error", plugin=name, error=str(exc))

    # ─── Introspection ───────────────────────────────────────────────

    def get_plugin_info(self) -> list[dict[str, Any]]:
        info: list[dict[str, Any]] = []
        for name, manifest in self._manifests.items():
            plugin = self._plugins.get(name)
            info.append(
                {
                    "name": name,
                    "version": manifest.version,
                    "description": manifest.description,
                    "loaded": name in self._plugins,
                    "enabled": plugin.enabled if plugin else False,
                    "requires": manifest.requires,
                    "frontend": [
                        {
                            "id": fm.id,
                            "mount": fm.mount,
                            "label": fm.label,
                        }
                        for fm in manifest.frontend
                    ],
                }
            )
        return info
