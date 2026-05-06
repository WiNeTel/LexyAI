"""Tests for Phase 11 hot-reload: PluginLoader load/reload + coder publish/unpublish.

Two layers:

* :class:`TestLoaderLoadReload` — exercises the primitives on a real
  :class:`PluginLoader` with a tiny stub plugin written to a
  ``tmp_path``. We minimise the LexyApp surface to just what the loader
  actually touches (config, hooks, etc.) so the test stays self-contained.

* :class:`TestPublishExtension` — exercises ``coder_publish_extension``
  end-to-end against a fake CoderPlugin that just executes the publish
  copy-and-reload code path. We don't boot the full LexyApp; we stub the
  bits the tool reaches into (``_app``, ``ws_broadcast``, etc.).
"""

from __future__ import annotations

import asyncio
import importlib
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from lexy_core.plugin_system import BasePlugin, PluginLoader


# ─── Fake LexyApp + Plugin scaffolding ──────────────────────────────


class _FakeConfigPlugins:
    """Mirrors the shape of LexyConfig.plugins the loader reads."""

    def __init__(self) -> None:
        self.enabled: list[str] = []
        self.disabled: list[str] = []


class _FakeRoutingConfig:
    default_brain = "e4b"
    rules: list[Any] = []


class _FakeSystem:
    profile = "chat"


class _FakeConfig:
    """Minimal config — only what the plugin loader inspects."""

    def __init__(self) -> None:
        self.plugins = _FakeConfigPlugins()
        self.routing = _FakeRoutingConfig()
        self.system = _FakeSystem()

    def profile_excludes_plugin(self, name: str) -> bool:
        return False


class _FakeHooks:
    async def execute_modifying(self, name, ctx):
        return ctx
    async def execute_void(self, *args, **kwargs):
        return None
    def register(self, *args, **kwargs):
        pass
    def unregister_all(self, *args, **kwargs):
        pass


class _FakeEventBus:
    async def emit(self, *args, **kwargs):
        return None
    def on(self, *args, **kwargs):
        pass
    def off(self, *args, **kwargs):
        pass
    def off_all(self, *args, **kwargs):
        pass


class _FakeApp:
    """Just enough of LexyApp to keep PluginAPI happy."""

    def __init__(self) -> None:
        self.config = _FakeConfig()
        self.hooks = _FakeHooks()
        self.event_bus = _FakeEventBus()
        self.tool_registry = None
        self.tool_caller = None
        self.ws_server = None
        self.signals = None
        self.session_store = None
        self.memory = None
        self.plugin_overrides: dict[str, dict[str, Any]] = {}
        self.plugin_loader: PluginLoader | None = None
        self.voice = None
        self.channel_router = None
        self.llm = None
        self.agent = None


# ─── Stub plugin we write to disk and (re)load ──────────────────────


_STUB_PLUGIN_YAML = """\
name: {name}
version: {version}
description: "Phase-11 hot-reload test stub"
entry: plugin.{class_name}
requires: []
optional: []
capabilities: [tool]
"""

_STUB_PLUGIN_PY_V1 = '''\
from lexy_core.plugin_system import BasePlugin


class {class_name}(BasePlugin):
    VERSION_TAG = "v1"

    async def on_load(self):
        pass

    async def on_enable(self):
        pass

    async def on_disable(self):
        pass
'''

_STUB_PLUGIN_PY_V2 = '''\
from lexy_core.plugin_system import BasePlugin


class {class_name}(BasePlugin):
    VERSION_TAG = "v2"

    async def on_load(self):
        pass

    async def on_enable(self):
        pass

    async def on_disable(self):
        pass
'''


def _write_stub(
    plugins_root: Path,
    name: str,
    *,
    class_name: str = "StubPlugin",
    version: str = "0.1.0",
    body: str | None = None,
) -> None:
    plugin_dir = plugins_root / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        _STUB_PLUGIN_YAML.format(
            name=name, version=version, class_name=class_name,
        ),
        encoding="utf-8",
    )
    (plugin_dir / "plugin.py").write_text(
        body or _STUB_PLUGIN_PY_V1.format(class_name=class_name),
        encoding="utf-8",
    )


# ─── Fixtures ───────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def loader_setup(tmp_path: Path):
    plugins_root = tmp_path / "plugins"
    plugins_root.mkdir()
    app = _FakeApp()
    loader = PluginLoader(plugins_path=plugins_root, app=app)
    app.plugin_loader = loader
    yield app, loader, plugins_root
    await loader.unload_all()
    # Drop any cached imports we created.
    for mod_name in list(sys.modules):
        if mod_name in {"hotone", "hottwo"} or mod_name.startswith(
            ("hotone.", "hottwo.")
        ):
            sys.modules.pop(mod_name, None)


# ─── Loader primitives ─────────────────────────────────────────────


class TestLoaderLoadReload:
    @pytest.mark.asyncio
    async def test_load_plugin_first_time(
        self, loader_setup: tuple[_FakeApp, PluginLoader, Path]
    ) -> None:
        _app, loader, plugins_root = loader_setup
        _write_stub(plugins_root, "hotone")
        ok = await loader.load_plugin("hotone")
        assert ok is True
        assert loader.is_loaded("hotone")
        plugin = loader.get_plugin("hotone")
        assert plugin is not None
        assert plugin.__class__.VERSION_TAG == "v1"

    @pytest.mark.asyncio
    async def test_load_plugin_already_loaded_returns_false(
        self, loader_setup: tuple[_FakeApp, PluginLoader, Path]
    ) -> None:
        _app, loader, plugins_root = loader_setup
        _write_stub(plugins_root, "hotone")
        assert await loader.load_plugin("hotone") is True
        # Second call must be a no-op (False).
        assert await loader.load_plugin("hotone") is False

    @pytest.mark.asyncio
    async def test_load_plugin_missing_manifest(
        self, loader_setup: tuple[_FakeApp, PluginLoader, Path]
    ) -> None:
        _app, loader, plugins_root = loader_setup
        # Directory exists but no plugin.yaml inside.
        (plugins_root / "ghost").mkdir()
        ok = await loader.load_plugin("ghost")
        assert ok is False

    @pytest.mark.asyncio
    async def test_load_plugin_name_mismatch_rejected(
        self, loader_setup: tuple[_FakeApp, PluginLoader, Path]
    ) -> None:
        _app, loader, plugins_root = loader_setup
        # Directory says one thing, manifest says another.
        _write_stub(plugins_root, "hotone")
        # Patch the manifest to claim a different name.
        (plugins_root / "hotone" / "plugin.yaml").write_text(
            _STUB_PLUGIN_YAML.format(
                name="not_hotone", version="0.1.0", class_name="StubPlugin",
            ),
            encoding="utf-8",
        )
        ok = await loader.load_plugin("hotone")
        assert ok is False
        assert not loader.is_loaded("hotone")
        assert not loader.is_loaded("not_hotone")

    @pytest.mark.asyncio
    async def test_reload_picks_up_source_changes(
        self, loader_setup: tuple[_FakeApp, PluginLoader, Path]
    ) -> None:
        _app, loader, plugins_root = loader_setup
        _write_stub(plugins_root, "hotone")
        await loader.load_plugin("hotone")
        first = loader.get_plugin("hotone")
        assert first is not None and first.__class__.VERSION_TAG == "v1"

        # Now bump the source on disk and reload.
        (plugins_root / "hotone" / "plugin.py").write_text(
            _STUB_PLUGIN_PY_V2.format(class_name="StubPlugin"),
            encoding="utf-8",
        )
        ok = await loader.reload_plugin("hotone")
        assert ok is True
        second = loader.get_plugin("hotone")
        assert second is not None
        # New class object — picked up the source change.
        assert second.__class__.VERSION_TAG == "v2"
        assert second is not first

    @pytest.mark.asyncio
    async def test_reload_unknown_plugin_falls_through_to_load(
        self, loader_setup: tuple[_FakeApp, PluginLoader, Path]
    ) -> None:
        _app, loader, plugins_root = loader_setup
        _write_stub(plugins_root, "hotone")
        # We never called load_plugin first — reload_plugin must still cope.
        ok = await loader.reload_plugin("hotone")
        assert ok is True
        assert loader.is_loaded("hotone")

    @pytest.mark.asyncio
    async def test_reload_clears_cached_submodules(
        self, loader_setup: tuple[_FakeApp, PluginLoader, Path]
    ) -> None:
        _app, loader, plugins_root = loader_setup
        # Plugin with a submodule we can sniff for staleness.
        plugin_dir = plugins_root / "hottwo"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(
            _STUB_PLUGIN_YAML.format(
                name="hottwo", version="0.1.0", class_name="StubPlugin",
            ),
            encoding="utf-8",
        )
        (plugin_dir / "plugin.py").write_text(
            "from .helper import VAL\n"
            "from lexy_core.plugin_system import BasePlugin\n"
            "class StubPlugin(BasePlugin):\n"
            "    VAL = VAL\n"
            "    async def on_load(self): pass\n"
            "    async def on_enable(self): pass\n"
            "    async def on_disable(self): pass\n",
            encoding="utf-8",
        )
        (plugin_dir / "helper.py").write_text("VAL = 1\n", encoding="utf-8")

        await loader.load_plugin("hottwo")
        assert loader.get_plugin("hottwo").__class__.VAL == 1

        # Bump helper.py and reload.
        (plugin_dir / "helper.py").write_text("VAL = 99\n", encoding="utf-8")
        await loader.reload_plugin("hottwo")
        assert loader.get_plugin("hottwo").__class__.VAL == 99


# ─── coder_publish_extension flow ───────────────────────────────────


class TestPublishExtensionFlow:
    """Integration-level test — exercises the publish copy + reload via a
    real workspace + a real plugin_loader, with a stripped-down CoderPlugin
    instance bound to a fake app."""

    @pytest_asyncio.fixture
    async def published_setup(self, tmp_path: Path, monkeypatch):
        # We override Path("plugins") in plugin.py by chdir-ing into the
        # tmp_path. The plugin's publish handler uses Path("plugins")
        # relative to cwd, so this gives us isolation without modifying
        # the source.
        original_cwd = Path.cwd()
        plugins_root = tmp_path / "plugins"
        plugins_root.mkdir()
        workspace_root = tmp_path / "workspace"
        (workspace_root / "skills").mkdir(parents=True)
        (workspace_root / "projects").mkdir(parents=True)
        (workspace_root / "extensions").mkdir(parents=True)

        # Build the workspace_mgr and approval gate manually so we can
        # plug them straight into the CoderPlugin without a real plugin
        # loader bootstrapping the manifest.
        from plugins.coder.plugin import CoderPlugin  # noqa: F401
        from plugins.coder.workspace_mgr import WorkspaceManager
        from plugins.coder.approval_gate import ApprovalGate
        import aiosqlite

        ws = WorkspaceManager(workspace_root)
        await ws.ensure_layout()
        broadcasts: list[dict] = []
        async def bc(payload):
            broadcasts.append(payload)
        gate = ApprovalGate(broadcast=bc, auto_approve_low=False)
        approvals_db = await aiosqlite.connect(":memory:")
        await gate.init_db(approvals_db)

        # Pre-approve all coder publish/unpublish requests for this test.
        gate.grant_session(session_id="", action="coder_publish_extension")
        gate.grant_session(session_id="", action="coder_unpublish_extension")
        # NOTE: HIGH-risk ignores session grants, so we resolve manually
        # via the loop below instead.

        loader_app = _FakeApp()
        loader = PluginLoader(plugins_path=plugins_root, app=loader_app)
        loader_app.plugin_loader = loader

        # Auto-resolver: every approval request gets approved as soon as
        # it lands on the broadcast queue.
        async def auto_approve_loop():
            while True:
                await asyncio.sleep(0.005)
                # Walk pending and approve them all.
                for req_id in list(gate.list_pending()):
                    gate.resolve(
                        request_id=req_id, approved=True, reason="user",
                    )

        approver = asyncio.create_task(auto_approve_loop())

        try:
            monkeypatch.chdir(tmp_path)
            yield {
                "tmp_path": tmp_path,
                "plugins_root": plugins_root,
                "workspace_root": workspace_root,
                "ws": ws,
                "gate": gate,
                "loader": loader,
                "loader_app": loader_app,
                "broadcasts": broadcasts,
            }
        finally:
            approver.cancel()
            try:
                await approver
            except asyncio.CancelledError:
                pass
            await loader.unload_all()
            await approvals_db.close()
            # Drop cached imports we created.
            for mod_name in list(sys.modules):
                if mod_name == "myext" or mod_name.startswith("myext."):
                    sys.modules.pop(mod_name, None)

    @pytest.mark.asyncio
    async def test_publish_creates_plugin_dir_and_loads(
        self, published_setup
    ) -> None:
        ws = published_setup["ws"]
        loader = published_setup["loader"]
        plugins_root = published_setup["plugins_root"]
        workspace_root = published_setup["workspace_root"]

        # Lay down a workspace extension by hand (init_project's template
        # uses CamelCase from the name, which here would be "Myext").
        info = await ws.init_project(name="myext", kind="extension")
        # Sanity: the template wrote a class called "MyextPlugin".
        plugin_py = (info.root / "plugin.py").read_text(encoding="utf-8")
        assert "class MyextPlugin(BasePlugin)" in plugin_py

        # Build a stand-in CoderPlugin and call _tool_publish_extension
        # directly. We don't go through the plugin loader — we just need
        # the tool's logic to execute against the real workspace + the
        # real loader.
        coder_plugin = await self._build_coder_plugin(
            published_setup, workspace_root,
        )
        result = await coder_plugin._tool_publish_extension(name="myext")
        assert result["ok"] is True
        assert result["loaded"] is True
        assert (plugins_root / "myext" / "plugin.yaml").exists()
        assert (plugins_root / "myext" / "_published.flag").exists()
        assert loader.is_loaded("myext")

    @pytest.mark.asyncio
    async def test_publish_refuses_to_overwrite_unflagged_dir(
        self, published_setup
    ) -> None:
        ws = published_setup["ws"]
        plugins_root = published_setup["plugins_root"]
        workspace_root = published_setup["workspace_root"]

        # Pre-existing core-style plugin dir without the sentinel.
        core_dir = plugins_root / "myext"
        core_dir.mkdir()
        (core_dir / "plugin.yaml").write_text(
            "name: myext\nversion: 1.0.0\nentry: plugin.X\n",
            encoding="utf-8",
        )

        await ws.init_project(name="myext", kind="extension")
        coder_plugin = await self._build_coder_plugin(
            published_setup, workspace_root,
        )
        result = await coder_plugin._tool_publish_extension(name="myext")
        assert result["ok"] is False
        assert "_published.flag" in result["error"]

    @pytest.mark.asyncio
    async def test_publish_then_unpublish_roundtrip(
        self, published_setup
    ) -> None:
        ws = published_setup["ws"]
        loader = published_setup["loader"]
        plugins_root = published_setup["plugins_root"]
        workspace_root = published_setup["workspace_root"]

        await ws.init_project(name="myext", kind="extension")
        coder_plugin = await self._build_coder_plugin(
            published_setup, workspace_root,
        )

        pub = await coder_plugin._tool_publish_extension(name="myext")
        assert pub["ok"] is True

        unp = await coder_plugin._tool_unpublish_extension(name="myext")
        assert unp["ok"] is True
        assert not (plugins_root / "myext").exists()
        assert not loader.is_loaded("myext")
        # Workspace copy stays.
        assert (workspace_root / "extensions" / "myext" / "plugin.yaml").exists()

    @pytest.mark.asyncio
    async def test_publish_rejects_invalid_name(
        self, published_setup
    ) -> None:
        coder_plugin = await self._build_coder_plugin(
            published_setup, published_setup["workspace_root"],
        )
        result = await coder_plugin._tool_publish_extension(name="../../etc")
        assert result["ok"] is False
        assert "invalid project name" in result["error"]

    @pytest.mark.asyncio
    async def test_publish_rejects_missing_workspace_extension(
        self, published_setup
    ) -> None:
        coder_plugin = await self._build_coder_plugin(
            published_setup, published_setup["workspace_root"],
        )
        result = await coder_plugin._tool_publish_extension(name="not_there")
        assert result["ok"] is False
        assert "no plugin.yaml" in result["error"]

    # ── Test helper ──────────────────────────────────────────────

    @staticmethod
    async def _build_coder_plugin(setup, workspace_root):
        """Build a barely-functional CoderPlugin instance for tool tests.

        We don't run the full ``on_load`` lifecycle because that would
        require a PluginAPI whose ``_app`` exposes everything. Instead
        we instantiate the plugin class with stubs and inject the few
        members the tools we test actually touch.
        """
        from plugins.coder.plugin import CoderPlugin

        class _FakeAPI:
            def __init__(self, app):
                self._app = app
            async def ws_broadcast(self, payload):
                setup["broadcasts"].append(payload)

        plugin = CoderPlugin.__new__(CoderPlugin)
        plugin.api = _FakeAPI(setup["loader_app"])
        plugin._workspace = setup["ws"]
        plugin._gate = setup["gate"]
        # Workspace name validator is reachable via _ws() — already wired.
        return plugin
