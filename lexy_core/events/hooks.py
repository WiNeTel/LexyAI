"""
Lexy AI - HookManager.

Priority-based hooks at well-defined extension points. Three execution modes:

* ``execute_void``      – fire-and-forget, parallel.
* ``execute_modifying`` – sequential pipeline, callbacks may return a modified context.
* ``execute_sync``      – synchronous (hot paths only).

Hook names are plain strings (no enums) so plugins can introduce new ones.

Example:
    hooks = HookManager()
    hooks.register("before_prompt_build", inject_persona, priority=20, source="character")
    ctx = await hooks.execute_modifying("before_prompt_build", {"messages": [...]})
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from lexy_core.utils.logging import get_logger

log = get_logger(module="hooks")

#: A hook callback. Modifying hooks should return a dict (or None for "no change").
HookCallback = Callable[[dict[str, Any]], Any | Awaitable[Any]]


@dataclass
class HookRegistration:
    """A single registered hook."""

    hook_name: str
    callback: HookCallback
    priority: int = 50  # 0 = first, 100 = last
    name: str = ""
    source: str = ""


class HookManager:
    """
    Priority-based pipeline for extension points.

    Hook types
    ----------
    * ``execute_void``      – parallel fire-and-forget.
    * ``execute_modifying`` – sequential pipeline (context dict in/out).
    * ``execute_sync``      – synchronous, skips async callbacks.
    """

    def __init__(self) -> None:
        self._hooks: dict[str, list[HookRegistration]] = defaultdict(list)

    # ─── Registration ────────────────────────────────────────────────

    def register(
        self,
        hook_name: str,
        callback: HookCallback,
        priority: int = 50,
        name: str = "",
        source: str = "",
    ) -> None:
        """Register a hook callback. Lower priority runs first."""
        registration = HookRegistration(
            hook_name=hook_name,
            callback=callback,
            priority=priority,
            name=name or getattr(callback, "__name__", "anonymous"),
            source=source,
        )
        self._hooks[hook_name].append(registration)
        self._hooks[hook_name].sort(key=lambda h: h.priority)
        log.debug(
            "hooks.registered",
            hook=hook_name,
            source=source,
            priority=priority,
        )

    def unregister(
        self,
        hook_name: str,
        callback: HookCallback | None = None,
        source: str | None = None,
    ) -> int:
        """Remove hooks matching callback and/or source. Returns count removed."""
        if hook_name not in self._hooks:
            return 0

        before = len(self._hooks[hook_name])
        self._hooks[hook_name] = [
            registration
            for registration in self._hooks[hook_name]
            if not (
                (callback is None or registration.callback is callback)
                and (source is None or registration.source == source)
            )
        ]
        return before - len(self._hooks[hook_name])

    def unregister_all(self, source: str) -> int:
        """Remove every hook owned by a given source."""
        total = 0
        for hook_name in list(self._hooks.keys()):
            before = len(self._hooks[hook_name])
            self._hooks[hook_name] = [
                registration
                for registration in self._hooks[hook_name]
                if registration.source != source
            ]
            total += before - len(self._hooks[hook_name])
        if total:
            log.debug("hooks.cleaned", source=source, removed=total)
        return total

    # ─── Execution ───────────────────────────────────────────────────

    async def execute_void(
        self,
        hook_name: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Run all callbacks in parallel; errors are logged but never raised."""
        registrations = list(self._hooks.get(hook_name, []))
        if not registrations:
            return

        ctx = dict(context) if context else {}

        async def _run(registration: HookRegistration) -> None:
            try:
                if asyncio.iscoroutinefunction(registration.callback):
                    await registration.callback(ctx)
                else:
                    registration.callback(ctx)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "hooks.void_error",
                    hook=hook_name,
                    name=registration.name,
                    source=registration.source,
                    error=str(exc),
                )

        await asyncio.gather(*[_run(reg) for reg in registrations], return_exceptions=True)

    async def execute_modifying(
        self,
        hook_name: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run callbacks sequentially. Each callback may return a modified context
        (or ``None`` to leave it unchanged).
        """
        ctx = dict(context) if context else {}
        registrations = list(self._hooks.get(hook_name, []))

        for registration in registrations:
            try:
                if asyncio.iscoroutinefunction(registration.callback):
                    result = await registration.callback(ctx)
                else:
                    result = registration.callback(ctx)
                if isinstance(result, dict):
                    ctx = result
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "hooks.modifying_error",
                    hook=hook_name,
                    name=registration.name,
                    source=registration.source,
                    error=str(exc),
                )

        return ctx

    def execute_sync(
        self,
        hook_name: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Synchronous execution for hot paths. Async callbacks are skipped with a warning.
        """
        ctx = dict(context) if context else {}
        registrations = list(self._hooks.get(hook_name, []))

        for registration in registrations:
            if asyncio.iscoroutinefunction(registration.callback):
                log.warning(
                    "hooks.sync_skipped_async",
                    hook=hook_name,
                    name=registration.name,
                )
                continue
            try:
                result = registration.callback(ctx)
                if isinstance(result, dict):
                    ctx = result
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "hooks.sync_error",
                    hook=hook_name,
                    name=registration.name,
                    error=str(exc),
                )

        return ctx

    # ─── Introspection ───────────────────────────────────────────────

    def has_hooks(self, hook_name: str) -> bool:
        return bool(self._hooks.get(hook_name))

    def get_hooks(self, hook_name: str) -> list[HookRegistration]:
        return list(self._hooks.get(hook_name, []))

    def get_all_hook_names(self) -> list[str]:
        return [name for name, registrations in self._hooks.items() if registrations]
