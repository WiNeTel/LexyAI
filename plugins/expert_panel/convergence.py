"""
Lexy AI - Expert Panel Convergence Detector (adapter).

Phase A2: the convergence logic now lives in the shared coordination
kernel (``lexy_core.coordination.ConvergenceDetector``) so the expert
panel, the orchestrator's council mode, and future deliberation features
all share one implementation.

This module is a thin **backward-compatible adapter**: it keeps the exact
public surface the panel + its tests rely on —
``check(messages, roles, threshold, api, brain)`` returning a plain dict
with ``agreeing_roles`` keys — while delegating the real work to the
kernel. Nothing in ``panel_plugin.py`` had to change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lexy_core.coordination import ConvergenceDetector as _KernelDetector
from lexy_core.utils.logging import get_logger

if TYPE_CHECKING:
    from lexy_core.plugin_system.plugin_api import PluginAPI

log = get_logger(module="expert_panel.convergence")


class ConvergenceDetector:
    """Adapter over :class:`lexy_core.coordination.ConvergenceDetector`.

    Preserves the panel's original dict-based API (``agreeing_roles``)
    while the implementation is shared via the coordination kernel.
    """

    def __init__(self) -> None:
        self._kernel = _KernelDetector()

    async def check(
        self,
        messages: list[dict[str, Any]],
        roles: list[str],
        threshold: int,
        api: "PluginAPI",
        brain: str = "e4b",
    ) -> dict[str, Any]:
        """Analyse discussion messages and count agreements.

        Returns the panel's historical shape:
        ``{converged: bool, agreements: [{point, agreeing_roles}], agreement_count: int}``.
        """
        result = await self._kernel.check(
            contributions=messages,
            participants=roles,
            threshold=threshold,
            llm_chat=api.llm_chat,
            brain=brain,
        )
        return {
            "converged": result.converged,
            "agreements": [
                {"point": a["point"], "agreeing_roles": a["agreeing"]}
                for a in result.agreements
            ],
            "agreement_count": result.agreement_count,
        }

    @staticmethod
    def _parse_agreements(
        raw: str, valid_roles: list[str]
    ) -> list[dict[str, Any]]:
        """Parse structured JSON agreements (delegates to the kernel).

        Kept for backwards-compat: maps the kernel's ``agreeing`` key back
        to the panel's historical ``agreeing_roles``.
        """
        parsed = _KernelDetector._parse_agreements(raw, valid_roles)
        return [
            {"point": a["point"], "agreeing_roles": a["agreeing"]} for a in parsed
        ]
