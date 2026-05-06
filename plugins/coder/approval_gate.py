"""
Approval-Gate — synchronous "ask the user before doing this" workflow.

When a tool needs Mike's go-ahead before touching the filesystem or
spawning a subprocess, it calls :meth:`ApprovalGate.request`. The gate:

1. Generates a short ``request_id``.
2. Broadcasts ``coder_approval_request`` over the websocket.
3. Awaits the matching ``coder_approval_response`` (handled by the
   plugin's WS handler, which calls :meth:`ApprovalGate.resolve`).
4. Honours a timeout — defaults to 180 s, configurable per request —
   and auto-rejects if no response landed.

Auto-approve:

* :data:`AUTO_APPROVE_LOW` — list/read/git_log run instantly.
* Per-session opt-in: Mike can flip ``auto_approve`` on for a category;
  subsequent MED-risk requests are granted without UI roundtrips.
* HIGH-risk actions ignore auto-approve and always wait for explicit
  confirmation.

The gate exposes two persistence paths:

* In-memory pending requests (asyncio.Future-keyed) — used for the
  active wait.
* SQLite audit log (``approvals.db``) — every request + decision is
  recorded so Mike can review what Lexy did, even after the fact.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import aiosqlite


log = logging.getLogger(__name__)


# Risk levels — keep in sync with the docs.
RISK_LOW = "low"
RISK_MED = "med"
RISK_HIGH = "high"
KNOWN_RISKS: tuple[str, ...] = (RISK_LOW, RISK_MED, RISK_HIGH)


@dataclass
class ApprovalRequest:
    request_id: str
    action: str                        # e.g. "workspace_write"
    risk: str                          # one of KNOWN_RISKS
    payload: dict[str, Any] = field(default_factory=dict)
    preview: str = ""                  # diff / cmd-line / filename
    requested_at: float = field(default_factory=time.time)
    timeout_seconds: float = 180.0
    session_id: str = ""               # for audit / scope filtering

    def to_public(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "action": self.action,
            "risk": self.risk,
            "payload": dict(self.payload),
            "preview": self.preview,
            "requested_at": self.requested_at,
            "timeout_seconds": self.timeout_seconds,
            "session_id": self.session_id,
        }


@dataclass
class ApprovalDecision:
    request_id: str
    approved: bool
    reason: str = ""                   # "user", "auto_low", "auto_session", "timeout"
    decided_at: float = field(default_factory=time.time)


# ─── Gate ────────────────────────────────────────────────────────────


# Action classes that ALWAYS auto-approve when ``auto_approve_low`` is on.
AUTO_APPROVE_LOW: frozenset[str] = frozenset(
    {
        "workspace_list",
        "workspace_read",
        "workspace_list_projects",
        "workspace_git_log",
        "workspace_git_diff",
    }
)


class ApprovalGate:
    """Async ask-the-user wrapper."""

    def __init__(
        self,
        *,
        broadcast: Callable[[dict[str, Any]], Awaitable[None]],
        default_timeout: float = 180.0,
        auto_approve_low: bool = True,
        db: aiosqlite.Connection | None = None,
    ) -> None:
        self._broadcast = broadcast
        self._default_timeout = float(default_timeout)
        self._auto_low = bool(auto_approve_low)
        self._db = db
        self._pending: dict[str, asyncio.Future[ApprovalDecision]] = {}
        # Per-session per-action auto-approve overrides ("yes for the
        # rest of this session"). Keyed by ``(session_id, action_prefix)``
        # so Mike can grant `workspace_write/skill/foo/*` without auto-
        # approving everything.
        self._session_grants: dict[tuple[str, str], float] = {}

    # ─── Audit DB ────────────────────────────────────────────────────

    async def init_db(self, db: aiosqlite.Connection) -> None:
        self._db = db
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS approvals (
                request_id   TEXT PRIMARY KEY,
                session_id   TEXT NOT NULL DEFAULT '',
                action       TEXT NOT NULL,
                risk         TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                preview      TEXT NOT NULL DEFAULT '',
                approved     INTEGER NOT NULL,
                reason       TEXT NOT NULL DEFAULT '',
                requested_at REAL NOT NULL,
                decided_at   REAL NOT NULL
            )
            """
        )
        await db.commit()

    async def _record(
        self, req: ApprovalRequest, decision: ApprovalDecision
    ) -> None:
        if self._db is None:
            return
        import json
        try:
            await self._db.execute(
                "INSERT OR REPLACE INTO approvals "
                "(request_id, session_id, action, risk, payload_json, preview, "
                "approved, reason, requested_at, decided_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    req.request_id,
                    req.session_id,
                    req.action,
                    req.risk,
                    json.dumps(req.payload, ensure_ascii=False),
                    req.preview,
                    1 if decision.approved else 0,
                    decision.reason,
                    req.requested_at,
                    decision.decided_at,
                ),
            )
            await self._db.commit()
        except Exception as exc:  # noqa: BLE001
            log.warning("coder.approval_audit_failed err=%s", exc)

    # ─── Session-scope auto-approve ──────────────────────────────────

    def grant_session(
        self,
        *,
        session_id: str,
        action: str,
        ttl_seconds: float = 3600.0,
    ) -> None:
        """Mike said "yes for the rest of this session" — remember it."""
        until = time.time() + max(60.0, float(ttl_seconds))
        self._session_grants[(session_id, action)] = until

    def revoke_session(self, *, session_id: str, action: str | None = None) -> int:
        """Drop session grants. Returns the number removed."""
        keys = [
            k for k in self._session_grants
            if k[0] == session_id and (action is None or k[1] == action)
        ]
        for k in keys:
            self._session_grants.pop(k, None)
        return len(keys)

    def has_session_grant(self, *, session_id: str, action: str) -> bool:
        until = self._session_grants.get((session_id, action))
        if until is None:
            return False
        if until < time.time():
            self._session_grants.pop((session_id, action), None)
            return False
        return True

    # ─── Core: request + resolve ────────────────────────────────────

    async def request(
        self,
        *,
        action: str,
        risk: str,
        payload: dict[str, Any] | None = None,
        preview: str = "",
        session_id: str = "",
        timeout_seconds: float | None = None,
    ) -> ApprovalDecision:
        """Ask the user. Returns the decision (approved + reason).

        Auto-approval shortcuts (no UI roundtrip):
        * Action in :data:`AUTO_APPROVE_LOW` AND gate has auto_low enabled.
        * Active per-session grant for this action AND risk != HIGH.
        """
        if risk not in KNOWN_RISKS:
            risk = RISK_MED  # be conservative

        req = ApprovalRequest(
            request_id=uuid.uuid4().hex[:12],
            action=action,
            risk=risk,
            payload=dict(payload or {}),
            preview=preview or "",
            timeout_seconds=float(
                timeout_seconds if timeout_seconds is not None else self._default_timeout
            ),
            session_id=session_id,
        )

        # ── Auto-approve shortcuts ────────────────────────────────
        if self._auto_low and action in AUTO_APPROVE_LOW and risk == RISK_LOW:
            decision = ApprovalDecision(
                request_id=req.request_id, approved=True, reason="auto_low",
            )
            await self._record(req, decision)
            return decision
        if (
            risk != RISK_HIGH
            and self.has_session_grant(session_id=session_id, action=action)
        ):
            decision = ApprovalDecision(
                request_id=req.request_id, approved=True, reason="auto_session",
            )
            await self._record(req, decision)
            return decision

        # ── Real ask ────────────────────────────────────────────
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ApprovalDecision] = loop.create_future()
        self._pending[req.request_id] = future

        try:
            await self._broadcast(
                {"type": "coder_approval_request", **req.to_public()}
            )
        except Exception as exc:  # noqa: BLE001
            log.error("coder.approval_broadcast_failed err=%s", exc)
            self._pending.pop(req.request_id, None)
            decision = ApprovalDecision(
                request_id=req.request_id, approved=False, reason="broadcast_failed",
            )
            await self._record(req, decision)
            return decision

        try:
            decision = await asyncio.wait_for(future, timeout=req.timeout_seconds)
        except asyncio.TimeoutError:
            decision = ApprovalDecision(
                request_id=req.request_id, approved=False, reason="timeout",
            )
        finally:
            self._pending.pop(req.request_id, None)

        await self._record(req, decision)
        # Side-effect: if the user picked "approve_session" we promote
        # the grant cache (the WS handler can also call grant_session
        # directly; this is the convenience path).
        if decision.approved and decision.reason == "approve_session":
            self.grant_session(session_id=session_id, action=action)
        return decision

    def resolve(
        self,
        *,
        request_id: str,
        approved: bool,
        reason: str = "user",
    ) -> bool:
        """Hand a decision in from the WS handler. Returns True if a
        pending request was found and resolved, False otherwise."""
        future = self._pending.pop(request_id, None)
        if future is None or future.done():
            return False
        future.set_result(
            ApprovalDecision(
                request_id=request_id,
                approved=bool(approved),
                reason=reason,
            )
        )
        return True

    # ─── Introspection ──────────────────────────────────────────────

    def list_pending(self) -> list[str]:
        """Return ``request_id``s that are still waiting for a decision."""
        return [
            rid for rid, fut in self._pending.items() if not fut.done()
        ]
