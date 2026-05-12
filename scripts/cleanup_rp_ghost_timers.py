"""
Phase 13.x — manual cleanup of stale character_pulse / autonomous_sim
timers in the scheduler plugin's SQLite.

Background: Phase 13.1 added an in-process timer-cancel routine that
runs once during the wipe. But it relies on the scheduler plugin
being loaded BEFORE character_chat (and on the scheduler's in-memory
list_timers RPC working). When that load order doesn't fire, the
old timers — left over from pre-Phase-13 sessions — stay in the
scheduler's DB and keep firing every few minutes.

This script does the cleanup directly against
``data/plugins/scheduler/scheduler.db``: it cancels every recurring
timer whose label starts with ``character_pulse:`` or
``autonomous_sim:``. Backend MUST be stopped while you run this —
otherwise the scheduler may re-write rows over your changes.

Usage::

    # 1. Backend stoppen (Ctrl+C im CMD-Fenster)
    conda activate lexyai
    python scripts/cleanup_rp_ghost_timers.py
    # 2. Backend wieder starten

The script prints what it would cancel first; pass ``--commit`` to
actually mark them cancelled.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCHEDULER_DB = (
    ROOT / "data" / "plugins" / "scheduler" / "scheduler.db"
)


def _open_db(path: Path) -> sqlite3.Connection:
    if not path.exists():
        print(f"[!] No scheduler DB at {path} — nothing to clean.")
        sys.exit(0)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _list_active_rp_timers(
    conn: sqlite3.Connection,
) -> list[sqlite3.Row]:
    """Active = not fired (or recurring) and not cancelled."""
    cur = conn.execute(
        """
        SELECT id, kind, label, repeat_pattern, fire_at, session_id,
               cancelled, fired
        FROM timers
        WHERE cancelled = 0
          AND (
              label LIKE 'character_pulse:%'
              OR label LIKE 'autonomous_sim:%'
          )
        ORDER BY label, fire_at
        """
    )
    return list(cur.fetchall())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cancel stale character_pulse / autonomous_sim timers.",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Actually write cancelled=1 (default: dry-run preview).",
    )
    args = parser.parse_args()

    conn = _open_db(SCHEDULER_DB)
    rows = _list_active_rp_timers(conn)
    if not rows:
        print("[ok] No active character_pulse / autonomous_sim timers found.")
        return 0

    print(f"[i] Found {len(rows)} stale RP timers in {SCHEDULER_DB}:")
    for r in rows:
        marker = "RECURRING" if r["repeat_pattern"] else "ONE-SHOT"
        sess = r["session_id"] or "(no session)"
        print(
            f"    {marker:9s}  {r['label']:38s}  "
            f"pattern={r['repeat_pattern'] or '-':10s}  "
            f"session={sess}"
        )

    if not args.commit:
        print()
        print("[dry-run] No changes written. Re-run with --commit to cancel.")
        return 0

    cur = conn.execute(
        """
        UPDATE timers
        SET cancelled = 1
        WHERE cancelled = 0
          AND (
              label LIKE 'character_pulse:%'
              OR label LIKE 'autonomous_sim:%'
          )
        """
    )
    cancelled = cur.rowcount or 0
    conn.commit()
    conn.close()
    print(f"[ok] Cancelled {cancelled} timers.")
    print("[i] Now restart the backend. Fresh pulse-timers will be")
    print("    re-registered the next time you attach a character to a session.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
