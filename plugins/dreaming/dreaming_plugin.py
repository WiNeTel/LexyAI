"""
Lexy AI - Dreaming Plugin.

Background memory consolidation that runs during idle periods:

* **Duplicate Detection & Merge** – finds similar memories in "facts" and
  "solutions", asks the LLM to merge them into one improved entry.
* **Fact Linkage** – picks two random facts and asks the LLM whether they
  are related; if yes, stores a one-sentence link as a new "facts" entry.
* **Staleness Report** – counts old "context" entries and broadcasts a
  summary to the GUI.

The consolidation loop runs every ``interval_minutes`` but only when the
user has been idle for at least ``min_idle_minutes`` and we are inside the
configured ``quiet_hours`` window.  Manual triggers are available via WS.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from datetime import datetime
from typing import Any

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

log = get_logger(module="dreaming")

# Maximale Zeichenlaenge fuer LLM-Anfragen (Sicherheitslimit)
_MAX_CONTENT_LEN: int = 2000


class DreamingPlugin(BasePlugin):
    """Memory consolidation during idle time."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._interval_minutes: int = 120
        self._quiet_start_hour: int = 2
        self._quiet_start_minute: int = 0
        self._quiet_end_hour: int = 6
        self._quiet_end_minute: int = 0
        self._max_ops: int = 10
        self._similarity_threshold: float = 0.85
        self._decay_days: int = 90
        self._min_idle_minutes: int = 30
        self._dreaming_enabled: bool = True

        self._last_user_activity: float = time.time()
        self._loop_task: asyncio.Task[None] | None = None
        self._running: bool = False

        # Statistik fuer den laufenden/letzten Zyklus
        self._cycle_count: int = 0

    # ─── Lifecycle ──────────────────────────────────────────────────

    async def on_load(self) -> None:
        """Read plugin config and parse quiet-hours."""
        config = self.api.get_config()
        self._interval_minutes = int(config.get("interval_minutes", 120))
        self._max_ops = int(config.get("max_operations_per_cycle", 10))
        self._similarity_threshold = float(config.get("similarity_threshold", 0.85))
        self._decay_days = int(config.get("decay_days", 90))
        self._min_idle_minutes = int(config.get("min_idle_minutes", 30))
        self._dreaming_enabled = bool(config.get("enabled", True))

        quiet_hours: list[str] = config.get("quiet_hours", ["02:00", "06:00"])
        self._parse_quiet_hours(quiet_hours)
        log.info(
            "dreaming.loaded",
            interval_minutes=self._interval_minutes,
            quiet_hours=quiet_hours,
            threshold=self._similarity_threshold,
        )

    async def on_enable(self) -> None:
        """Register event listeners, WS handlers, and start the background loop."""
        # Benutzeraktivitaet tracken
        self.api.on_event("core.user_message", self._on_user_activity)

        # WebSocket-Handler fuer manuelle Steuerung
        self.api.register_ws_handler("dreaming_trigger", self._handle_ws_trigger)
        self.api.register_ws_handler("dreaming_toggle", self._handle_ws_toggle)

        # Hintergrund-Loop starten
        self._running = True
        self._loop_task = asyncio.create_task(self._loop(), name="dreaming.loop")
        log.info("dreaming.enabled")

    async def on_disable(self) -> None:
        """Stop background loop and clean up."""
        self._running = False
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._loop_task = None
        log.info("dreaming.disabled")

    # ─── Event handlers ─────────────────────────────────────────────

    async def _on_user_activity(self, data: dict[str, Any]) -> None:
        """Track last user interaction timestamp."""
        self._last_user_activity = time.time()

    # ─── WebSocket handlers ─────────────────────────────────────────

    async def _handle_ws_trigger(self, client: Any, message: dict[str, Any]) -> None:
        """Manual dreaming cycle trigger from the GUI."""
        log.info("dreaming.manual_trigger")
        results = await self._run_cycle(force=True)
        await client.send_json({"type": "dreaming_result", "operations": results})

    async def _handle_ws_toggle(self, client: Any, message: dict[str, Any]) -> None:
        """Toggle dreaming on/off from the GUI."""
        self._dreaming_enabled = bool(message.get("enabled", not self._dreaming_enabled))
        log.info("dreaming.toggled", enabled=self._dreaming_enabled)
        await client.send_json({
            "type": "dreaming_toggled",
            "enabled": self._dreaming_enabled,
        })

    # ─── Background loop ────────────────────────────────────────────

    async def _loop(self) -> None:
        """Periodically run consolidation cycles when conditions are met."""
        try:
            while self._running:
                await asyncio.sleep(self._interval_minutes * 60)
                if not self._running:
                    break
                if not self._dreaming_enabled:
                    log.debug("dreaming.skipped", reason="disabled")
                    continue
                if not self._is_quiet_hour():
                    log.debug("dreaming.skipped", reason="outside_quiet_hours")
                    continue
                if not self._is_idle():
                    log.debug("dreaming.skipped", reason="user_active")
                    continue
                try:
                    results = await self._run_cycle(force=False)
                    if results:
                        await self.api.ws_broadcast({
                            "type": "dreaming_result",
                            "operations": results,
                        })
                except Exception as exc:  # noqa: BLE001
                    log.error("dreaming.cycle_failed", error=str(exc))
        except asyncio.CancelledError:
            pass

    # ─── Cycle orchestration ────────────────────────────────────────

    async def _run_cycle(self, *, force: bool) -> list[dict[str, Any]]:
        """
        Execute one dreaming cycle with a mix of the three operation types.

        Returns a list of operation result dicts for WS broadcast and events.
        """
        self._cycle_count += 1
        log.info("dreaming.cycle_start", cycle=self._cycle_count, force=force)
        results: list[dict[str, Any]] = []
        ops_done: int = 0

        # Verteilung: 40 % Duplikat-Erkennung, 30 % Fact-Linkage, 30 % Staleness
        operations: list[str] = (
            ["deduplicate"] * 4
            + ["link_facts"] * 3
            + ["staleness"] * 3
        )
        random.shuffle(operations)

        for op_type in operations:
            if ops_done >= self._max_ops:
                break
            try:
                if op_type == "deduplicate":
                    result = await self._op_deduplicate()
                elif op_type == "link_facts":
                    result = await self._op_link_facts()
                else:
                    result = await self._op_staleness_report()

                if result is not None:
                    results.append(result)
                    ops_done += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("dreaming.op_failed", op=op_type, error=str(exc))

        # Events emittieren
        summary = {
            "cycle": self._cycle_count,
            "operations_completed": ops_done,
            "results": results,
        }
        await self.api.emit("core.dreaming_cycle", summary)
        log.info("dreaming.cycle_done", cycle=self._cycle_count, ops=ops_done)
        return results

    # ─── Operation 1: Duplicate Detection & Merge ───────────────────

    async def _op_deduplicate(self) -> dict[str, Any] | None:
        """
        Pick a random recent memory from "facts" or "solutions",
        search for duplicates, and merge via LLM if found.
        """
        collection = random.choice(["facts", "solutions"])

        # Zufaelligen "Seed"-Eintrag holen ueber eine breite Suche
        seed_query = random.choice([
            "wichtig", "problem", "wissen", "fakt", "erinnerung",
            "solution", "fehler", "gelernt", "tipp", "merkregel",
        ])
        seed_results = await self.api.memory_recall(
            query=seed_query, collection=collection, limit=10,
        )
        if not seed_results:
            log.debug("dreaming.dedup_skip", reason="no_seed_results", collection=collection)
            return None

        # Einen zufaelligen Eintrag als Ausgangspunkt waehlen
        seed = random.choice(seed_results)
        seed_content: str = seed.get("content", "")
        seed_id: str = seed.get("id", "")
        if not seed_content:
            return None

        # Aehnliche Eintraege suchen
        similar = await self.api.memory_recall(
            query=seed_content[:500], collection=collection, limit=10,
        )

        # Duplikate filtern: hohe Aehnlichkeit, aber nicht der Seed selbst
        duplicates: list[dict[str, Any]] = []
        for item in similar:
            if item.get("id") == seed_id:
                continue
            score = item.get("score", 0.0)
            if score >= self._similarity_threshold:
                duplicates.append(item)

        if not duplicates:
            log.debug("dreaming.dedup_no_dupes", seed_id=seed_id, collection=collection)
            return None

        # LLM-Merge: die besten Duplikate zusammenfuehren
        best_dupe = duplicates[0]
        dupe_content: str = best_dupe.get("content", "")
        dupe_id: str = best_dupe.get("id", "")

        merge_prompt = (
            "You are a memory consolidation assistant. "
            "Two memory entries appear to be duplicates or very similar. "
            "Merge them into one improved, concise entry that preserves "
            "all unique information from both. Output ONLY the merged text, "
            "nothing else.\n\n"
            f"Entry A:\n{seed_content[:_MAX_CONTENT_LEN]}\n\n"
            f"Entry B:\n{dupe_content[:_MAX_CONTENT_LEN]}"
        )

        merged_text = await self.api.llm_chat(
            messages=[{"role": "user", "content": merge_prompt}],
            brain="e4b",
        )
        merged_text = merged_text.strip()

        if not merged_text or len(merged_text) < 10:
            log.warning("dreaming.dedup_llm_empty", seed_id=seed_id, dupe_id=dupe_id)
            return None

        # Zusammengefuehrten Eintrag speichern
        merged_id = await self.api.memory_store(
            text=merged_text,
            collection=collection,
            metadata={
                "type": "merged",
                "source_ids": [seed_id, dupe_id],
                "merged_at": time.time(),
                "dreaming_cycle": self._cycle_count,
            },
        )

        # ChromaDB hat kein einfaches delete-by-query, daher loggen wir die
        # Originale als "zu bereinigen". Ein zukuenftiger Maintenance-Job
        # kann diese aufgrund der IDs entfernen.
        log.info(
            "dreaming.dedup_merged",
            merged_id=merged_id,
            source_ids=[seed_id, dupe_id],
            collection=collection,
            similarity=best_dupe.get("score", 0.0),
        )

        return {
            "op": "deduplicate",
            "collection": collection,
            "merged_id": merged_id,
            "source_ids": [seed_id, dupe_id],
            "similarity": round(best_dupe.get("score", 0.0), 3),
            "merged_preview": merged_text[:200],
        }

    # ─── Operation 2: Fact Linkage ──────────────────────────────────

    async def _op_link_facts(self) -> dict[str, Any] | None:
        """
        Pick two random facts and ask the LLM whether they are related.
        If yes, store a link entry.
        """
        # Zwei verschiedene Seed-Queries um unterschiedliche Fakten zu bekommen
        queries = [
            "wichtig", "wissen", "fakt", "erinnerung", "gelernt",
            "person", "ort", "projekt", "technik", "konzept",
        ]
        query_a, query_b = random.sample(queries, 2)

        results_a = await self.api.memory_recall(query=query_a, collection="facts", limit=5)
        results_b = await self.api.memory_recall(query=query_b, collection="facts", limit=5)

        if not results_a or not results_b:
            log.debug("dreaming.link_skip", reason="insufficient_facts")
            return None

        fact_a = random.choice(results_a)
        fact_b = random.choice(results_b)

        # Sicherstellen dass es nicht der gleiche Eintrag ist
        if fact_a.get("id") == fact_b.get("id"):
            return None

        content_a: str = fact_a.get("content", "")
        content_b: str = fact_b.get("content", "")
        id_a: str = fact_a.get("id", "")
        id_b: str = fact_b.get("id", "")

        if not content_a or not content_b:
            return None

        link_prompt = (
            "You are a knowledge graph assistant. Determine if these two facts "
            "are meaningfully related. If they ARE related, respond with EXACTLY "
            "one sentence describing the connection (in the same language as the "
            "facts). If they are NOT related, respond with exactly: NO_LINK\n\n"
            f"Fact A:\n{content_a[:_MAX_CONTENT_LEN]}\n\n"
            f"Fact B:\n{content_b[:_MAX_CONTENT_LEN]}"
        )

        response = await self.api.llm_chat(
            messages=[{"role": "user", "content": link_prompt}],
            brain="e4b",
        )
        response = response.strip()

        if not response or "NO_LINK" in response.upper():
            log.debug("dreaming.link_no_relation", id_a=id_a, id_b=id_b)
            return None

        # Verbindung als neuen Fakt speichern
        link_text = f"[Verbindung] {response}"
        link_id = await self.api.memory_store(
            text=link_text,
            collection="facts",
            metadata={
                "type": "link",
                "source_ids": [id_a, id_b],
                "created_at": time.time(),
                "dreaming_cycle": self._cycle_count,
            },
        )

        log.info(
            "dreaming.link_created",
            link_id=link_id,
            source_ids=[id_a, id_b],
            link_preview=response[:200],
        )

        return {
            "op": "link_facts",
            "link_id": link_id,
            "source_ids": [id_a, id_b],
            "link_text": response[:200],
        }

    # ─── Operation 3: Staleness Report ──────────────────────────────

    async def _op_staleness_report(self) -> dict[str, Any] | None:
        """
        Count old entries in the "context" collection and broadcast a summary.
        """
        cutoff = time.time() - (self._decay_days * 86400)

        # Breite Suche um moeglichst viele Context-Eintraege zu erfassen
        broad_queries = [
            "kontext", "gespraech", "session", "unterhaltung", "chat",
        ]
        query = random.choice(broad_queries)

        results = await self.api.memory_recall(
            query=query, collection="context", limit=50,
        )

        if not results:
            log.debug("dreaming.staleness_skip", reason="no_context_entries")
            return None

        total_checked = len(results)
        stale_count = 0
        stale_ids: list[str] = []

        for item in results:
            metadata = item.get("metadata", {})
            # Verschiedene Timestamp-Formate unterstuetzen
            item_time = (
                metadata.get("timestamp")
                or metadata.get("created_at")
                or metadata.get("stored_at")
                or 0.0
            )
            try:
                item_time = float(item_time)
            except (ValueError, TypeError):
                item_time = 0.0

            if 0 < item_time < cutoff:
                stale_count += 1
                stale_ids.append(item.get("id", "unknown"))

        report = {
            "total_checked": total_checked,
            "stale_count": stale_count,
            "decay_days": self._decay_days,
            "stale_ids": stale_ids[:20],  # Maximal 20 IDs im Report
        }

        # Event fuer andere Plugins
        await self.api.emit("core.dreaming_report", report)

        # GUI-Broadcast mit lesbarer Zusammenfassung
        if stale_count > 0:
            summary_text = (
                f"Dreaming: {stale_count} von {total_checked} Context-Eintraegen "
                f"sind aelter als {self._decay_days} Tage."
            )
        else:
            summary_text = (
                f"Dreaming: Alle {total_checked} geprueften Context-Eintraege "
                f"sind aktuell (< {self._decay_days} Tage)."
            )

        await self.api.ws_broadcast({
            "type": "dreaming_staleness",
            "summary": summary_text,
            **report,
        })

        log.info(
            "dreaming.staleness_report",
            total_checked=total_checked,
            stale_count=stale_count,
        )

        return {
            "op": "staleness_report",
            "total_checked": total_checked,
            "stale_count": stale_count,
            "decay_days": self._decay_days,
        }

    # ─── Helpers ────────────────────────────────────────────────────

    def _parse_quiet_hours(self, quiet_hours: list[str]) -> None:
        """Parse quiet_hours config list like ["02:00", "06:00"]."""
        try:
            start_parts = quiet_hours[0].split(":")
            end_parts = quiet_hours[1].split(":")
            self._quiet_start_hour = int(start_parts[0])
            self._quiet_start_minute = int(start_parts[1])
            self._quiet_end_hour = int(end_parts[0])
            self._quiet_end_minute = int(end_parts[1])
        except (IndexError, ValueError) as exc:
            log.warning("dreaming.quiet_hours_parse_failed", error=str(exc))
            # Fallback: 02:00 - 06:00
            self._quiet_start_hour = 2
            self._quiet_start_minute = 0
            self._quiet_end_hour = 6
            self._quiet_end_minute = 0

    def _is_quiet_hour(self) -> bool:
        """Check if the current time is within the configured quiet-hours window."""
        now = datetime.now()
        current_minutes = now.hour * 60 + now.minute
        start_minutes = self._quiet_start_hour * 60 + self._quiet_start_minute
        end_minutes = self._quiet_end_hour * 60 + self._quiet_end_minute

        if start_minutes <= end_minutes:
            # Normaler Bereich, z.B. 02:00 - 06:00
            return start_minutes <= current_minutes < end_minutes
        else:
            # Ueber Mitternacht, z.B. 22:00 - 06:00
            return current_minutes >= start_minutes or current_minutes < end_minutes

    def _is_idle(self) -> bool:
        """Check if user has been idle long enough."""
        idle_seconds = time.time() - self._last_user_activity
        return idle_seconds >= self._min_idle_minutes * 60
