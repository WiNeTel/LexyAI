"""
Phase 12 — Memory-capture plugin.

Two layers of fact-capture, both fed off the agent's
``before_user_input`` hook:

1. **Explicit-trigger** (Layer 2 in the plan, ``Layer B``)
   Synchronous regex match for "merke dir das" / "remember this"
   style phrasings. Cheap, deterministic, no LLM call. Stores the
   captured fact into the ``facts`` collection with metadata
   ``source="trigger_phrase"``.

2. **Implicit classification** (Layer 3, ``Layer C``)
   Optional. When ``implicit_capture_enabled=true`` in config, a tiny
   LLM call (e4b) classifies whether the user's message contains a
   stable fact worth remembering even WITHOUT an explicit trigger
   ("Ich wohne am Nordpol" alone). Gated by a per-session cooldown
   plus a confidence threshold plus a semantic-dedup recall against
   the existing facts so we don't store the same fact twice.

Both layers broadcast ``memory_captured`` so the frontend can show
the user a small toast confirming the capture.

The plugin is intentionally additive: it never modifies the ctx the
agent is processing, never sets ``skip_agent``, never alters the
user message. If detection fails or storage errors out, we log and
continue — the chat round still goes through normally.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

from .trigger_patterns import extract_fact

log = get_logger(module="memory_capture")


# Tiny JSON-only classifier prompt for Layer 3. Kept inline because
# it's a single-purpose template and inlining lets us keep the whole
# Layer-3 logic in one file. Feel free to tune phrasing — the
# response shape is fixed (``{is_fact, fact, confidence}``).
_CLASSIFY_PROMPT = """\
Du bist ein Fact-Extractor. Lies die folgende User-Nachricht und entscheide:

1. Enthält sie einen STABILEN User-Fakt? Stabile Facts sind:
   - Adresse, Wohnort, Geburtstag, Alter
   - Allergien, gesundheitliche Einschränkungen
   - Job, Beruf, Studium, Ausbildung
   - Wichtige Personen (Namen, Beziehungen)
   - Stabile Vorlieben/Abneigungen ("Ich liebe Jazz")
   - Lifestyle/Routinen ("Ich arbeite Nachts")
   - Wichtige Besitztümer/Tools (Auto, Hund, Lieblings-IDE)

   NICHT speichern:
   - Aktuelle Stimmung ("Ich bin müde")
   - Aktuelle Aktivität ("Ich koche gerade")
   - Fragen, Kommandos, Smalltalk
   - Hypothesen, Annahmen, Möglichkeiten

2. Falls ja: Extrahiere den Fakt als 1-2 kurze Sätze, in der dritten
   Person formuliert ("User wohnt am Nordpol", nicht "Ich wohne am
   Nordpol").

3. Antworte AUSSCHLIESSLICH in diesem JSON-Format, ohne weiteren Text,
   ohne Markdown-Codeblock:

{"is_fact": true|false, "fact": "...", "confidence": 0.0-1.0}

User-Nachricht: %USER_MESSAGE%"""


# Tolerant JSON extractor. The LLM might wrap the response in
# ```json``` code fences or add stray prose; we strip those before
# json.loads.
_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


class MemoryCapturePlugin(BasePlugin):
    """Captures user facts via trigger phrases and (optionally) LLM."""

    # ─── Lifecycle ──────────────────────────────────────────────────

    async def on_load(self) -> None:
        self._apply_config(self.api.get_config())
        # Per-fact cooldown so the same fact-text within ``cooldown``
        # seconds is dropped (saves a wasted store call when Mike
        # accidentally double-sends).
        self._recent_facts: dict[str, float] = {}
        # Per-session cooldown for the LLM classifier so we don't burn
        # an extra LLM call on every single user message in a busy
        # session.
        self._session_last_classify: dict[str, float] = {}
        log.info(
            "memory_capture.loaded",
            languages=self._enabled_languages,
            implicit=self._implicit_enabled,
        )

    async def on_enable(self) -> None:
        # Layer 2 — explicit trigger detection. Always on.
        self.api.register_hook(
            "before_user_input",
            self._capture_explicit_facts,
            priority=80,  # before character_chat's hook (60), after upload-prep
        )
        # Layer 3 — implicit LLM classifier. Only registers when
        # enabled in config. Avoids a no-op hook entry that costs
        # nothing now but might confuse debugging later.
        if self._implicit_enabled:
            self.api.register_hook(
                "before_user_input",
                self._capture_implicit_facts,
                priority=70,
            )
        log.info(
            "memory_capture.enabled",
            implicit=self._implicit_enabled,
        )

    async def on_disable(self) -> None:
        """No plugin-owned resources besides the in-memory cooldown
        dicts; PluginAPI cleans up the hook registrations."""
        self._recent_facts.clear()
        self._session_last_classify.clear()
        log.info("memory_capture.disabled")

    async def on_config_changed(self, cfg: dict[str, Any]) -> None:
        """Live-reload config — toggling implicit-capture takes effect
        immediately for the next user message; no restart needed."""
        prev_implicit = self._implicit_enabled
        self._apply_config(cfg)
        if prev_implicit != self._implicit_enabled:
            log.info(
                "memory_capture.implicit_toggled",
                enabled=self._implicit_enabled,
            )

    def _apply_config(self, cfg: dict[str, Any]) -> None:
        self._enabled_languages: tuple[str, ...] = tuple(
            str(lang).lower()
            for lang in cfg.get("enabled_languages", ["de", "en"])
            if lang
        ) or ("de", "en")
        self._cooldown = float(cfg.get("cooldown_seconds", 60))
        self._implicit_enabled = bool(cfg.get("implicit_capture_enabled", False))
        self._implicit_confidence_min = float(
            cfg.get("implicit_confidence_min", 0.7)
        )
        self._implicit_brain = str(cfg.get("implicit_brain", "e4b"))
        self._implicit_classify_cooldown = float(
            cfg.get("implicit_classify_cooldown", 30)
        )

    # ─── Layer 2: explicit trigger ──────────────────────────────────

    async def _capture_explicit_facts(
        self, ctx: dict[str, Any]
    ) -> dict[str, Any]:
        """``before_user_input`` hook — fires for every chat message."""
        text = str(ctx.get("text") or "").strip()
        if not text:
            return ctx

        match = extract_fact(text, enabled_languages=self._enabled_languages)
        if match is None:
            return ctx

        lang, fact = match
        if not await self._store_fact_with_dedup(
            fact=fact,
            session_id=str(ctx.get("session_id", "") or ""),
            source="trigger_phrase",
            metadata={
                "language": lang,
                "captured_at": time.time(),
                "trigger_text": text[:200],
            },
        ):
            return ctx

        # Mark this message as already-captured so the implicit
        # classifier in Layer 3 doesn't make an LLM call for the
        # same thing.
        ctx["_memory_capture_explicit"] = fact
        return ctx

    # ─── Layer 3: implicit LLM classifier ───────────────────────────

    async def _capture_implicit_facts(
        self, ctx: dict[str, Any]
    ) -> dict[str, Any]:
        """LLM-driven fact detection for messages without an explicit
        trigger phrase. Gated by per-session cooldown + confidence
        threshold + semantic dedup."""

        # If Layer 2 already fired, no need to reclassify.
        if ctx.get("_memory_capture_explicit"):
            return ctx

        text = str(ctx.get("text") or "").strip()
        if not text or len(text) < 10:
            return ctx

        session_id = str(ctx.get("session_id", "") or "")
        now = time.time()
        last = self._session_last_classify.get(session_id, 0.0)
        if now - last < self._implicit_classify_cooldown:
            return ctx
        self._session_last_classify[session_id] = now

        classification = await self._classify_fact_worthiness(text)
        if classification is None:
            return ctx
        fact, confidence = classification
        if confidence < self._implicit_confidence_min:
            log.debug(
                "memory_capture.implicit_below_threshold",
                confidence=confidence,
                threshold=self._implicit_confidence_min,
            )
            return ctx
        if not fact or len(fact.strip()) < 5:
            return ctx

        # Semantic dedup: if the fact is essentially already in
        # facts, skip the store.
        try:
            similar = await self.api.memory_recall(
                query=fact,
                collection="facts",
                limit=3,
            )
        except Exception:  # noqa: BLE001
            similar = []
        if similar and similar[0].get("score", 0.0) >= 0.85:
            log.info(
                "memory_capture.implicit_dedup_skip",
                fact=fact[:60],
                top_score=similar[0].get("score"),
            )
            return ctx

        await self._store_fact_with_dedup(
            fact=fact,
            session_id=session_id,
            source="implicit_capture",
            metadata={
                "confidence": float(confidence),
                "captured_at": now,
                "trigger_text": text[:200],
            },
        )
        return ctx

    async def _classify_fact_worthiness(
        self, user_message: str
    ) -> tuple[str, float] | None:
        """Run the LLM classifier. Returns ``(fact, confidence)`` or None.

        Errors are swallowed and logged — the classifier is best-effort
        and must not break the user's chat.
        """
        if self.api._app.llm is None:
            return None

        prompt = _CLASSIFY_PROMPT.replace("%USER_MESSAGE%", user_message)
        try:
            response = await self.api._app.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                brain=self._implicit_brain,
                max_tokens=200,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "memory_capture.classifier_error",
                error=str(exc),
            )
            return None

        text = str(response or "").strip()
        if not text:
            return None
        # Strip ```json fences if any
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        # Take the first {...} block in case the LLM added prose.
        match = _JSON_BLOCK_RE.search(text)
        if match:
            text = match.group(0)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning(
                "memory_capture.classifier_bad_json",
                error=str(exc),
                response_preview=text[:120],
            )
            return None
        if not isinstance(data, dict):
            return None
        if not data.get("is_fact"):
            return None
        fact = str(data.get("fact") or "").strip()
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return (fact, confidence)

    # ─── Storage helper (shared by both layers) ─────────────────────

    async def _store_fact_with_dedup(
        self,
        *,
        fact: str,
        session_id: str,
        source: str,
        metadata: dict[str, Any],
    ) -> bool:
        """Persist a fact to ``facts``, with cooldown dedup + broadcast.

        Returns True if the fact was actually stored, False if skipped
        (cooldown, error, or empty fact).
        """
        fact = fact.strip()
        if not fact:
            return False

        now = time.time()
        last_seen = self._recent_facts.get(fact)
        if last_seen is not None and (now - last_seen) < self._cooldown:
            log.debug(
                "memory_capture.cooldown_skip",
                fact=fact[:80],
                age=now - last_seen,
            )
            return False

        full_metadata = {"source": source, "session_id": session_id}
        full_metadata.update(metadata)

        try:
            await self.api.memory_store(
                text=fact,
                collection="facts",
                metadata=full_metadata,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "memory_capture.store_failed",
                fact=fact[:80],
                source=source,
                error=str(exc),
            )
            return False

        self._recent_facts[fact] = now
        log.info(
            "memory_capture.fact_stored",
            source=source,
            fact_preview=fact[:80],
            session_id=session_id,
        )
        try:
            await self.api.ws_broadcast({
                "type": "memory_captured",
                "kind": source,  # "trigger_phrase" / "implicit_capture"
                "fact": fact[:200],
                "session_id": session_id,
            })
        except Exception:  # noqa: BLE001
            pass
        return True
