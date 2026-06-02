"""
Lexy AI - Coordination: SceneDirector.

The auto-provisioning brain. Reads a character's persona / relationships /
scenario and the recent chat, and proposes which simulation needs the scene
should have — e.g. a character who "has a 2-month-old baby" → a
``baby.hunger`` need with the character as caregiver; a birth narrated in the
chat → provision the newborn's needs.

It only *proposes* (returns a structured plan). What happens with the plan is
the plugin's call, gated by ``scene_director_mode`` (off / confirm / auto).

Built on :class:`FactExtractor` (one cheap LLM call, tolerant JSON, fail-safe
to an empty plan), then strictly sanitised so a wild LLM answer can never
reach ``define_need`` malformed.
"""

from __future__ import annotations

from typing import Any

from lexy_core.coordination.convergence import LLMChat
from lexy_core.coordination.fact_extractor import FactExtractor
from lexy_core.utils.logging import get_logger

log = get_logger(module="coordination.scene_director")

_INSTRUCTION: str = (
    "Du bist ein Spielleiter-Setup-Assistent. Lies die Figur und die letzten "
    "Ereignisse. Wenn es Abhaengige oder Zustaende gibt, die als fortlaufende "
    "Beduerfnisse simuliert werden sollten (z.B. ein Baby -> hunger; ein "
    "gerade geborenes Kind -> neues Baby mit hunger), gib sie als JSON zurueck. "
    "Format:\n"
    '{"needs": [{"entity": "baby", "attribute": "hunger", '
    '"rate_per_minute": 2, "caregiver": "<Figurname>", '
    '"thresholds": [{"at": 70, "need": "feed_baby", "urgency": 1}, '
    '{"at": 100, "need": "baby_sick", "urgency": 3}]}], "note": "kurz"}\n'
    "Wenn nichts zu simulieren ist: {\"needs\": []}. Antworte NUR mit dem JSON."
)

# Cheap pre-filter: only spend an LLM call when the text hints at a dependent.
_DEPENDENT_HINTS: tuple[str, ...] = (
    "baby", "saeugling", "säugling", "neugeboren", "schwanger", "geboren",
    "kind", "tochter", "sohn", "kleinkind", "windel", "stillen", "wiege",
)


def looks_like_has_dependent(text: str) -> bool:
    """True if ``text`` mentions something that might need simulating."""
    low = (text or "").lower()
    return any(h in low for h in _DEPENDENT_HINTS)


class SceneDirector:
    """Proposes simulation needs from a character + recent narrative."""

    async def analyze(
        self,
        *,
        persona: str = "",
        scenario: str = "",
        relationships: str = "",
        recent_text: str = "",
        llm_chat: LLMChat,
        brain: str = "e4b",
    ) -> dict[str, Any]:
        """Return ``{"needs": [...], "note": str}`` (needs may be empty)."""
        source = (
            f"Persona: {persona}\n"
            f"Szenario: {scenario}\n"
            f"Beziehungen: {relationships}\n"
            f"Letzte Ereignisse:\n{recent_text}"
        )
        data = await FactExtractor().extract(
            source, _INSTRUCTION, llm_chat, brain=brain, max_tokens=500
        )
        needs = self._sanitise_needs(data.get("needs"))
        log.info("scene_director.analyzed", proposed=len(needs))
        return {"needs": needs, "note": str(data.get("note", ""))[:200]}

    @staticmethod
    def _sanitise_needs(raw: Any) -> list[dict[str, Any]]:
        """Validate/coerce the LLM's needs so define_need can never choke."""
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity", "") or "").strip()
            attribute = str(item.get("attribute", "") or "").strip()
            if not entity or not attribute:
                continue
            try:
                rate = float(item.get("rate_per_minute", 0) or 0)
            except (TypeError, ValueError):
                rate = 0.0
            thresholds: list[dict[str, Any]] = []
            for t in item.get("thresholds") or []:
                if not isinstance(t, dict):
                    continue
                need = str(t.get("need", "") or "").strip()
                if not need or "at" not in t:
                    continue
                try:
                    at = float(t.get("at"))
                except (TypeError, ValueError):
                    continue
                try:
                    urgency = int(t.get("urgency", 1) or 1)
                except (TypeError, ValueError):
                    urgency = 1
                thresholds.append({"at": at, "need": need, "urgency": urgency})
            if not thresholds:
                continue
            out.append(
                {
                    "entity": entity,
                    "attribute": attribute,
                    "rate_per_minute": rate,
                    "caregiver": str(item.get("caregiver", "") or "").strip(),
                    "thresholds": thresholds,
                }
            )
        return out
