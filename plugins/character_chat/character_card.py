"""
Character card model for the character_chat plugin.

A ``CharacterCard`` is the persistent definition of an RP character (Lexy,
Luna, a baby agent, an NPC) — name, persona, greeting, scenario, example
dialog, avatar, age stage, relationships. It mirrors the Silly-Tavern spec
loosely enough to import their character cards directly (both v1 flat
format and v2 ``data``-nested format).

The card is *plugin-owned*; it has no coupling to the core session store or
memory system. The orchestrator turns a card into an LLM system prompt via
``build_system_prompt()``.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ─── Age stages ──────────────────────────────────────────────────────────────

# RP growth ladder. Used by proactive pulses (a baby cries on a shorter
# schedule than a toddler) and by build_system_prompt (a baby doesn't form
# full sentences).
AGE_STAGES: tuple[str, ...] = (
    "baby",
    "toddler",
    "child",
    "teen",
    "adult",
)


class CharacterCardError(ValueError):
    """Raised when a card or import payload is malformed."""


# ─── Card model ──────────────────────────────────────────────────────────────


class CharacterCard(BaseModel):
    """A persistent RP character definition.

    The field set is intentionally close to the Silly-Tavern v2 spec so cards
    can round-trip through ``parse_silly_tavern_card`` with minimal loss.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    persona: str = ""
    greeting: str = ""
    scenario: str = ""
    example_dialog: str = ""
    avatar: str = ""  # relative path under data/plugins/character_chat/avatars/
    color: str = "#7aa2f7"
    age_stage: str = "adult"
    # Optional provider-specific voice name for TTS (e.g. a CosyVoice
    # speaker id like "luna"). Empty = use the default configured TTS
    # voice. The character_chat plugin passes this to ``api.tts_speak``
    # so each character can speak in their own voice.
    voice: str = ""
    # other_character_id → free-form label ("mother", "sister", "friend")
    relationships: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    # Session ids the character is currently an active speaker in.
    active_sessions: list[str] = Field(default_factory=list)
    # Optional scheduler pattern string. The scheduler plugin interprets this
    # (e.g. "every 3h" → baby cries). Empty string = no proactive pulses.
    proactive_pulse_pattern: str = ""
    # Optional prompt that the character sends *as themselves* during a pulse
    # — e.g. "*weint laut und sucht nach Mama*". Empty = orchestrator picks a
    # default based on age_stage.
    proactive_pulse_prompt: str = ""
    archived: bool = False
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    @field_validator("age_stage")
    @classmethod
    def _validate_age_stage(cls, v: str) -> str:
        # Plain ValueError — pydantic wraps it into ValidationError. The
        # store layer re-raises wrapped errors as CharacterCardError for
        # stable external contract.
        if v not in AGE_STAGES:
            raise ValueError(
                f"age_stage must be one of {AGE_STAGES!r}, got {v!r}"
            )
        return v

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be empty")
        if len(v) > 80:
            raise ValueError("name must be ≤ 80 chars")
        return v

    @field_validator("color")
    @classmethod
    def _validate_color(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            return "#7aa2f7"
        if not v.startswith("#") or len(v) not in (4, 7):
            raise ValueError(f"color must be hex like '#7aa2f7', got {v!r}")
        return v

    # ─── Prompt rendering ────────────────────────────────────────────────

    def build_system_prompt(
        self,
        *,
        scene: str = "",
        other_characters: list["CharacterCard"] | None = None,
        extra_instructions: str = "",
    ) -> str:
        """Render this card as an LLM system prompt.

        The prompt is written in the first person ("Du bist {name}…") and
        includes persona, scenario, age-stage guidance, relationship hints
        for any ``other_characters`` currently in the scene, and finally the
        example dialog as a few-shot anchor.
        """
        parts: list[str] = []
        parts.append(f"Du bist {self.name}.")

        if self.persona.strip():
            parts.append(f"\n## Persona\n{self.persona.strip()}")

        if self.scenario.strip():
            parts.append(f"\n## Szenario\n{self.scenario.strip()}")
        elif scene.strip():
            parts.append(f"\n## Szene\n{scene.strip()}")

        stage_guidance = _AGE_STAGE_GUIDANCE.get(self.age_stage, "")
        if stage_guidance:
            parts.append(f"\n## Alter/Entwicklung\n{stage_guidance}")

        if other_characters:
            lines: list[str] = []
            for other in other_characters:
                if other.id == self.id:
                    continue
                rel = self.relationships.get(other.id) or other.relationships.get(
                    self.id
                )
                if rel:
                    lines.append(f"- {other.name}: {rel}")
                else:
                    lines.append(f"- {other.name}")
            if lines:
                parts.append("\n## Andere Anwesende\n" + "\n".join(lines))

        if self.example_dialog.strip():
            parts.append(f"\n## Beispiel-Dialog\n{self.example_dialog.strip()}")

        parts.append(
            "\n## Regeln\n"
            "- Antworte AUSSCHLIESSLICH in deiner eigenen Stimme.\n"
            "- Kein Meta-Kommentar, keine Regie-Anweisungen in eckigen Klammern "
            "(außer kurze *Aktionen* wenn es die Szene trägt).\n"
            "- Halte dich kurz (1-4 Sätze), außer die Szene verlangt mehr.\n"
            "- Sprich andere Anwesende gegebenenfalls namentlich an."
        )

        if extra_instructions.strip():
            parts.append(f"\n## Zusatz\n{extra_instructions.strip()}")

        return "\n".join(parts)

    # ─── Persistence helpers ─────────────────────────────────────────────

    def to_row(self) -> dict[str, Any]:
        """Flatten to a dict suitable for the SQLite row format."""
        return {
            "id": self.id,
            "name": self.name,
            "persona": self.persona,
            "greeting": self.greeting,
            "scenario": self.scenario,
            "example_dialog": self.example_dialog,
            "avatar": self.avatar,
            "color": self.color,
            "age_stage": self.age_stage,
            "voice": self.voice,
            "relationships": json.dumps(self.relationships),
            "tags": json.dumps(self.tags),
            "active_sessions": json.dumps(self.active_sessions),
            "proactive_pulse_pattern": self.proactive_pulse_pattern,
            "proactive_pulse_prompt": self.proactive_pulse_prompt,
            "archived": 1 if self.archived else 0,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "CharacterCard":
        """Rebuild a card from a SQLite row."""
        return cls(
            id=row["id"],
            name=row["name"],
            persona=row.get("persona", "") or "",
            greeting=row.get("greeting", "") or "",
            scenario=row.get("scenario", "") or "",
            example_dialog=row.get("example_dialog", "") or "",
            avatar=row.get("avatar", "") or "",
            color=row.get("color") or "#7aa2f7",
            age_stage=row.get("age_stage") or "adult",
            voice=row.get("voice", "") or "",
            relationships=_json_loads_dict(row.get("relationships")),
            tags=_json_loads_list(row.get("tags")),
            active_sessions=_json_loads_list(row.get("active_sessions")),
            proactive_pulse_pattern=row.get("proactive_pulse_pattern", "") or "",
            proactive_pulse_prompt=row.get("proactive_pulse_prompt", "") or "",
            archived=bool(row.get("archived", 0)),
            created_at=float(row.get("created_at") or time.time()),
            updated_at=float(row.get("updated_at") or time.time()),
        )


# ─── Age-stage guidance ──────────────────────────────────────────────────────

_AGE_STAGE_GUIDANCE: dict[str, str] = {
    "baby": (
        "Du bist ein Säugling. Du kannst nicht sprechen. Deine Kommunikation "
        "besteht aus *Lauten und Aktionen* in Sternchen: *schreit*, *gurgelt*, "
        "*sucht Mamas Brust*, *greift nach einem Gegenstand*. Keine Sätze. "
        "Höchstens einzelne Wörter wie 'mama' oder 'da' wenn du gegen Ende "
        "deiner Babyzeit bist."
    ),
    "toddler": (
        "Du bist ein Kleinkind (1-3 Jahre). Sprich in sehr einfachen kurzen "
        "Sätzen, mit Baby-Grammatik ('Ich will Apfel!', 'Mama wo?'). Nutze "
        "*Aktionen* in Sternchen wenn sinnvoll. Sei neugierig und emotional."
    ),
    "child": (
        "Du bist ein Kind (4-10 Jahre). Sprich einfach und direkt, mit "
        "kindlicher Fantasie und Fragen. Du verstehst viel, aber nicht alles. "
        "Sei spielerisch."
    ),
    "teen": (
        "Du bist ein Teenager. Du bildest dir eine Meinung, widersprichst, "
        "bist manchmal trotzig oder nachdenklich. Sprich natürlich, mit "
        "eigenem Vokabular."
    ),
    "adult": "",  # no extra guidance — defer to persona
}


# ─── Silly-Tavern import ─────────────────────────────────────────────────────


def parse_silly_tavern_card(payload: dict[str, Any]) -> CharacterCard:
    """Convert a Silly-Tavern character JSON to a :class:`CharacterCard`.

    Supports both the v1 flat format (``name``/``description``/…) and the
    v2 format where everything lives in a nested ``data`` object.
    """
    if not isinstance(payload, dict):
        raise CharacterCardError("silly-tavern payload must be a dict")

    # v2: { "spec": "chara_card_v2", "data": {...} }
    if isinstance(payload.get("data"), dict):
        data: dict[str, Any] = payload["data"]
    else:
        data = payload

    name = str(data.get("name") or "").strip()
    if not name:
        raise CharacterCardError("silly-tavern card missing 'name'")

    persona = str(
        data.get("description") or data.get("personality") or ""
    ).strip()
    scenario = str(data.get("scenario") or "").strip()
    greeting = str(
        data.get("first_mes") or data.get("first_message") or ""
    ).strip()
    example_dialog = str(data.get("mes_example") or "").strip()

    tags_raw = data.get("tags") or []
    tags: list[str] = []
    if isinstance(tags_raw, str):
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    elif isinstance(tags_raw, list):
        # Some exports store all tags as one comma-separated string inside a
        # single-element list (quirky Silly-Tavern behaviour). Split eagerly.
        for item in tags_raw:
            text = str(item).strip()
            if "," in text:
                tags.extend(t.strip() for t in text.split(",") if t.strip())
            elif text:
                tags.append(text)

    return CharacterCard(
        name=name,
        persona=persona,
        scenario=scenario,
        greeting=greeting,
        example_dialog=example_dialog,
        tags=tags,
    )


def parse_silly_tavern_file(path: Path) -> CharacterCard:
    """Read and parse a Silly-Tavern JSON character card from disk."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CharacterCardError(f"cannot read {path}: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CharacterCardError(f"{path} is not valid JSON: {exc}") from exc
    return parse_silly_tavern_card(payload)


# ─── helpers ─────────────────────────────────────────────────────────────────


def _json_loads_dict(raw: Any) -> dict[str, str]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v) for k, v in parsed.items()}


def _json_loads_list(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x) for x in parsed]
