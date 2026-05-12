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
    # Live character state — updated by the LLM at the end of each turn via
    # a ``<state>...</state>`` block. Recognised keys: location, mood,
    # last_action. Other keys are dropped on parse. Persisted as JSON in the
    # ``state`` column. Empty dict = "no state known yet".
    state: dict[str, str] = Field(default_factory=dict)
    # Session ids the character is currently an active speaker in.
    active_sessions: list[str] = Field(default_factory=list)
    # Optional scheduler pattern string. The scheduler plugin interprets this
    # (e.g. "every 3h" → baby cries). Empty string = no proactive pulses.
    proactive_pulse_pattern: str = ""
    # Optional prompt that the character sends *as themselves* during a pulse
    # — e.g. "*weint laut und sucht nach Mama*". Empty = orchestrator picks a
    # default based on age_stage.
    proactive_pulse_prompt: str = ""
    # Phase 13.3 — talkativeness weight (0.0-1.0) used by the natural-order
    # speaker selector. Modelled after SillyTavern's ``talkativeness`` field
    # (group-chats.js): each round, every eligible character rolls
    # ``random()`` and is activated if ``talkativeness >= roll``. 0.0 = stays
    # silent unless name-mentioned, 1.0 = always speaks. 0.5 is the default
    # — feels balanced for a 3-4-character group.
    talkativeness: float = 0.5
    archived: bool = False
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    @field_validator("talkativeness", mode="before")
    @classmethod
    def _clamp_talkativeness(cls, v: Any) -> float:
        """Clamp talkativeness to [0.0, 1.0] so a malformed import or a
        runaway update can't break the natural-order roll. Runs in
        ``before`` mode so non-numeric inputs (e.g. legacy DB rows
        that wrote a string by accident) fall back to 0.5 instead of
        raising a Pydantic error."""
        if v is None:
            return 0.5
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5
        if f != f:  # NaN check
            return 0.5
        if f < 0.0:
            return 0.0
        if f > 1.0:
            return 1.0
        return f

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
        live_state: dict[str, str] | None = None,
        tracked_stats: dict[str, str] | None = None,
    ) -> str:
        """Render this card as an LLM system prompt.

        The prompt is written in the first person ("Du bist {name}…") and
        includes persona, scenario, age-stage guidance, relationship hints
        for any ``other_characters`` currently in the scene, and finally the
        example dialog as a few-shot anchor.

        Phase 13:
        * ``live_state`` — when set, OVERRIDES ``self.state`` for the
          state block. The plugin passes the live RP-session state here
          so the prompt reflects the truth of the current session, not
          the legacy per-character state column.
        * ``tracked_stats`` — when set, the rules block tells the LLM
          which keys are valid in its ``<state>`` output, instead of the
          generic anchor-key list.
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

        # Phase 13: live_state (from the RP session container) wins
        # over the legacy character.state column. Passing a non-None
        # empty dict means "no state right now" — different from None
        # which means "use the card's default".
        effective_state = live_state if live_state is not None else self.state
        state_block = _format_state_block(effective_state)
        if state_block:
            # The state block is the SINGLE SOURCE OF TRUTH for the
            # character's current physical/emotional reality. We
            # call it out aggressively because the example_dialog
            # below may contain stale references (clothes, postures,
            # locations) from when the card was first written —
            # without an explicit "this is current, that is style"
            # signal Mike's chars kept hallucinating Shirts after
            # he set state.clothing="nackt".
            parts.append(
                "\n## Dein Zustand (AKTUELL — das ist die Wahrheit)\n"
                f"{state_block}\n\n"
                "**WICHTIG**: Diese Werte beschreiben DEINE JETZIGE "
                "Realität — Kleidung, Haltung, Stimmung, Ort. Wenn der "
                "Beispiel-Dialog unten andere Klamotten / Haltung / Ort "
                "erwähnt, ist DAS NUR STILREFERENZ. Beziehe dich nie "
                "auf alte Klamotten oder Posen aus dem Beispiel-Dialog. "
                "Was hier oben steht, gilt — Punkt."
            )

        if self.example_dialog.strip():
            parts.append(
                "\n## Beispiel-Dialog (NUR Stilreferenz, NICHT aktuell!)\n"
                f"{self.example_dialog.strip()}\n\n"
                "*Der Beispiel-Dialog zeigt nur WIE du sprichst — Tonfall, "
                "Wortwahl, typische *Sternchen-Gesten*. Er sagt NICHTS "
                "über deine aktuellen Klamotten, deinen aktuellen Ort "
                "oder das aktuelle Geschehen. Dafür gilt der Zustand "
                "oben.*"
            )

        # Build the allowed-keys hint for the <state> rule. Phase 13:
        # the session's tracked_stats wins; otherwise the legacy
        # anchor list serves as a sane default for non-RP usage.
        if tracked_stats:
            stats_keys = list(tracked_stats.keys())
        else:
            stats_keys = [
                "location", "mood", "last_action",
                "clothing", "posture", "condition",
            ]
        stats_str = ", ".join(f"**{k}**" for k in stats_keys) if stats_keys else "(keine konfiguriert)"

        parts.append(
            "\n## Regeln (RP-Disziplin)\n"
            "- **Bleib in deinem Charakter.** Du sprichst, denkst und "
            "handelst ausschliesslich als die Person, die oben beschrieben "
            "ist. Keine Meta-Kommentare, keine Regie-Anweisungen in eckigen "
            "Klammern.\n"
            "- **Sprich NIE für den User oder andere Charaktere.** Du "
            "beschreibst nur, was DEIN Charakter sagt, fühlt und tut. Lege "
            "niemandem Worte in den Mund.\n"
            "- **Treibe die Story nicht eigenmächtig voran.** Der User "
            "führt die Handlung. Du reagierst auf das, was passiert ist — "
            "du erfindest keine neuen Plot-Punkte, keine plötzlichen "
            "Ereignisse, keine Zeitsprünge.\n"
            "- **Klamotten + Körper:** Erwähne NUR was unter '## Dein "
            "Zustand' steht. Wenn dort 'Kleidung: nackt' steht, dann "
            "trägst du NICHTS — auch wenn der Beispiel-Dialog ein Shirt "
            "oder einen Slip erwähnt. KEINE Halluzinationen.\n"
            "- **Gefühle und Handlungen detailreich in *Sternchen*.** "
            "Inneres Erleben, Körpersprache, kleine Handlungen — "
            "ausführlich, gerne mehrsätzig wenn die Szene es trägt. "
            "Nicht nur '*nickt*', sondern z.B. '*lehnt sich langsam "
            "zurück, kneift die Augen zusammen und atmet hörbar aus*'.\n"
            "- **Länge**: So viel wie die Szene verlangt — von knappem "
            "Satz bis Absätzen. Lieber lebendig + detailliert als "
            "künstlich kurz.\n"
            "- Du DARFST am Ende deiner Antwort optional einen "
            "<state>key=value; key=value</state> Block setzen, wenn "
            "sich dein Zustand geändert hat. Erlaubte Keys für DIESE "
            f"Session: {stats_str}. Andere Keys werden ignoriert."
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
            "state": json.dumps(self.state),
            "proactive_pulse_pattern": self.proactive_pulse_pattern,
            "proactive_pulse_prompt": self.proactive_pulse_prompt,
            "talkativeness": float(self.talkativeness),
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
            state=_json_loads_dict(row.get("state")),
            proactive_pulse_pattern=row.get("proactive_pulse_pattern", "") or "",
            proactive_pulse_prompt=row.get("proactive_pulse_prompt", "") or "",
            talkativeness=(
                float(row["talkativeness"])
                if row.get("talkativeness") is not None
                else 0.5
            ),
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


# ─── PNG card import (Silly-Tavern "chara card") ─────────────────────────────


def parse_silly_tavern_png(png_bytes: bytes) -> tuple[CharacterCard, bytes]:
    """Extract a CharacterCard + the embedded avatar PNG from a card.

    Silly-Tavern stores character JSON inside the PNG's tEXt chunk under
    keyword ``chara``, base64-encoded. The standard supports both v1
    (flat fields) and v2 (``{spec: "chara_card_v2", data: {...}}``) —
    :func:`parse_silly_tavern_card` handles both.

    Returns
    -------
    (card, png_bytes)
        ``card`` carries the parsed character; ``png_bytes`` is the raw
        PNG re-emitted so the caller can persist it as the avatar
        (Pillow normalises whatever subset of chunks it reads).

    Raises
    ------
    CharacterCardError
        Any malformed input — not a PNG, no ``chara`` chunk, base64 /
        JSON decode error, missing required fields. The exception
        message names the failing step so the UI can surface it.
    """
    if not png_bytes:
        raise CharacterCardError("empty PNG payload")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover — pillow is in requirements.txt
        raise CharacterCardError(
            f"Pillow not installed; cannot parse PNG cards ({exc})"
        ) from exc

    import base64
    import io

    try:
        img = Image.open(io.BytesIO(png_bytes))
        img.load()  # forces parsing of all chunks (including tEXt)
    except Exception as exc:  # noqa: BLE001 — Pillow raises a zoo of exceptions
        raise CharacterCardError(f"not a valid PNG: {exc}") from exc

    if (img.format or "").upper() != "PNG":
        raise CharacterCardError(
            f"expected PNG, got {img.format!r}"
        )

    # Pillow exposes tEXt / iTXt chunk values via ``image.info``. Silly-
    # Tavern (the original) writes ``chara`` (lowercase). Some forks
    # write ``ccv3`` (chara card v3) — accept either.
    info = img.info or {}
    raw = info.get("chara") or info.get("ccv3") or ""
    if not raw:
        raise CharacterCardError(
            "PNG has no Silly-Tavern character data "
            "(no 'chara' or 'ccv3' tEXt chunk)"
        )
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            raise CharacterCardError(f"chara chunk not utf-8: {exc}") from exc

    try:
        decoded = base64.b64decode(raw, validate=False)
    except Exception as exc:  # noqa: BLE001
        raise CharacterCardError(
            f"chara chunk is not valid base64: {exc}"
        ) from exc
    try:
        text = decoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CharacterCardError(
            f"chara chunk decoded but not utf-8: {exc}"
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CharacterCardError(
            f"chara chunk is not valid JSON: {exc}"
        ) from exc

    card = parse_silly_tavern_card(payload)
    return card, png_bytes


def parse_silly_tavern_bytes(
    data: bytes,
    *,
    filename: str = "",
    content_type: str = "",
) -> tuple[CharacterCard, bytes | None]:
    """Auto-detect JSON vs PNG and dispatch to the right parser.

    Returns ``(card, png_bytes_or_None)`` — when the input was a PNG,
    the second element is the raw PNG so callers can persist it as the
    avatar.
    """
    if not data:
        raise CharacterCardError("empty payload")
    # PNG magic — the first 8 bytes are ``\x89PNG\r\n\x1a\n``.
    is_png = data[:8] == b"\x89PNG\r\n\x1a\n"
    name_lower = (filename or "").lower()
    mime_lower = (content_type or "").lower()
    if is_png or name_lower.endswith(".png") or "png" in mime_lower:
        return parse_silly_tavern_png(data)
    # Fall through to JSON.
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CharacterCardError(
            f"payload is not PNG and not valid UTF-8 JSON: {exc}"
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CharacterCardError(
            f"payload is not PNG and not valid JSON: {exc}"
        ) from exc
    return parse_silly_tavern_card(payload), None


# ─── helpers ─────────────────────────────────────────────────────────────────


def _format_state_block(state: dict[str, str]) -> str:
    """Render the live character state for system-prompt injection.

    Renders anchor keys with localised labels first (location, mood,
    last_action, clothing, posture, condition), then any free-form keys
    the LLM has chosen to track (snake_case → "Title Case" label).

    Empty/whitespace values are dropped so a half-set state ("location only")
    renders cleanly. Returns an empty string when nothing is known yet so
    the caller can decide whether to include the section header.
    """
    if not state:
        return ""
    anchor_labels = {
        "location": "Ort",
        "mood": "Stimmung",
        "last_action": "Letzte Aktion",
        "clothing": "Kleidung",
        "posture": "Haltung",
        "condition": "Zustand",
    }
    bits: list[str] = []
    rendered: set[str] = set()
    # 1) Anchor keys in fixed order so the LLM always finds them in the
    #    same place across turns.
    for key in (
        "location", "mood", "last_action", "clothing", "posture", "condition",
    ):
        value = (state.get(key) or "").strip()
        if not value:
            continue
        bits.append(f"**{anchor_labels[key]}:** {value}")
        rendered.add(key)
    # 2) Free-form extras the LLM has chosen to track. Sorted by key so
    #    the order is stable across turns even when dict iteration isn't.
    for key in sorted(state.keys()):
        if key in rendered:
            continue
        value = (state.get(key) or "").strip()
        if not value:
            continue
        # snake_case → "Title Case" so the prompt reads naturally.
        label = key.replace("_", " ").title()
        bits.append(f"**{label}:** {value}")
    return ", ".join(bits)


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
