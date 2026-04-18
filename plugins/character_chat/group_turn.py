"""
Group-turn orchestrator for the character_chat plugin.

The **sequential prompting** pattern implemented here is the critical design
decision the user called out: in a group RP with multiple characters, each
speaker must see the previous speakers' turns in the *same round* before
producing their own. A round-robin prompt with "all characters answer at
once" loses the conversational physics.

Round layout
------------
A round starts with either a user message OR a proactive pulse, then each
selected speaker produces one turn. Every speaker's prompt includes:

    system:  their own card.build_system_prompt() with peers as context
    user:    - session history tail (shared)
             - current trigger (user message or pulse)
             - previous speakers' turns in THIS round
             - "Dein Beitrag jetzt (oder [PASS] wenn du schweigen willst):"

Speakers can answer ``[PASS]`` (or produce an empty line) to stay silent —
that keeps the "automatic reaction with option to stay silent" contract
from Decision #5. Direct addressing via ``@Name`` forces that character to
the front of the speaker list and bypasses the LLM-picker.

The orchestrator is deliberately transport-agnostic: it yields
:class:`CharacterTurn` objects and leaves persistence + WS-broadcasting to
the caller (the plugin). That makes it cheap to unit-test with a fake LLM
callable.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .character_card import CharacterCard, _AGE_STAGE_GUIDANCE
from .context_budget import ContextBudget, Priority, PromptSection


log = logging.getLogger(__name__)


# ─── Data types ──────────────────────────────────────────────────────────────


@dataclass
class CharacterTurn:
    """One character's contribution in a round."""

    character_id: str
    character_name: str
    content: str
    skipped: bool = False  # True if the character answered [PASS] or was empty
    # 0-based index within the round; useful for UI/tests.
    order: int = 0


@dataclass
class GroupTurnRequest:
    """A single round's input."""

    session_id: str
    # The chronological tail of the session as {"role": "user"/"assistant"/"character",
    # "name": <display name>, "content": <text>} dicts. The orchestrator does NOT
    # fetch these itself — it's handed them so tests can inject fakes.
    history: list[dict[str, Any]]
    # All characters currently bound to the session (including Lexy if Lexy is
    # treated as a character). The orchestrator won't invent cards.
    characters: list[CharacterCard]
    # The user's new message. Empty string means "no user message this round"
    # (proactive pulse or spontaneous group turn).
    user_message: str = ""
    # If the round was kicked off by a proactive pulse, this is the character
    # whose pulse triggered it (e.g. the baby crying). They've already "spoken"
    # via the pulse text, so they're skipped from the speaker list.
    pulse_from_id: str = ""
    pulse_text: str = ""
    # Optional scene description that gets threaded into each system prompt.
    scene: str = ""


@dataclass
class GroupTurnResult:
    """Output of one round."""

    turns: list[CharacterTurn] = field(default_factory=list)
    speaker_order: list[str] = field(default_factory=list)  # character_ids
    # Echoed from the request so downstream knows how to log the round.
    user_message: str = ""
    pulse_from_id: str = ""
    pulse_text: str = ""


# ─── Typing helpers ──────────────────────────────────────────────────────────

# The LLM callable is abstracted so tests don't need a real network path.
# Signature: (messages, brain, max_tokens, temperature) -> str.
LLMChat = Callable[..., Awaitable[str]]

# Optional per-character memory fetch. Called as
# ``recall_fn(character_id=..., query=..., limit=...)`` just before a turn's
# LLM call so the character's own remembered snippets can be threaded into
# their prompt. ``None`` disables the feature.
MemoryRecallFn = Callable[..., Awaitable[list[dict[str, Any]]]]


# ─── Orchestrator ────────────────────────────────────────────────────────────


class GroupTurnOrchestrator:
    """Drives one round of character turns with sequential prompting."""

    # Characters who want to stay silent answer [PASS]. We accept a few
    # variants because LLMs are creative.
    _PASS_MARKERS: tuple[str, ...] = ("[PASS]", "[pass]", "[SILENT]", "[silent]")

    def __init__(
        self,
        *,
        llm_chat: LLMChat,
        brain: str = "e4b",
        max_tokens: int = 320,
        temperature: float = 0.8,
        max_speakers_per_round: int = 4,
        turn_selection: str = "autonomous",
        recall_fn: MemoryRecallFn | None = None,
        recall_limit: int = 3,
        context_size_fn: Callable[[], int] | None = None,
        safety_margin_tokens: int = 256,
    ) -> None:
        self._llm_chat = llm_chat
        self._brain = brain
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_speakers = max_speakers_per_round
        if turn_selection not in ("autonomous", "round_robin"):
            raise ValueError(
                f"turn_selection must be 'autonomous' or 'round_robin', "
                f"got {turn_selection!r}"
            )
        self._turn_selection = turn_selection
        self._recall_fn = recall_fn
        self._recall_limit = max(0, int(recall_limit))
        # Context-size callback. Called *per turn* so a config reload or a
        # brain switch is picked up without any orchestrator rebuild. The
        # default (16384) matches the current Gemma 4 brains — keeps tests
        # working when they don't care about budgets.
        self._context_size_fn = context_size_fn or (lambda: 16384)
        self._safety_margin = max(0, int(safety_margin_tokens))

    # ─── Public entry point ──────────────────────────────────────────────

    async def run_round(self, req: GroupTurnRequest) -> GroupTurnResult:
        """Pick speakers, run each of their turns sequentially, return result."""
        if not req.characters:
            return GroupTurnResult(
                user_message=req.user_message,
                pulse_from_id=req.pulse_from_id,
                pulse_text=req.pulse_text,
            )

        eligible = [c for c in req.characters if not c.archived]
        if req.pulse_from_id:
            eligible = [c for c in eligible if c.id != req.pulse_from_id]
        if not eligible:
            return GroupTurnResult(
                user_message=req.user_message,
                pulse_from_id=req.pulse_from_id,
                pulse_text=req.pulse_text,
            )

        forced = _parse_at_mentions(req.user_message, req.characters)
        speaker_order = await self._pick_speakers(
            req=req, eligible=eligible, forced=forced
        )
        speaker_order = speaker_order[: self._max_speakers]

        by_id = {c.id: c for c in req.characters}
        turns: list[CharacterTurn] = []
        for idx, char_id in enumerate(speaker_order):
            card = by_id.get(char_id)
            if card is None:
                continue
            turn = await self._run_single_turn(
                card=card,
                order=idx,
                previous_turns=turns,
                req=req,
                all_cards=req.characters,
            )
            turns.append(turn)

        return GroupTurnResult(
            turns=turns,
            speaker_order=speaker_order,
            user_message=req.user_message,
            pulse_from_id=req.pulse_from_id,
            pulse_text=req.pulse_text,
        )

    # ─── Speaker selection ───────────────────────────────────────────────

    async def _pick_speakers(
        self,
        *,
        req: GroupTurnRequest,
        eligible: list[CharacterCard],
        forced: list[str],
    ) -> list[str]:
        """Decide who speaks in this round and in what order.

        Priority:
            1. If the user @-mentioned a character, that character goes first.
            2. If ``turn_selection == "round_robin"``, remaining characters
               follow in alphabetical order by name.
            3. Otherwise, ask the LLM for an ordered list.
        """
        # Deduplicate but preserve insertion order.
        seen: set[str] = set()
        order: list[str] = []

        for mention in forced:
            if mention in seen:
                continue
            seen.add(mention)
            order.append(mention)

        if self._turn_selection == "round_robin" or len(eligible) == 1:
            rest = sorted(
                (c for c in eligible if c.id not in seen),
                key=lambda c: c.name.lower(),
            )
            order.extend(c.id for c in rest)
            return order

        # autonomous: LLM picks order over the remaining candidates.
        remaining = [c for c in eligible if c.id not in seen]
        if not remaining:
            return order
        llm_order = await self._ask_llm_for_order(req=req, candidates=remaining)
        order.extend(llm_order)
        # Safety net: if LLM picked nothing, fall back to round-robin.
        if not llm_order and not order:
            order = [c.id for c in remaining]
        return order

    async def _ask_llm_for_order(
        self,
        *,
        req: GroupTurnRequest,
        candidates: list[CharacterCard],
    ) -> list[str]:
        roster = "\n".join(
            f"- {c.name} (id={c.id}): {_brief_persona(c)}" for c in candidates
        )
        trigger = req.user_message.strip() or (
            f"*{req.pulse_text}*" if req.pulse_text else ""
        )
        if not trigger:
            trigger = "(Gruppendynamik ohne neuen User-Impuls — wer spricht jetzt?)"

        history_blurb = _format_history_tail(req.history, limit=4)

        system = (
            "Du bist der Turn-Orchestrator einer RP-Gruppe. Deine Aufgabe: "
            "entscheiden, welche Charaktere auf den aktuellen Impuls reagieren "
            "und in welcher Reihenfolge. Keine Erfindungen — nur IDs aus der "
            "Liste. Antworte AUSSCHLIESSLICH mit IDs, komma-separiert, in der "
            "Reihenfolge in der sie sprechen sollen. Wenn jemand schweigen "
            "würde, lass sie/ihn einfach weg. Maximal "
            f"{self._max_speakers} Charaktere."
        )
        user = (
            f"## Roster\n{roster}\n\n"
            f"## Letzte Zeilen\n{history_blurb}\n\n"
            f"## Aktueller Impuls\n{trigger}\n\n"
            "## Deine Antwort (IDs komma-separiert, keine Erklärung):"
        )

        try:
            # Speaker-order is a deterministic classification task, not
            # a reasoning task. a4b's default thinking=true would waste
            # its 120-token budget on CoT and return an empty list →
            # fallback to round-robin and we lose the LLM's judgement.
            raw = await self._llm_chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                brain=self._brain,
                max_tokens=120,
                temperature=0.3,
                thinking=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("character_chat.order_llm_failed: %s", exc)
            return [c.id for c in candidates]

        valid_ids = {c.id: c.id for c in candidates}
        # Also map name→id so the LLM can reply with names if it can't keep
        # the ids straight (happens with tiny models).
        valid_ids.update({c.name.lower(): c.id for c in candidates})

        picks: list[str] = []
        for tok in re.split(r"[,\n;]+", raw):
            tok = tok.strip().strip("[]()<>\"' \t`")
            if not tok:
                continue
            # Tolerate "1. luna" → "luna"
            tok = re.sub(r"^\d+[\.\)\-:\s]+", "", tok).strip()
            resolved = valid_ids.get(tok) or valid_ids.get(tok.lower())
            if resolved and resolved not in picks:
                picks.append(resolved)
        return picks

    # ─── One character's turn ────────────────────────────────────────────

    async def _run_single_turn(
        self,
        *,
        card: CharacterCard,
        order: int,
        previous_turns: list[CharacterTurn],
        req: GroupTurnRequest,
        all_cards: list[CharacterCard],
    ) -> CharacterTurn:
        """Build the prompt for ``card`` and call the LLM exactly once.

        Internally this now routes through :class:`ContextBudget` so the
        prompt is trimmed to fit under the brain's current ``context_size``
        (queried live via ``context_size_fn`` — no hardcoded 16K).
        """
        # Fetch this character's own memory snippets (strict isolation).
        own_memories: list[dict[str, Any]] = []
        if self._recall_fn is not None and self._recall_limit > 0:
            # Semantic query: the trigger that's driving this turn.
            query = (
                req.user_message.strip()
                or req.pulse_text.strip()
                or card.name
            )
            try:
                own_memories = await self._recall_fn(
                    character_id=card.id,
                    query=query,
                    limit=self._recall_limit,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "character_chat.recall_failed: %s (char=%s)",
                    exc,
                    card.name,
                )

        # Build named sections for this turn, then fit them under budget.
        sections = self._build_turn_sections(
            card=card,
            req=req,
            all_cards=all_cards,
            previous_turns=previous_turns,
            own_memories=own_memories,
        )
        budget = ContextBudget(
            context_size=self._context_size_fn(),
            max_output_tokens=self._max_tokens,
            safety_margin=self._safety_margin,
        )
        fitted, trim_log = budget.fit_sections(sections)
        if trim_log:
            log.info(
                "character_chat.context_trimmed character=%s "
                "context_size=%d available=%d trims=%s",
                card.name,
                budget.context_size,
                budget.available,
                trim_log,
            )

        system_prompt = _assemble_sections(fitted, role="system")
        user_content = _assemble_sections(fitted, role="user")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            # Characters don't need Chain-of-Thought — they need a fast,
            # in-voice reply. a4b has ``thinking=true`` by default which
            # eats ``max_tokens`` budget for reasoning tokens and often
            # returns empty content (→ turn marked skipped → no bubble in
            # chat). Force it off here regardless of brain config.
            raw = await self._llm_chat(
                messages=messages,
                brain=self._brain,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                thinking=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "character_chat.turn_llm_failed: %s (char=%s)", exc, card.name
            )
            return CharacterTurn(
                character_id=card.id,
                character_name=card.name,
                content="",
                skipped=True,
                order=order,
            )

        content = (raw or "").strip()
        skipped = False
        if not content or self._is_pass(content):
            skipped = True
            content = ""

        return CharacterTurn(
            character_id=card.id,
            character_name=card.name,
            content=content,
            skipped=skipped,
            order=order,
        )

    # ─── Section-based prompt building (context-budget aware) ────────────

    # System section names used during reassembly. Order matters: it defines
    # the sequence the system prompt is built in.
    _SYSTEM_SECTION_ORDER: tuple[str, ...] = (
        "identity",
        "persona",
        "scenario",
        "age_guidance",
        "others",
        "example_dialog",
        "rules",
    )
    # User section order (fed as the user message for the turn).
    _USER_SECTION_ORDER: tuple[str, ...] = (
        "history",
        "memory",
        "pulse",
        "user_message",
        "prev_turns",
        "instruction",
    )

    def _build_turn_sections(
        self,
        *,
        card: CharacterCard,
        req: GroupTurnRequest,
        all_cards: list[CharacterCard],
        previous_turns: list[CharacterTurn],
        own_memories: list[dict[str, Any]] | None,
    ) -> list[PromptSection]:
        """Split the prompt into named, priority-tagged sections.

        The budget manager rearranges / trims these so the final prompt
        fits under the brain's context window. Sections not in the budget
        assembly (e.g. dropped LOW ones) reassemble to empty strings.
        """
        # ─── SYSTEM sections ──────────────────────────────────────────
        sections: list[PromptSection] = []
        sections.append(
            PromptSection(
                name="identity",
                priority=Priority.MUST,
                text=f"Du bist {card.name}.",
                role="system",
            )
        )
        persona = (card.persona or "").strip()
        if persona:
            sections.append(
                PromptSection(
                    name="persona",
                    priority=Priority.HIGH,
                    text=f"## Persona\n{persona}",
                    role="system",
                    # Hard upper cap — ST cards can legitimately reach 5k tok.
                    # 1500 keeps the character's voice while leaving room.
                    max_tokens=1500,
                )
            )

        scenario_text = ""
        if (card.scenario or "").strip():
            scenario_text = f"## Szenario\n{card.scenario.strip()}"
        elif (req.scene or "").strip():
            scenario_text = f"## Szene\n{req.scene.strip()}"
        if scenario_text:
            sections.append(
                PromptSection(
                    name="scenario",
                    priority=Priority.MEDIUM,
                    text=scenario_text,
                    role="system",
                    max_tokens=500,
                )
            )

        age = _AGE_STAGE_GUIDANCE.get(card.age_stage, "")
        if age:
            sections.append(
                PromptSection(
                    name="age_guidance",
                    priority=Priority.MEDIUM,
                    text=f"## Alter/Entwicklung\n{age}",
                    role="system",
                )
            )

        others_text = _format_others_block(card, all_cards)
        if others_text:
            sections.append(
                PromptSection(
                    name="others",
                    priority=Priority.LOW,
                    text=others_text,
                    role="system",
                    max_tokens=200,
                )
            )

        example = (card.example_dialog or "").strip()
        if example:
            sections.append(
                PromptSection(
                    name="example_dialog",
                    priority=Priority.LOW,
                    text=f"## Beispiel-Dialog\n{example}",
                    role="system",
                    max_tokens=500,
                )
            )

        sections.append(
            PromptSection(
                name="rules",
                priority=Priority.MUST,
                text=(
                    "## Regeln\n"
                    "- Antworte AUSSCHLIESSLICH in deiner eigenen Stimme.\n"
                    "- Kein Meta-Kommentar, keine Regie-Anweisungen in "
                    "eckigen Klammern (außer kurze *Aktionen* wenn es die "
                    "Szene trägt).\n"
                    "- Halte dich kurz (1-4 Sätze), außer die Szene verlangt "
                    "mehr.\n"
                    "- Sprich andere Anwesende gegebenenfalls namentlich an."
                ),
                role="system",
            )
        )

        # ─── USER sections ────────────────────────────────────────────
        history_text = _format_history_tail(req.history, limit=6)
        if history_text:
            # Snapshot req.history so the reduce_fn closure stays pure.
            history_ref = req.history

            def _reduce_history(step: int, h: list[dict[str, Any]] = history_ref) -> str:
                trimmed = _format_history_tail(h, limit=step)
                return f"## Bisheriger Chat\n{trimmed}" if trimmed else ""

            sections.append(
                PromptSection(
                    name="history",
                    priority=Priority.MEDIUM,
                    text=f"## Bisheriger Chat\n{history_text}",
                    role="user",
                    reduce_fn=_reduce_history,
                    reduce_steps=[4, 2],
                )
            )

        memory_text = _format_memory_tail(own_memories)
        if memory_text:
            memory_ref: list[dict[str, Any]] = list(own_memories or [])

            def _reduce_memory(step: int, mems: list[dict[str, Any]] = memory_ref) -> str:
                trimmed = _format_memory_tail(mems[:step])
                return (
                    f"## Deine Erinnerungen (was du selbst erlebt hast)\n"
                    f"{trimmed}"
                    if trimmed
                    else ""
                )

            sections.append(
                PromptSection(
                    name="memory",
                    priority=Priority.MEDIUM,
                    text=(
                        "## Deine Erinnerungen (was du selbst erlebt hast)\n"
                        f"{memory_text}"
                    ),
                    role="user",
                    reduce_fn=_reduce_memory,
                    reduce_steps=[2, 1],
                )
            )

        if req.pulse_text and req.pulse_from_id:
            pulse_name = _resolve_name(
                req.pulse_from_id,
                [card, *previous_turns_as_cards(previous_turns)],
            )
            pulse_body = (
                f"## Impuls\n*{pulse_name}* {req.pulse_text}"
                if pulse_name
                else f"## Impuls\n{req.pulse_text}"
            )
            sections.append(
                PromptSection(
                    name="pulse",
                    priority=Priority.HIGH,
                    text=pulse_body,
                    role="user",
                )
            )

        if req.user_message.strip():
            sections.append(
                PromptSection(
                    name="user_message",
                    priority=Priority.HIGH,
                    text=f"## User (Mike)\n{req.user_message.strip()}",
                    role="user",
                    max_tokens=800,
                )
            )

        if previous_turns:
            lines: list[str] = []
            for t in previous_turns:
                if t.skipped:
                    lines.append(f"*{t.character_name} schweigt*")
                else:
                    lines.append(f"**{t.character_name}:** {t.content}")
            sections.append(
                PromptSection(
                    name="prev_turns",
                    priority=Priority.HIGH,
                    text="## Reaktionen dieser Runde\n" + "\n".join(lines),
                    role="user",
                )
            )

        sections.append(
            PromptSection(
                name="instruction",
                priority=Priority.MUST,
                text=(
                    f"## Du bist jetzt dran: {card.name}\n"
                    "Antworte in deiner Stimme (1-4 Sätze, außer die Szene "
                    "verlangt mehr).\n"
                    "Schreib ausschließlich deine Worte/Aktionen — keine "
                    "Regie für andere.\n"
                    "Wenn du nichts beitragen willst, antworte exakt: [PASS]"
                ),
                role="user",
            )
        )

        return sections

    def _is_pass(self, text: str) -> bool:
        stripped = text.strip().strip(" .!")
        return stripped in self._PASS_MARKERS or stripped.upper() == "[PASS]"


# ─── Module-level helpers ────────────────────────────────────────────────────


# "@Luna hey!" or "@luna" → forces Luna into position 0. Matches @Word.
_AT_MENTION_RE = re.compile(r"@([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_\-]{1,40})")


def _parse_at_mentions(
    text: str, characters: list[CharacterCard]
) -> list[str]:
    """Return character ids named via ``@Name`` in ``text``, in order."""
    if not text or "@" not in text:
        return []
    name_to_id = {c.name.lower(): c.id for c in characters}
    picks: list[str] = []
    for match in _AT_MENTION_RE.finditer(text):
        candidate = match.group(1).lower()
        char_id = name_to_id.get(candidate)
        if char_id and char_id not in picks:
            picks.append(char_id)
    return picks


def _brief_persona(card: CharacterCard, limit: int = 120) -> str:
    text = (card.persona or "").strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    if not text:
        text = f"(Stage: {card.age_stage})"
    return text


def _format_history_tail(
    history: list[dict[str, Any]], *, limit: int
) -> str:
    if not history:
        return ""
    tail = history[-limit:]
    lines: list[str] = []
    for entry in tail:
        role = str(entry.get("role", "user"))
        name = str(entry.get("name") or ("Du" if role == "user" else "Lexy"))
        content = str(entry.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"{name}: {content}")
        else:
            lines.append(f"**{name}:** {content}")
    return "\n".join(lines)


def _format_others_block(
    card: CharacterCard, all_cards: list[CharacterCard]
) -> str:
    """Render the '## Andere Anwesende' block with relationship hints."""
    lines: list[str] = []
    for other in all_cards or []:
        if other.id == card.id:
            continue
        rel = card.relationships.get(other.id) or other.relationships.get(
            card.id
        )
        if rel:
            lines.append(f"- {other.name}: {rel}")
        else:
            lines.append(f"- {other.name}")
    if not lines:
        return ""
    return "## Andere Anwesende\n" + "\n".join(lines)


def _assemble_sections(
    sections: list[PromptSection], *, role: str
) -> str:
    """Reassemble non-empty sections of ``role`` into one message string.

    Sections are joined with a blank line so the resulting prompt stays
    readable. Section **order** follows the orchestrator's fixed layout —
    we iterate through ``sections`` in the order they were built, which
    matches the `_SYSTEM_SECTION_ORDER` / `_USER_SECTION_ORDER` layout
    the method produces.
    """
    parts: list[str] = []
    for s in sections:
        if s.role != role:
            continue
        if not s.text:
            continue
        parts.append(s.text)
    return "\n\n".join(parts)


def _format_memory_tail(memories: list[dict[str, Any]] | None) -> str:
    """Shape memory-recall hits into a compact bullet list for prompts."""
    if not memories:
        return ""
    lines: list[str] = []
    for item in memories:
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        # Long memory snippets would blow up the prompt — truncate aggressively.
        if len(content) > 220:
            content = content[:220].rstrip() + "…"
        lines.append(f"- {content}")
    return "\n".join(lines)


def _resolve_name(
    character_id: str, cards: list[CharacterCard]
) -> str:
    for c in cards:
        if c.id == character_id:
            return c.name
    return ""


def previous_turns_as_cards(turns: list[CharacterTurn]) -> list[CharacterCard]:
    """Build lightweight stand-in cards from turns (for name lookup only)."""
    out: list[CharacterCard] = []
    for t in turns:
        try:
            out.append(
                CharacterCard(
                    id=t.character_id,
                    name=t.character_name,
                )
            )
        except Exception:  # noqa: BLE001
            continue
    return out


# Re-export asyncio just so `from .group_turn import asyncio` in tests works
# on the off-chance someone wants to schedule tasks against the orchestrator.
__all__ = [
    "CharacterTurn",
    "GroupTurnOrchestrator",
    "GroupTurnRequest",
    "GroupTurnResult",
    "LLMChat",
    "asyncio",
]
