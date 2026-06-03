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
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .character_card import CharacterCard, _AGE_STAGE_GUIDANCE, _format_state_block
from .context_budget import ContextBudget, Priority, PromptSection
from .lorebook_engine import (
    ActivationResult,
    SECTION_NAME_PER_POSITION,
    render_position_block,
)
from .mention_parser import parse_nl_mentions


log = logging.getLogger(__name__)


# ─── Prompt debug (opt-in via LEXY_DEBUG_PROMPTS) ────────────────────────────
#
# When the env var ``LEXY_DEBUG_PROMPTS`` is truthy (1/true/yes/on), the exact
# system + user content sent to the LLM — plus the raw response — is printed to
# the backend console for every character turn. Default OFF so normal runs stay
# quiet. ``configure_logging`` routes stdlib logging to stdout with a bare
# ``%(message)s`` format, so the multi-line block surfaces verbatim in the CMD.
#
# Usage (Windows CMD):  set LEXY_DEBUG_PROMPTS=1  &&  python -m lexy_core
_DEBUG_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_DEBUG_RULE = "─" * 72
_DEBUG_FRAME = "═" * 72


def _prompt_debug_enabled() -> bool:
    """Return True when LEXY_DEBUG_PROMPTS is set to a truthy value."""
    return os.environ.get("LEXY_DEBUG_PROMPTS", "").strip().lower() in _DEBUG_TRUTHY


def _emit_prompt_debug(
    *,
    character: str,
    brain: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
) -> None:
    """Print the exact prompt sent to the LLM (gated by LEXY_DEBUG_PROMPTS)."""
    if not _prompt_debug_enabled():
        return
    block = (
        f"\n{_DEBUG_FRAME}\n"
        f"PROMPT DEBUG -> character={character} brain={brain} "
        f"max_tokens={max_tokens} temperature={temperature}\n"
        f"system={len(system_prompt)} chars  user={len(user_content)} chars\n"
        f"{_DEBUG_RULE}\n"
        f"[SYSTEM]\n{system_prompt}\n"
        f"{_DEBUG_RULE}\n"
        f"[USER]\n{user_content}\n"
        f"{_DEBUG_FRAME}"
    )
    log.info(block)


def _emit_response_debug(*, character: str, content: str, skipped: bool) -> None:
    """Print the raw LLM response next to its prompt (gated by the same flag)."""
    if not _prompt_debug_enabled():
        return
    body = content if content else "(empty)"
    block = (
        f"\n{_DEBUG_RULE}\n"
        f"LLM RESPONSE <- character={character} skipped={skipped} "
        f"chars={len(content)}\n"
        f"{body}\n"
        f"{_DEBUG_FRAME}"
    )
    log.info(block)


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
    # Extra forced speakers appended AFTER any user @-/NL-mentions. The plugin
    # populates this for pulse-mention propagation: when a pulsing character
    # addresses someone (e.g. "Drell, hast du das gemacht?"), Drell ends up in
    # this list so his answer happens in the same round. Empty list = no extras.
    extra_forced: list[str] = field(default_factory=list)
    # Per-speaker lore activations, keyed by character_id. The plugin
    # populates this BEFORE calling ``run_round`` because:
    #   * the engine needs the DB-resident lore store (plugin-owned)
    #   * activations differ per speaker (character-scoped books only
    #     fire for the matching character_id; the haystack uses each
    #     speaker's persona + state).
    # An empty dict (or missing key) means "no lore for this speaker".
    lore_by_speaker: dict[str, "ActivationResult"] = field(default_factory=dict)
    # Phase 13 — per-character live state from the RP session container,
    # keyed by character_id. Overrides ``card.state`` when building the
    # prompt's state-block. Empty dict / missing key falls back to the
    # legacy ``card.state`` so non-RP code paths keep working.
    live_state_by_char: dict[str, dict[str, str]] = field(default_factory=dict)
    # Phase 13.2 — speakers to exclude from selection this round (skip-
    # cooldown). The plugin populates this with chars that returned
    # empty/pass turns recently. Speaker selection filters these out;
    # if filtering would empty the candidate list, the orchestrator
    # falls back to the unfiltered set so the round isn't completely
    # silent.
    excluded_speaker_ids: set[str] = field(default_factory=set)
    # Phase 13.5 (B+D) — cross-round repetition memory. Last N turns of
    # each character keyed by character_id, oldest first. The plugin
    # populates this from the per-session turns store (RP container or
    # legacy character_turns table). The repetition guard compares the
    # new generation against BOTH this char's own past turns AND the
    # other speakers' current-round turns. Without this, Mira repeats
    # 'wische mir den Salzfilm von der Stirn' across three rounds
    # because the 13.2 guard only saw within-round predecessors.
    prior_turns_by_char: dict[str, list[str]] = field(default_factory=dict)
    # RP-v2 — shared scene awareness from the world-state (open demands),
    # shown to EVERY present speaker so anyone can react or prod the caregiver
    # ("Shani, schau nach dem Baby"). Plus a per-character strong obligation
    # keyed by character_id (the caregiver "du MUSST handeln"). Both populated
    # by the plugin before run_round; empty = no open demands this round.
    scene_awareness: str = ""
    obligations_by_char: dict[str, str] = field(default_factory=dict)
    # RP-v2 Phase 2 — shared physical-continuity facts (who holds the baby,
    # where it is …) as a MUST system section so no character contradicts the
    # scene's physical reality. Plugin populates it from world.json "facts".
    physical_facts: str = ""


@dataclass
class GroupTurnResult:
    """Output of one round."""

    turns: list[CharacterTurn] = field(default_factory=list)
    speaker_order: list[str] = field(default_factory=list)  # character_ids
    # Echoed from the request so downstream knows how to log the round.
    user_message: str = ""
    pulse_from_id: str = ""
    pulse_text: str = ""


# ─── Repetition guard (Phase 13.2) ───────────────────────────────────


# German stopwords — cheap n-gram comparison ignores these so we
# don't flag two turns as similar just because both contain "der die
# das mit von ich". Kept intentionally short — we want enough signal
# left in the comparison set that real similarity (clothing/sand/
# salt) lights up Jaccard.
_REPETITION_STOPWORDS: frozenset[str] = frozenset({
    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einer", "eines", "einem", "einen",
    "ich", "du", "er", "sie", "es", "wir", "ihr",
    "mich", "dich", "ihn", "uns", "euch",
    "mein", "dein", "sein", "ihr", "unser", "euer",
    "und", "oder", "aber", "doch", "denn",
    "in", "im", "auf", "an", "am", "zu", "zum", "zur",
    "von", "mit", "bei", "nach", "über", "unter", "vor",
    "ist", "sind", "war", "waren", "wird", "werden", "hat",
    "haben", "hatte", "kann", "könnte", "muss", "soll",
    "nicht", "kein", "keine", "schon", "noch", "auch",
    "wenn", "als", "wie", "was", "wer", "warum", "weil",
    "sich", "selbst", "nur", "fast",
})


_NGRAM_TOKEN_RE = re.compile(r"[a-zäöüß]{3,}", re.IGNORECASE)


def _last_history_user_text(history: list[dict[str, Any]] | None) -> str:
    """Phase 13.3 helper: pull the most recent USER message from a
    session history list. Used by the natural-order activator when no
    explicit ``user_message`` is set this round (pulse turns)."""
    if not history:
        return ""
    for msg in reversed(history):
        if (msg.get("role") or "").lower() == "user":
            return str(msg.get("content") or "")
    return ""


def _ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    """Tokenise + lowercase + drop stopwords + emit n-gram tuples.

    Default ``n=3`` (trigram) — German narrative RP tends to insert
    enough variation between adjacent content words that 4-grams miss
    real repetition. Trigrams strike the right balance between false
    positives (single-word matches) and false negatives (entirely
    re-worded sand-staring).
    """
    tokens = [
        t.lower() for t in _NGRAM_TOKEN_RE.findall(text or "")
        if t.lower() not in _REPETITION_STOPWORDS
    ]
    if len(tokens) < n:
        # For very short turns, fall back to bigrams so we still have
        # SOME signal. A 3-word sentence vs another 3-word sentence
        # would otherwise always be "no match".
        n = max(2, min(n, len(tokens)))
    if len(tokens) < n or n < 2:
        return set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def detect_repetition(
    new_text: str,
    previous_texts: list[str],
    threshold: float = 0.4,
) -> tuple[bool, float, list[str]]:
    """Return (is_repetitive, max_jaccard, sample_repeated_phrases).

    Phase 13.2 helper. The orchestrator calls this AFTER the LLM
    returns to check whether the new turn re-mixes phrases its
    same-round predecessors already used. ``threshold=0.4`` was
    picked from Mike's Castaway log — two turns with "Sand starren /
    Schläfen reiben / Salz brennt" share roughly 40-50% of their
    (stopword-stripped) 4-grams.

    Returns the sample phrases so the re-prompt can name them.
    """
    if not new_text or not previous_texts:
        return False, 0.0, []
    new_grams = _ngrams(new_text)
    if not new_grams:
        return False, 0.0, []
    max_jac = 0.0
    overlapping_grams: set[tuple[str, ...]] = set()
    for prev in previous_texts:
        prev_grams = _ngrams(prev)
        if not prev_grams:
            continue
        jac = _jaccard(new_grams, prev_grams)
        if jac > max_jac:
            max_jac = jac
            overlapping_grams = new_grams & prev_grams
    if max_jac < threshold:
        return False, max_jac, []
    samples = [" ".join(g) for g in list(overlapping_grams)[:5]]
    return True, max_jac, samples


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
        speaker_selection_brain: str | None = None,
        global_style_prompt: str = "",
        always_call_orchestrator: bool = False,
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
        # Brain to use for the speaker-selection classification call. Defaults
        # to ``brain`` for backward compatibility, but the plugin pins it to
        # E4B so the cheap deciding-call doesn't tie up the big A4B brain.
        self._speaker_brain = speaker_selection_brain or brain
        # Optional global RP style prompt — appended as a MUST-priority
        # system section into every character turn so all chars write
        # in a uniform style on top of their individual persona.
        self._global_style_prompt = (global_style_prompt or "").strip()
        # When True, the orchestrator brain (e4b by default) is always
        # consulted before a round, even when @-mentions or NL-mentions
        # already covered every speaker. The orchestrator gets the
        # mention-derived order as a "preferred" hint and can confirm
        # or refine it — useful for catching cases where the user
        # named a char that shouldn't actually respond (different room,
        # asleep, etc.). Default off because it costs one extra LLM
        # call per round.
        self._always_call_orchestrator = bool(always_call_orchestrator)
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

        # Phase 13.2: skip-cooldown filter. Chars that returned an
        # empty turn last round are excluded for this round so the
        # LLM-orchestrator doesn't pick them silent again. If filtering
        # would empty the candidate list, ignore the cooldown — better
        # one repeat skip than a totally silent round.
        if req.excluded_speaker_ids:
            filtered = [
                c for c in eligible
                if c.id not in req.excluded_speaker_ids
            ]
            if filtered:
                if len(filtered) < len(eligible):
                    log.info(
                        "character_chat.skip_cooldown_filtered "
                        "session=%s excluded=%s remaining=%d",
                        req.session_id,
                        sorted(req.excluded_speaker_ids),
                        len(filtered),
                    )
                eligible = filtered

        # Speaker selection priority:
        #   1. @-mentions (explicit user intent)
        #   2. natural-language name mentions (e.g. "Mara, schau mal...")
        #   3. extra_forced (pulse-mention propagation, set by the plugin)
        #   4. autonomous LLM order or round-robin fallback
        # Each layer only contributes when the previous one yielded nothing
        # OR adds non-overlapping ids — see _pick_speakers for the merge.
        at_mentions = _parse_at_mentions(req.user_message, req.characters)
        nl_mentions: list[str] = []
        if not at_mentions and req.user_message.strip():
            nl_mentions = parse_nl_mentions(req.user_message, req.characters)
        forced = at_mentions or nl_mentions
        # extra_forced is appended after the user-driven order. Filter out
        # the pulse-from character (already spoke via pulse_text) and any
        # ids already in `forced` so we don't double-schedule a speaker.
        extras = [
            cid
            for cid in (req.extra_forced or [])
            if cid != req.pulse_from_id and cid not in forced
        ]
        speaker_order = await self._pick_speakers(
            req=req, eligible=eligible, forced=forced + extras
        )
        # Phase 13.5 hotfix v3 — when the user explicitly @-mentions or
        # NL-names character(s), DON'T auto-fill to max_speakers. Mike's
        # gripe: he writes '@Sandra hilf mir' and Lena also chimes in
        # because slot 2 of max_speakers=2 gets filled by the LLM/round-
        # robin path. Plus when both chars share identical tracked_stats
        # (arousal=extrem_notgeil etc.) the second answer mirrors the
        # first emotionally even though sequential prompting differs.
        # When user names someone, only the named char(s) speak.
        # ``extras`` (pulse-mention-propagation) ALSO counts as 'user-
        # like intent' from a pulse — keep it in the cap too.
        explicit_count = len(forced) + len(extras)
        if explicit_count > 0:
            speaker_order = speaker_order[: explicit_count]
        else:
            speaker_order = speaker_order[: self._max_speakers]
        if at_mentions or nl_mentions or extras:
            log.info(
                "character_chat.mentions_parsed at=%s nl=%s extras=%s order=%s",
                at_mentions,
                nl_mentions,
                extras,
                speaker_order,
            )

        by_id = {c.id: c for c in req.characters}
        turns: list[CharacterTurn] = []

        # Phase 13.5 (A): if a pulse fired, synthesise a visible turn for
        # the pulse-from character containing the pulse_text. Without this
        # the trigger char never appears in the chat — only the OTHERS
        # who reacted to her pulse get persisted, and the user can't see
        # what she actually did. Mike's diagnosis: "Yara taucht nie auf"
        # despite firing pulses every 10 min. The pulse_text is already
        # her "voice"; we just persist it under her name so it's visible.
        if req.pulse_from_id and req.pulse_text:
            pulse_card = by_id.get(req.pulse_from_id)
            if pulse_card is not None:
                turns.append(CharacterTurn(
                    character_id=pulse_card.id,
                    character_name=pulse_card.name,
                    content=req.pulse_text,
                    skipped=False,
                    order=0,
                ))

        for idx, char_id in enumerate(speaker_order):
            card = by_id.get(char_id)
            if card is None:
                continue
            turn = await self._run_single_turn(
                card=card,
                order=idx + (1 if turns else 0),
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

    def _activate_natural_order(
        self,
        *,
        req: GroupTurnRequest,
        eligible: list[CharacterCard],
    ) -> list[str]:
        """Phase 13.3 — SillyTavern-style deterministic speaker pick.

        Mirrors ``activateNaturalOrder`` in
        ``SillyTavern/public/scripts/group-chats.js``. Three signals,
        in priority order:

        1. **Name-mention** in the most recent input — that character
           is activated immediately and gets pole position.
        2. **Talkativeness roll** — for every other character a single
           ``random()`` is compared against ``card.talkativeness``;
           below threshold = silent this round. With Mike's 4-char
           Castaway group at default 0.5, the expected speaker count
           per round is ~2 — exactly what ``max_speakers_per_round``
           caps anyway.
        3. **Recency / chattiness fallback** — if no character was
           activated by either rule, pick the eligibly-chattiest one
           so the round isn't completely silent.

        No LLM call, no async I/O. Returns an ordered list of
        ``character_id``. Empty list = "couldn't decide cleanly,
        the caller should fall back to the LLM-based picker".
        """
        if not eligible:
            return []
        haystack = (
            req.user_message
            or req.pulse_text
            or _last_history_user_text(req.history)
            or ""
        ).lower()

        activated: list[str] = []
        seen: set[str] = set()

        # Pass 1: name mentions go first, in name-order.
        for card in eligible:
            name = (card.name or "").strip().lower()
            if not name:
                continue
            # Word-boundary-ish match — avoids "Lena" inside "Galena".
            pattern = r"\b" + re.escape(name) + r"\b"
            if re.search(pattern, haystack):
                if card.id not in seen:
                    activated.append(card.id)
                    seen.add(card.id)

        # Pass 2: talkativeness roll for everyone not already activated.
        for card in eligible:
            if card.id in seen:
                continue
            roll = random.random()
            if float(card.talkativeness) >= roll:
                activated.append(card.id)
                seen.add(card.id)

        # If the rolls left us empty, return [] so the caller can fall
        # back to the LLM picker. We DON'T blindly pick the chattiest
        # — that would mask a "nobody really fits" signal that the LLM
        # might handle better.
        return activated

    async def _pick_speakers(
        self,
        *,
        req: GroupTurnRequest,
        eligible: list[CharacterCard],
        forced: list[str],
    ) -> list[str]:
        """Decide who speaks in this round and in what order.

        Priority:
            1. If the user @-mentioned or NL-mentioned a character, that
               character goes first (the order from ``forced`` is preserved).
            2. If ``turn_selection == "round_robin"`` OR only one char
               eligible, remaining characters follow alphabetically.
            3. Otherwise, ask the LLM (``speaker_selection_brain`` —
               e4b by default) for an ordered list of the *remaining*
               candidates.
            4. When ``always_call_orchestrator=True``, the LLM is also
               consulted to confirm/refine a fully-mention-derived order.

        Every path emits one ``character_chat.speakers_picked`` log line
        with ``method=mention|round_robin|llm|llm_refined`` so the user
        can see in the log exactly which path fired (and whether the
        e4b brain was consulted).
        """
        # Deduplicate but preserve insertion order.
        seen: set[str] = set()
        order: list[str] = []

        for mention in forced:
            if mention in seen:
                continue
            seen.add(mention)
            order.append(mention)

        method = "mention" if order else ""

        if self._turn_selection == "round_robin" or len(eligible) == 1:
            rest = sorted(
                (c for c in eligible if c.id not in seen),
                key=lambda c: c.name.lower(),
            )
            order.extend(c.id for c in rest)
            method = method or "round_robin"
            self._log_speakers_picked(
                method=method, order=order, brain_called=False, req=req,
            )
            return order

        # autonomous: LLM picks order over the remaining candidates.
        remaining = [c for c in eligible if c.id not in seen]

        # Phase 13.3b — UPDATED HIERARCHY (Mike's choice):
        #   1. Name-mention (handled above, in ``forced``)
        #   2. LLM story-match — picks the *contextually right* speaker
        #      with full persona + relationships + last-turn context
        #      (e.g. "Sandra is the nurse → answers when Lena is hurt")
        #   3. Talkativeness roll as last-resort for idle phases when
        #      the LLM also can't decide cleanly.
        #
        # always_call_orchestrator path: LLM gets *all* eligibles and the
        # mention-derived order as a preferred hint. We use this to
        # catch cases where the user named a char that shouldn't
        # actually respond (different room, asleep, etc.).
        if self._always_call_orchestrator and (order or remaining):
            llm_order = await self._ask_llm_for_order(
                req=req,
                candidates=eligible,
                preferred_order=list(order),
            )
            if llm_order:
                self._log_speakers_picked(
                    method="llm_refined", order=llm_order,
                    brain_called=True, req=req,
                )
                return llm_order
            # Fall through if LLM returned nothing usable.

        if not remaining:
            self._log_speakers_picked(
                method=method or "mention", order=order,
                brain_called=False, req=req,
            )
            return order

        # Phase 13.3b: PRIMARY autonomous path — story-match LLM call.
        # The selector now sees personas + relationships + last turn
        # so it can answer "who would naturally react RIGHT NOW".
        llm_order = await self._ask_llm_for_order(req=req, candidates=remaining)
        if llm_order:
            order.extend(llm_order)
            self._log_speakers_picked(
                method=(method or "llm_story_match"),
                order=order, brain_called=True, req=req,
            )
            return order

        # LAST RESORT: LLM had no clear pick — roll talkativeness so
        # idle group chat (everyone equally likely) still produces a
        # speaker instead of silence.
        natural_order = self._activate_natural_order(
            req=req, eligible=remaining,
        )
        if natural_order:
            order.extend(natural_order)
            self._log_speakers_picked(
                method=(method or "talkativeness_fallback"),
                order=order, brain_called=True, req=req,
            )
            return order

        # Absolute last resort — neither LLM nor talkativeness picked
        # anyone. Fall back to round-robin so the round isn't silent.
        if not order:
            order = [c.id for c in remaining]
            self._log_speakers_picked(
                method="round_robin_safety",
                order=order, brain_called=True, req=req,
            )
            return order

        self._log_speakers_picked(
            method=method or "mention",
            order=order, brain_called=True, req=req,
        )
        return order

    def _log_speakers_picked(
        self,
        *,
        method: str,
        order: list[str],
        brain_called: bool,
        req: GroupTurnRequest,
    ) -> None:
        """Single-line summary of how this round's speakers were chosen.

        Mike's audit: "der LLM Server wird aber nie angesprochen" — this
        log makes it obvious whether e4b was consulted on a given round.
        Search the log for ``character_chat.speakers_picked`` to see the
        path that fired (mention / round_robin / llm / llm_refined /
        llm_failed_round_robin).
        """
        log.info(
            "character_chat.speakers_picked method=%s brain=%s "
            "brain_called=%s order=%s session=%s",
            method,
            self._speaker_brain if brain_called else "(skipped)",
            brain_called,
            order,
            req.session_id,
        )

    async def _ask_llm_for_order(
        self,
        *,
        req: GroupTurnRequest,
        candidates: list[CharacterCard],
        preferred_order: list[str] | None = None,
    ) -> list[str]:
        # Phase 13.3b: build a richer roster so the selector can reason
        # about expertise + relationships ("Sandra ist Krankenschwester
        # → reagiert auf Verletzungen", "Mira kann tauchen → reagiert
        # wenn jemand was im Wasser sieht"). Mike's brief: characters
        # should ASK each other based on who can do what.
        roster_blocks: list[str] = []
        char_by_id = {c.id: c for c in candidates}
        for c in candidates:
            persona_brief = _brief_persona(c)
            chat_score = (
                f"talkativeness={float(c.talkativeness):.1f}"
            )
            # Resolve relationships against THIS roster only — labels
            # for non-eligible chars don't help the picker decide.
            rel_lines: list[str] = []
            for other_id, label in (c.relationships or {}).items():
                other = char_by_id.get(other_id)
                if other is not None and label:
                    rel_lines.append(f"  · {other.name}: {label}")
            block = (
                f"- **{c.name}** (id={c.id}, {chat_score})\n"
                f"  Profil: {persona_brief}"
            )
            if rel_lines:
                block += "\n  Beziehungen:\n" + "\n".join(rel_lines)
            roster_blocks.append(block)
        roster = "\n\n".join(roster_blocks)

        trigger = req.user_message.strip() or (
            f"*{req.pulse_text}*" if req.pulse_text else ""
        )
        if not trigger:
            trigger = "(Gruppendynamik ohne neuen User-Impuls — wer spricht jetzt?)"

        history_blurb = _format_history_tail(req.history, limit=4)

        # When the caller has a mention-derived order, hand it to the LLM
        # as a hint so it can confirm or override (e.g. when a mentioned
        # char is in a different room and shouldn't actually answer).
        hint_block = ""
        if preferred_order:
            hint_block = (
                "\n## Vorgeschlagene Reihenfolge (vom User benannt)\n"
                + ", ".join(preferred_order)
                + "\nBestätige diese Reihenfolge ODER gib eine bessere zurück, "
                "falls einer der Genannten gerade nicht reagieren sollte "
                "(unangebrachte Anwesenheit, Schlaf, anderer Raum etc.).\n"
            )

        system = (
            "Du bist der Turn-Orchestrator einer RP-Gruppe. Deine Aufgabe: "
            "entscheiden, **welcher Charakter** kontextuell am besten geeignet "
            "ist auf den Impuls zu reagieren — und ob noch ein zweiter "
            "naheliegend ist.\n\n"
            "Heuristik:\n"
            "1. **Expertise-Match** wiegt am höchsten. Wenn jemand "
            "verletzt ist → Krankenschwester reagiert. Wenn was im Wasser "
            "treibt → die Surfer/Taucherin. Wenn Lager gebaut werden muss "
            "→ Architektin. Lies die Profile.\n"
            "2. **Beziehungs-Bezug**: wer wurde direkt angesprochen oder "
            "im letzten Turn erwähnt?\n"
            "3. **Relevanz**: kann dieser Charakter etwas konkret zur "
            "Situation BEITRAGEN — oder würde er nur 'mhm' sagen?\n"
            "4. **Talkativeness** ist ein Bias, kein Muss — niedrige Werte "
            "(z.B. 0.3) sprechen weniger oft, aber wenn ihre Expertise "
            "gefragt ist, spricht sie trotzdem.\n\n"
            "Antworte AUSSCHLIESSLICH mit IDs, komma-separiert, in der "
            "Reihenfolge in der sie sprechen sollen. Maximal "
            f"{self._max_speakers} Charaktere. Wenn niemand klar passt, "
            "gib eine leere Liste zurück (ein Wort: NONE)."
        )
        user = (
            f"## Charaktere\n{roster}\n\n"
            f"## Letzte Zeilen aus dem Chat\n{history_blurb}\n\n"
            f"## Aktueller Impuls\n{trigger}"
            f"{hint_block}\n\n"
            "## Deine Antwort (IDs komma-separiert, oder NONE):"
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
                brain=self._speaker_brain,
                max_tokens=120,
                temperature=0.3,
                thinking=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("character_chat.order_llm_failed: %s", exc)
            return [c.id for c in candidates]

        # Phase 13.3b: explicit "no clear match" signal. If the LLM
        # answers ``NONE`` (we ask for it in the system prompt), we
        # return [] so the caller can fall through to the
        # talkativeness-roll last-resort. Stripped of the usual
        # punctuation cosmetic.
        cleaned_raw = (raw or "").strip().strip("[]()<>\"' \t`.,;:!\n")
        if cleaned_raw.upper() == "NONE":
            return []

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
                # Phase 13: pass the session_id so the recall function
                # can route to the per-RP-session collection. Plugins
                # implementing the legacy 3-arg signature still work
                # because we pass session_id as a keyword.
                own_memories = await self._recall_fn(
                    character_id=card.id,
                    query=query,
                    limit=self._recall_limit,
                    session_id=req.session_id,
                )
            except TypeError:
                # Fallback for older recall_fn signatures without session_id.
                try:
                    own_memories = await self._recall_fn(
                        character_id=card.id,
                        query=query,
                        limit=self._recall_limit,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "character_chat.recall_failed_legacy: %s (char=%s)",
                        exc,
                        card.name,
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

        _emit_prompt_debug(
            character=card.name,
            brain=self._brain,
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

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

        _emit_response_debug(
            character=card.name,
            content=(raw or "").strip(),
            skipped=skipped,
        )

        # Phase 13.2 + 13.5 (B+D): repetition guard. Compares the new
        # turn against TWO pools:
        #   * same-round others (13.2 — who else just spoke this round)
        #   * THIS char's own recent turns from prior rounds (13.5 —
        #     stops Mira repeating 'wische mir den Salzfilm von der
        #     Stirn' three rounds in a row)
        # If overlap above threshold, re-prompt ONCE with an anti-rep
        # hint. Only one retry — a second loop doubles the token cost
        # and rarely helps.
        own_prior = req.prior_turns_by_char.get(card.id, [])
        if (
            content
            and not skipped
            and (previous_turns or own_prior)
        ):
            prev_texts = [
                pt.content for pt in previous_turns
                if pt.content and not pt.skipped
            ]
            # Add this char's own prior turns so self-repetition across
            # rounds also triggers the guard. Keep a small window —
            # comparing against 50 ancient turns is wasted work.
            prev_texts.extend(own_prior[-5:])
            is_rep, jac, samples = detect_repetition(
                content, prev_texts, threshold=0.4,
            )
            if is_rep:
                log.info(
                    "character_chat.repetition_detected character=%s "
                    "jaccard=%.2f samples=%s",
                    card.name, jac, samples[:3],
                )
                anti_rep_hint = (
                    "\n\n## WICHTIG (Anti-Wiederholung)\n"
                    "Folgende Phrasen wurden bereits verwendet (von dir "
                    "selbst in einer vorherigen Runde oder von "
                    "Mit-Charakteren in dieser Runde). Vermeide sie und "
                    "schreib KEINE Variation davon — beschreibe etwas "
                    "ANDERES (ein neuer Geruch, ein neues Geräusch, "
                    "eine spezifische Aktion, eine andere Körpergeste, "
                    "eine konkrete Beobachtung). "
                    f"Vermeiden: {', '.join(samples[:5])}."
                )
                retry_messages = [
                    {"role": "system",
                     "content": system_prompt + anti_rep_hint},
                    {"role": "user", "content": user_content},
                ]
                try:
                    retry_raw = await self._llm_chat(
                        messages=retry_messages,
                        brain=self._brain,
                        max_tokens=self._max_tokens,
                        temperature=min(
                            1.0, self._temperature + 0.1,
                        ),  # nudge up for variety
                        thinking=False,
                    )
                    retry_content = (retry_raw or "").strip()
                    if retry_content and not self._is_pass(retry_content):
                        content = retry_content
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "character_chat.repetition_retry_failed "
                        "character=%s error=%s", card.name, exc,
                    )

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
        "lorebook_before_persona",   # NEW (Phase 9.8)
        "identity",
        "persona",
        "lorebook_after_persona",    # NEW
        "lorebook_before_scenario",  # NEW (default lore position)
        "scenario",
        "age_guidance",
        "char_state",
        "open_obligations",          # RP-v2 — caregiver "act now" duty
        "physical_continuity",       # RP-v2 Phase 2 — shared physical reality
        "others",
        "example_dialog",
        "global_style",
        "rules",
        "impersonation_guard",       # NEW Phase 13.3 — last system word
    )
    # User section order (fed as the user message for the turn).
    _USER_SECTION_ORDER: tuple[str, ...] = (
        "group_roster",              # NEW Phase 13.3 — pre-history
        "scene_awareness",           # RP-v2 — shared open-demand awareness
        "lorebook_before_history",
        "history",
        "memory",
        "pulse",
        "lorebook_before_user_message",
        "user_message",
        "prev_turns",
        "group_nudge",               # NEW Phase 13.3 — post-history
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

        # Lorebook activations land in their declared position. We add
        # the section here so it sits in the right slot of
        # _SYSTEM_SECTION_ORDER (or _USER_SECTION_ORDER for the
        # user-side positions). Empty positions are skipped.
        lore_result = (req.lore_by_speaker or {}).get(card.id)
        lore_sections = _lore_sections(lore_result) if lore_result else {}
        for sec_name in (
            "lorebook_before_persona",
            "lorebook_after_persona",
            "lorebook_before_scenario",
        ):
            if sec_name in lore_sections:
                sections.append(lore_sections[sec_name])

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

        # Phase 13: live session state takes precedence over the
        # legacy character-scoped ``state`` column. The plugin pre-
        # populates ``live_state_by_char`` from the RP container; any
        # other code path (chat-tab character mode, tests) still falls
        # back to ``card.state`` so this is backward compatible.
        live_state = (req.live_state_by_char or {}).get(card.id)
        effective_state = live_state if live_state else (card.state or {})
        state_text = _format_state_block(effective_state)
        if state_text:
            sections.append(
                PromptSection(
                    name="char_state",
                    # MUST so state survives prompt-trimming AND the LLM
                    # can't ignore it on tight contexts. State drift was
                    # Mike's audit point #2: char "ist nackt" but the
                    # turn says "zupft an Kleidung" — that contradiction
                    # is impossible if the state stays visible.
                    priority=Priority.MUST,
                    text=(
                        f"## Dein Zustand (verbindlich!)\n{state_text}\n\n"
                        "Halte dich strikt an diesen Zustand. Wenn du etwas "
                        "anderes tun willst (Ort wechseln, Kleidung ändern, "
                        "Haltung wechseln), schreib am Ende der Antwort einen "
                        "<state>...</state>-Block der den Zustand explizit "
                        "aktualisiert. Niemals einen Zustand erzählen der "
                        "deinem aktuellen Zustand widerspricht."
                    ),
                    role="system",
                    max_tokens=200,
                )
            )

        # RP-v2 — strong "you are responsible, act now" obligation for the
        # caregiver of an open demand (e.g. the mother for baby.hunger). Only
        # the responsible character gets it; everyone else just sees the
        # shared scene_awareness in their user content.
        obligation = (req.obligations_by_char or {}).get(card.id)
        if obligation:
            sections.append(
                PromptSection(
                    name="open_obligations",
                    priority=Priority.HIGH,
                    text=f"## Offene Verpflichtung (jetzt handeln!)\n{obligation}",
                    role="system",
                    max_tokens=200,
                )
            )

        # RP-v2 Phase 2 — shared physical reality (who holds the baby, where
        # it is). MUST so it survives trimming and binds every character.
        if (req.physical_facts or "").strip():
            sections.append(
                PromptSection(
                    name="physical_continuity",
                    priority=Priority.MUST,
                    text=req.physical_facts.strip(),
                    role="system",
                    max_tokens=240,
                )
            )

        others_text = _format_others_block(
            card, all_cards,
            live_state_by_char=req.live_state_by_char,
        )
        if others_text:
            sections.append(
                PromptSection(
                    name="others",
                    # Phase 13.5 (C): bumped LOW → MEDIUM so peer states
                    # survive context-trimming. With LOW the section was
                    # often dropped under tight budgets, leaving the LLM
                    # blind to peer locations and free to hallucinate.
                    priority=Priority.MEDIUM,
                    text=others_text,
                    role="system",
                    # Higher cap because each peer now carries state
                    # bits, not just a name. ~50 tokens per peer × 5
                    # peers = 250 ceiling with headroom.
                    max_tokens=350,
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

        # Global style — config-driven, MUST so the same uniform style
        # applies to every character regardless of their individual
        # persona's tone. Sits between the per-char blocks (above) and
        # the per-turn rules (below) — the LLM sees: "I am X with persona
        # Y; here is the house style; here are the rules; here is what
        # I'm responding to right now."
        if self._global_style_prompt:
            sections.append(
                PromptSection(
                    name="global_style",
                    priority=Priority.MUST,
                    text=self._global_style_prompt,
                    role="system",
                )
            )

        sections.append(
            PromptSection(
                name="rules",
                priority=Priority.MUST,
                text=(
                    "## Regeln (RP-Disziplin)\n"
                    "- **Bleib in deinem Charakter.** Du sprichst, denkst und "
                    "handelst ausschliesslich als die Person, die in der "
                    "Persona-Sektion beschrieben ist. Keine Meta-Kommentare, "
                    "keine Regie-Anweisungen in eckigen Klammern.\n"
                    "- **Sprich NIE für den User oder andere Charaktere.** "
                    "Du beschreibst nur, was DEIN Charakter sagt, fühlt und "
                    "tut. Was die anderen tun, entscheiden sie selbst (oder "
                    "der User). Lege niemandem Worte in den Mund.\n"
                    "- **Treibe die Story nicht eigenmächtig voran.** Der "
                    "User führt die Handlung. Du REAGIERST auf das, was "
                    "passiert ist — du erfindest keine neuen Plot-Punkte, "
                    "keine plötzlichen Ereignisse, keine Zeitsprünge. Wenn "
                    "die Szene stockt, fragst du in deiner Stimme nach, "
                    "statt die Story selbst weiterzuschreiben.\n"
                    "- **Gefühle und Handlungen detailreich in *Sternchen*.** "
                    "Inneres Erleben, Körpersprache, kleine Handlungen — "
                    "beschreibe sie ausführlich, nicht nur '*nickt*' sondern "
                    "z.B. '*lehnt sich langsam zurück, kneift die Augen "
                    "zusammen und atmet hörbar aus*'. Mehrere Sätze pro "
                    "Aktion sind erlaubt und gewünscht, wenn die Szene es "
                    "trägt.\n"
                    "- **Länge**: Schreib so viel wie die Szene verlangt — "
                    "von einem knappen Satz bis zu mehreren Absätzen. Lieber "
                    "lebendig und detailliert als künstlich kurz. Aber kein "
                    "Roman, wenn die Szene nichts hergibt.\n"
                    "- **Andere Anwesende**: Wenn es sich anbietet, sprich "
                    "sie namentlich an oder reagiere körperlich auf sie.\n"
                    "- Du DARFST am Ende deiner Antwort optional einen "
                    "<state>key=value; key=value</state> Block setzen, wenn "
                    "sich dein Zustand ändert. Anker-Keys: location, mood, "
                    "last_action, clothing, posture, condition. Du darfst "
                    "auch eigene Keys ergänzen (snake_case, z.B. "
                    "holds_object=Schwert, proximity_to_user=nah). Wird "
                    "nicht angezeigt, dient als dein Gedächtnis für die "
                    "nächste Runde. Aktualisiere den Zustand IMMER, wenn "
                    "deine Aktion ihn ändert (anziehen → clothing, "
                    "umziehen → location, ...) — sonst entstehen Lücken."
                ),
                role="system",
            )
        )

        # Phase 13.3 — impersonation-guard (system, very last). Mirrors
        # SillyTavern's "[Don't write as {{user}}…]" canned line. Lands
        # AFTER ``rules`` so it's the last thing in the system block.
        sections.append(
            PromptSection(
                name="impersonation_guard",
                priority=Priority.MUST,
                text=(
                    f"## Du bist NICHT der User\n"
                    f"Du bist ausschließlich {card.name}. Schreibe nicht "
                    f"aus der Sicht des Users (Mike). Beschreibe keine "
                    f"Worte, Gedanken oder Handlungen des Users — der "
                    f"spricht für sich selbst, in einem eigenen Turn. "
                    f"Auch keine Worte oder Handlungen anderer "
                    f"Charaktere — die haben ihre eigenen Turns."
                ),
                role="system",
                max_tokens=120,
            )
        )

        # ─── USER sections ────────────────────────────────────────────
        # Phase 13.3 — group roster: pre-history "[Gruppenchat. Anwesend: …]"
        # marker, modeled after SillyTavern's ``default_new_group_chat_prompt``.
        # Goes at the very top of the user content so the LLM enters
        # the chat context with a clear "this is a group" framing.
        peer_names = [c.name for c in (all_cards or []) if c.name]
        if peer_names:
            roster_line = (
                f"[Gruppenchat. Anwesend: {', '.join(peer_names)}.]"
            )
            sections.append(
                PromptSection(
                    name="group_roster",
                    priority=Priority.HIGH,
                    text=roster_line,
                    role="user",
                    max_tokens=80,
                )
            )

        # RP-v2 — shared scene awareness: open world-state demands every
        # present character notices, so anyone can react or prod the caregiver.
        if (req.scene_awareness or "").strip():
            sections.append(
                PromptSection(
                    name="scene_awareness",
                    priority=Priority.HIGH,
                    text=req.scene_awareness.strip(),
                    role="user",
                    max_tokens=300,
                )
            )

        # Lore that should land in the user-content area (before history
        # or right before the user message).
        if "lorebook_before_history" in lore_sections:
            sections.append(lore_sections["lorebook_before_history"])

        # RP-Fix: show a real recent transcript. ``limit`` is the message
        # window (user lines + every character's turns); HIGH priority so the
        # recent exchange isn't trimmed to 2 messages while 16k+ context sits
        # free — that drop was why characters "forgot" facts from 2 turns ago.
        history_text = _format_history_tail(req.history, limit=16)
        if history_text:
            # Snapshot req.history so the reduce_fn closure stays pure.
            history_ref = req.history

            def _reduce_history(step: int, h: list[dict[str, Any]] = history_ref) -> str:
                trimmed = _format_history_tail(h, limit=step)
                return f"## Bisheriger Chat\n{trimmed}" if trimmed else ""

            sections.append(
                PromptSection(
                    name="history",
                    priority=Priority.HIGH,
                    text=f"## Bisheriger Chat\n{history_text}",
                    role="user",
                    max_tokens=3000,
                    reduce_fn=_reduce_history,
                    reduce_steps=[10, 6, 3],
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

        # Lore right before the user message.
        if "lorebook_before_user_message" in lore_sections:
            sections.append(lore_sections["lorebook_before_user_message"])

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
                    # MUST: the prior speaker's reply is the most important
                    # context for the current turn and would render the
                    # round incoherent if dropped under tight budgets. We
                    # accept that long prior-turns push history out first
                    # — that's the right trade-off for sequential RP.
                    priority=Priority.MUST,
                    text="## Reaktionen dieser Runde\n" + "\n".join(lines),
                    role="user",
                )
            )

        # Tailor the "you're up now" instruction so the LLM understands
        # whom to address. When a previous speaker just spoke (and there's
        # no fresh user message), this turn is effectively a Char-to-Char
        # reply — frame it that way instead of leaning on the generic
        # "react to user" wording.
        last_speaker_name = ""
        if previous_turns:
            for t in reversed(previous_turns):
                if not t.skipped and t.content and t.character_id != card.id:
                    last_speaker_name = t.character_name
                    break

        # Phase 13.3 — group nudge (post-history). The single biggest
        # behavioural lever from the SillyTavern research: stamping
        # "[Schreibe als <char> + reagiere auf das Letzte + verteilt
        # die Aufgaben]" right before the LLM generates breaks the
        # parallel-monologue loop. Lands AFTER prev_turns and before
        # the final ``instruction`` so the LLM sees it last.
        if last_speaker_name:
            nudge_text = (
                f"[Schreibe die nächste Antwort ausschließlich als "
                f"{card.name}. Reagiere konkret auf das, was "
                f"{last_speaker_name} gerade gesagt oder getan hat — "
                f"keine parallele Wiederholung. **Erfinde KEINE "
                f"Aktionen oder Aufenthaltsorte für andere Charaktere** "
                f"— ihr aktueller Zustand steht oben unter 'Andere "
                f"Anwesende'; nimm den als Fakt. Wenn die Gruppe gerade "
                f"diskutiert was zu tun ist, übernimm eine konkrete, "
                f"andere Aufgabe als die anderen — einer sammelt Holz, "
                f"ein anderer Wasser, ein dritter Essen. Nicht alle "
                f"das Gleiche.]"
            )
        else:
            nudge_text = (
                f"[Schreibe die nächste Antwort ausschließlich als "
                f"{card.name}. Reagiere konkret auf das, was zuletzt "
                f"passiert ist. **Erfinde KEINE Aktionen oder "
                f"Aufenthaltsorte für andere Charaktere** — ihr "
                f"aktueller Zustand steht oben unter 'Andere "
                f"Anwesende'; nimm den als Fakt. Wenn die Gruppe gerade "
                f"diskutiert was zu tun ist, übernimm eine konkrete "
                f"Aufgabe — einer sammelt Holz, ein anderer Wasser, "
                f"ein dritter Essen. Nicht alle das Gleiche.]"
            )
        sections.append(
            PromptSection(
                name="group_nudge",
                # MUST so it survives any token trimming. This is the
                # last thing the LLM reads before generating, and it's
                # the spine of the group dynamic.
                priority=Priority.MUST,
                text=nudge_text,
                role="user",
                max_tokens=200,
            )
        )

        instruction_text = _build_instruction(
            card_name=card.name,
            has_user_message=bool(req.user_message.strip()),
            has_pulse=bool(req.pulse_text and req.pulse_from_id),
            last_speaker_name=last_speaker_name,
        )
        sections.append(
            PromptSection(
                name="instruction",
                priority=Priority.MUST,
                text=instruction_text,
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


def _lore_sections(
    activation: ActivationResult,
) -> dict[str, PromptSection]:
    """Build one ``PromptSection`` per non-empty lore position.

    The role is derived from the position: positions inside the system
    prompt (before/after persona, before scenario) carry ``role="system"``;
    user-side positions (before history / before user_message) live
    in the user message. Priority is HIGH so the lore survives most
    trimming — but not MUST, so under hard budget pressure the engine
    still drops lore before it drops rules.
    """
    out: dict[str, PromptSection] = {}
    for position, items in activation.by_position.items():
        if not items:
            continue
        sec_name = SECTION_NAME_PER_POSITION.get(position)
        if sec_name is None:
            continue
        text = render_position_block(items)
        if not text.strip():
            continue
        # Map system vs user role by section name. Cleaner than
        # exposing a per-position role table — the names are stable.
        role = "user" if sec_name in (
            "lorebook_before_history",
            "lorebook_before_user_message",
        ) else "system"
        out[sec_name] = PromptSection(
            name=sec_name,
            priority=Priority.HIGH,
            text=text,
            role=role,
            max_tokens=2000,  # absolute cap — engine already budgets
        )
    return out


def _build_instruction(
    *,
    card_name: str,
    has_user_message: bool,
    has_pulse: bool,
    last_speaker_name: str,
) -> str:
    """Build the per-turn "you're up now" instruction.

    The instruction adapts to what the speaker is *actually* responding
    to so the LLM gets a coherent target. The four cases:

    1. **Char-to-Char reply** (no user message, prior speaker exists) →
       frame the prior turn as a direct address, like a normal chat.
    2. **Group dynamic with user msg + prior speaker** → instruct to
       weave both in.
    3. **User-only** → react to the user.
    4. **Pulse-only** → react to the impuls.
    """
    head = f"## Du bist jetzt dran: {card_name}"
    style_block = (
        "Antworte in deiner eigenen Stimme — so lang oder so kurz, "
        "wie die Szene es verlangt. Lieber lebendig + detailliert als "
        "gehetzt-knapp."
    )
    actions_block = (
        "**Was du tun sollst**: Beschreibe was DEIN Charakter denkt, "
        "fühlt, sagt und tut — Aktionen und inneres Erleben in "
        "*Sternchen*, gerne mehrsätzig."
    )
    forbid_block = (
        "**Was du NICHT tust**: Sprechen für den User oder andere "
        "Charaktere. Plot vorantreiben, wenn niemand dich darum gebeten "
        "hat. Zeitsprünge oder neue Ereignisse erfinden."
    )
    pass_block = "Wenn du nichts beitragen willst, antworte exakt: [PASS]"

    # Case 1: Char-to-Char reply — last speaker is another character and
    # there's no fresh user line. Treat the prior reply as if it were
    # addressed to us directly.
    if last_speaker_name and not has_user_message and not has_pulse:
        target_block = (
            f"**Reagiere auf {last_speaker_name}.** Du antwortest auf das, "
            f"was {last_speaker_name} gerade gesagt/getan hat (siehe "
            "*Reaktionen dieser Runde*) — als wäre es eine Nachricht "
            "an dich. Sprich {last_speaker_name} bei Bedarf direkt an, "
            "stelle Rückfragen, reagiere emotional. Behandle das "
            "exakt wie eine User-Nachricht — der einzige Unterschied "
            "ist der Absender."
        ).format(last_speaker_name=last_speaker_name)
        return "\n".join([head, target_block, "", actions_block, forbid_block, "", style_block, pass_block])

    # Case 2: Group dynamic — both a user message AND a prior speaker.
    if last_speaker_name and has_user_message:
        target_block = (
            "**Reagiere auf User + die vorigen Reaktionen.** Du sprichst "
            f"nach {last_speaker_name} — ergänze, widersprich oder bring "
            "deine Sicht ein, ohne ihn/sie zu wiederholen."
        )
        return "\n".join([head, target_block, "", actions_block, forbid_block, "", style_block, pass_block])

    # Case 4: Pulse-only.
    if has_pulse and not has_user_message:
        target_block = (
            "**Reagiere auf den Impuls.** Etwas ist gerade passiert (siehe "
            "*Impuls*) — antworte darauf in deiner Rolle."
        )
        return "\n".join([head, target_block, "", actions_block, forbid_block, "", style_block, pass_block])

    # Case 3 (default): User message, no prior speaker.
    target_block = "**Reagiere auf die User-Nachricht.**"
    return "\n".join([head, target_block, "", actions_block, forbid_block, "", style_block, pass_block])


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
    card: CharacterCard,
    all_cards: list[CharacterCard],
    live_state_by_char: dict[str, dict[str, str]] | None = None,
) -> str:
    """Render the '## Andere Anwesende' block with relationship hints.

    Phase 13.5 (C): when ``live_state_by_char`` is provided, each peer
    line ALSO carries their current ``location`` / ``last_action`` so
    this char can react truthfully instead of inventing scenes for
    others. Mike's Castaway log: Sandra hallucinated 'Mira tritt aus
    den Palmen mit einem Becher' while Mira was in the lagoon — the
    LLM had no way to know Mira's actual location was 'lagune'.
    """
    state_by_id = live_state_by_char or {}
    lines: list[str] = []
    for other in all_cards or []:
        if other.id == card.id:
            continue
        rel = card.relationships.get(other.id) or other.relationships.get(
            card.id
        )
        # Build a "currently" suffix from the most truth-y state keys.
        st = state_by_id.get(other.id) or {}
        bits: list[str] = []
        loc = st.get("location") or ""
        if loc:
            bits.append(f"Ort: {loc}")
        last_act = st.get("last_action") or ""
        if last_act:
            bits.append(f"macht: {last_act}")
        mood = st.get("mood") or ""
        if mood:
            bits.append(f"Stimmung: {mood}")
        currently = (" — " + "; ".join(bits)) if bits else ""

        head = f"- {other.name}"
        if rel:
            head += f" ({rel})"
        lines.append(f"{head}{currently}")
    if not lines:
        return ""
    return (
        "## Andere Anwesende — was sie GERADE tun (Wahrheit für deinen Turn)\n"
        + "\n".join(lines)
        + "\n\n**Wichtig**: Erfinde KEINE Aktionen oder Aufenthaltsorte für "
        "diese Charaktere. Nimm die hier genannte Realität als Fakt — "
        "wenn du nicht weisst was sie tun, frag oder reagier nur auf "
        "das, was du in den letzten Reaktionen liest."
    )


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
