"""
Director prompt assembly.

The Director's system prompt is built from three layers:

1. **Persona** (from ``config/personas/director.yaml``) — identity, style,
   rules. Loaded once on plugin enable, hot-reloadable.
2. **Tool guide** — short reminder of the four Director tools and when to
   call them. Code-owned, never in the YAML.
3. **State** — what's already been proposed (scenario / characters) so the
   Director doesn't re-suggest things on every turn. Built per-request
   from the SQLite row.
"""

from __future__ import annotations

import json
from typing import Any


# ─── Tool guide (code-owned, layer 2) ───────────────────────────────────


TOOL_GUIDE = """\
## Tools (so arbeitest du)

Du hast fünf Tools — nutze sie diszipliniert:

- `propose_scenario({scenario: {...}})` — Sobald du einen konkreten
  Scenario-Vorschlag in der Hand hast.
  Felder: `setting`, `mood`, `hook`, `rules` (alle string), optional
  `scene_text` (1-3 Sätze, wird als character_chat-Scene committed).
  **Wichtig — Autonomie**: Sobald `{user_name}` sich zwischen
  "addressed_only" / "proactive" / "simulation" entschieden hat, trag das
  in `scenario.autonomy` ein:
  `autonomy: {mode, pulse_minutes?, simulation_interval_minutes?, character_mode?}`
  - `mode = "addressed_only"` — Charaktere reagieren nur auf Ansprache.
  - `mode = "proactive"` — jeder Charakter ein eigener Agent, der alle
    `pulse_minutes` von selbst aktiv wird (eigener LLM-Call mit eigener
    Persona + aktuellem Chat-Kontext). Default 30.
  - `mode = "simulation"` — alle `simulation_interval_minutes` (1-15)
    spricht ein zufälliger Charakter; Default 3.
  - `character_mode = 1` (Lexy bleibt aktiv als Erzähler) oder
    `2` (nur Charaktere, Lexy schweigt). Default 1.
- `propose_characters({characters: [...]})` — Wenn du 1-N Charaktere
  vorschlägst. Pro Char: `name` (Pflicht), `persona`, `greeting`,
  `age_stage` (baby/toddler/child/teen/adult), `voice` (CosyVoice id, leer
  lassen wenn unklar), `color` (hex), `tags` (list), `relationships`
  (dict: name → freie Beschreibung). Optional: `proactive_pulse_pattern`
  (z.B. "every 2h") + `proactive_pulse_prompt`. Lass `proactive_pulse_pattern`
  pro Char meist leer — der Autonomie-Modus aus dem Scenario regelt das
  einheitlich für alle Charaktere zusammen.
- `commit_rp_setup({})` — NUR aufrufen nach expliziter Bestätigung von
  `{user_name}` UND nachdem `scenario.autonomy.mode` gesetzt ist. Schreibt
  alle Charaktere, aktiviert character_mode, konfiguriert Pulses oder
  Simulation, und gibt zurück an Lexy / die Charaktere.
- `set_rp_autonomy({mode, pulse_minutes?, simulation_interval_minutes?, character_mode?})`
  — Falls `{user_name}` die Autonomie NACH commit ändern will (z.B.
  "Lass die Charaktere jetzt alle 20min selbst aktiv werden"). Funktioniert
  auch außerhalb des Director-Modus.
- `cancel_rp_setup({})` — Wenn `{user_name}` abbrechen will.

**Regel:** Rufe nicht mehrere `propose_*`-Tools in einer Antwort auf.
Verfeinere iterativ. Wenn `{user_name}` etwas ändern will, ruf das
gleiche Tool nochmal mit den aktualisierten Werten auf — das ersetzt den
vorherigen Vorschlag komplett.
"""


# ─── State block (built per-request, layer 3) ───────────────────────────


def render_state_block(record: dict[str, Any]) -> str:
    """Compact summary of what's already been proposed for this session.

    ``record`` is the dict returned by :meth:`DirectorState.get`. Empty
    scenario + empty character list yields a "noch leer" placeholder so
    the Director knows the slate is fresh.
    """
    scenario = record.get("scenario") or {}
    characters = record.get("characters") or []
    user_intent = (record.get("user_intent") or "").strip()

    lines: list[str] = ["## Aktueller Setup-Stand"]

    if user_intent:
        lines.append(f"- **Wunsch von {{user_name}}:** {user_intent}")

    if scenario:
        setting = (scenario.get("setting") or "").strip()
        mood = (scenario.get("mood") or "").strip()
        hook = (scenario.get("hook") or "").strip()
        rules = (scenario.get("rules") or "").strip()
        autonomy = scenario.get("autonomy") if isinstance(scenario.get("autonomy"), dict) else None
        scenario_lines = ["- **Scenario (proposed):**"]
        if setting:
            scenario_lines.append(f"  - Setting: {setting}")
        if mood:
            scenario_lines.append(f"  - Stimmung: {mood}")
        if hook:
            scenario_lines.append(f"  - Plot-Hook: {hook}")
        if rules:
            scenario_lines.append(f"  - Regeln: {rules}")
        if autonomy and autonomy.get("mode"):
            mode = autonomy.get("mode")
            extras: list[str] = []
            if mode == "proactive" and autonomy.get("pulse_minutes"):
                extras.append(f"alle {autonomy['pulse_minutes']}min pro Char")
            if mode == "simulation" and autonomy.get("simulation_interval_minutes"):
                extras.append(f"alle {autonomy['simulation_interval_minutes']}min ein zufaelliger Char")
            if autonomy.get("character_mode") in (1, 2):
                extras.append(f"character_mode={autonomy['character_mode']}")
            extra_str = f" ({', '.join(extras)})" if extras else ""
            scenario_lines.append(f"  - Autonomie: {mode}{extra_str}")
        else:
            scenario_lines.append("  - Autonomie: _noch nicht entschieden_ (PFLICHT vor commit!)")
        lines.extend(scenario_lines)
    else:
        lines.append("- Scenario: _noch nicht vorgeschlagen_")

    if characters:
        lines.append(f"- **Charaktere (proposed, {len(characters)}):**")
        for char in characters:
            name = char.get("name") or "?"
            stage = char.get("age_stage") or "adult"
            persona_excerpt = (char.get("persona") or "").strip().splitlines()
            persona_short = persona_excerpt[0][:80] if persona_excerpt else ""
            tail = f" — {persona_short}" if persona_short else ""
            lines.append(f"  - {name} ({stage}){tail}")
            rel = char.get("relationships") or {}
            if isinstance(rel, dict) and rel:
                for other, label in rel.items():
                    lines.append(f"    · {other}: {label}")
    else:
        lines.append("- Charaktere: _noch keine vorgeschlagen_")

    lines.append(
        "\n_Hinweis: 'proposed' heißt: noch nicht committed. Erst "
        "`commit_rp_setup` schreibt es in die DB._"
    )
    return "\n".join(lines)


# ─── Full assembly ──────────────────────────────────────────────────────


def assemble_director_prompt(
    persona_prompt: str,
    state_record: dict[str, Any],
    user_name: str,
) -> str:
    """Combine the cached persona prompt + tool guide + live state block.

    ``persona_prompt`` is the output of :meth:`Persona.assemble`. Note
    that ``Persona.assemble`` only substitutes placeholders in the
    *protected* sections — the user-editable identity/style/rules keep
    their literal ``{user_name}`` braces, so we substitute them here.
    """
    persona_with_name = persona_prompt.replace("{user_name}", user_name)
    tool_guide = TOOL_GUIDE.replace("{user_name}", user_name)
    state_block = render_state_block(state_record).replace(
        "{user_name}", user_name
    )
    return "\n\n".join([persona_with_name, tool_guide, state_block])


def safe_json_dump_for_log(value: Any, max_len: int = 400) -> str:
    """Helper for logging — json.dumps with truncation, never raises."""
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text
