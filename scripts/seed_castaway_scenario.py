"""
Seed the *Castaway* RP scenario: 4 characters + a global lorebook.

Phase 13 smoke-test setup. Mike's idea: four women shipwrecked on a
Caribbean island, autonomous simulation every 5 minutes. If they keep
their clothes consistent, never leak memory between two such sessions
and produce plausible state updates over 30 minutes, Phase 13 is shipped.

Why a script instead of UI clicks:
    The character form has lost its state fields in Phase 13 (Mike's
    explicit decision). Per-character clothing defaults still need to
    land in the SQLite ``state`` column so the RP container snapshots
    them correctly into ``data/rp_sessions/<id>/state.json`` on
    ``character_attach``. Setting that via SQL is faster than clicking.

Idempotent: re-running skips already-existing characters and lorebook
entries. Only run side-effects on first invocation.

Usage::

    conda activate lexyai
    python scripts/seed_castaway_scenario.py            # real run
    python scripts/seed_castaway_scenario.py --dry-run  # plan only
    python scripts/seed_castaway_scenario.py --reset    # delete first

The ``--reset`` flag drops ALL four characters and the lorebook before
re-creating them. Existing RP sessions are NOT touched — they keep
their snapshotted state from before the reset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import aiosqlite

# Make ``plugins/`` importable when running the script standalone.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plugins.character_chat.character_card import CharacterCard  # noqa: E402
from plugins.character_chat.character_store import CharacterStore  # noqa: E402
from plugins.character_chat.lorebook_store import (  # noqa: E402
    LorebookStore,
    POSITION_AFTER_PERSONA,
    POSITION_BEFORE_SCENARIO,
    SCOPE_GLOBAL,
)


DB_PATH = ROOT / "data" / "plugins" / "character_chat" / "character_chat.db"
MARKER_PATH = ROOT / "data" / "plugins" / "character_chat" / ".castaway_seeded"
LOREBOOK_NAME = "Stranded — Karibik-Insel"

# ─── Character profiles ─────────────────────────────────────────────

# Shared scenario text — every character sees the same setup so the
# scene framing is consistent across speakers.
SCENARIO = (
    "Du bist mit Sandra, Lena, Mira und Yara nach dem Untergang der "
    "MS Kanaria auf einer einsamen Karibik-Insel gestrandet. Die "
    "anderen drei sind genau wie du nur in dem, womit du geschlafen "
    "hast. Es ist kurz nach Sonnenaufgang."
)

# Phase 13.2 — appended to every persona to break the passive
# "starre-auf-Sand"-loop the first 30-min smoke-test produced. The
# explicit "du HANDELST"-instruction works hand-in-hand with the
# action-erzwingende Pulse-Prompts.
ACTION_DISCIPLINE = (
    " Im RP fragst du nicht hilflos 'Was ist los?' und starrst "
    "nicht endlos auf den Sand — du HANDELST. Jeder deiner Beiträge "
    "enthält mindestens eine konkrete Aktion (sich bewegen, etwas "
    "anfassen, etwas zeigen) ODER eine namentliche Anrede an einen "
    "der anderen Charaktere. Wiederholungen mit den anderen vermeidest "
    "du — wenn jemand 'Salz brennt' sagt, sagst du etwas ANDERES."
)

CHARACTERS: list[dict[str, Any]] = [
    {
        "name": "Sandra",
        "age_stage": "adult",
        "color": "#e0af68",
        "persona": (
            "Sandra, Mitte zwanzig, Krankenschwester aus Hamburg. "
            "Pragmatisch, behält die Übersicht, redet wenig aber "
            "zielführend. Hat schon zwei Bergrettungs-Wochenenden "
            "hinter sich und denkt sofort in Prioritäten: Wasser, "
            "Schatten, Inventur, dann Plan. Findet die Panik der "
            "anderen verständlich, aber nicht hilfreich. Wenn sie "
            "weint, dann allein, nachts, ohne dass es jemand merken "
            "soll." + ACTION_DISCIPLINE
        ),
        "scenario": SCENARIO,
        "greeting": (
            "*Sandra wischt sich nasses Haar aus dem Gesicht und "
            "blickt den Strand entlang.* \"Okay. Atmen. Wer ist hier? "
            "Ist jemand verletzt?\""
        ),
        "example_dialog": "",  # Leer – Phase-12-Lehre.
        "tags": ["castaway", "leader"],
        "proactive_pulse_pattern": "every 8m",
        "proactive_pulse_prompt": (
            "Sandra MACHT eine konkrete Aktion statt nur zu fühlen. "
            "Wähle EINS: prüft Lena/Mira/Yara auf Verletzungen, geht "
            "5 Schritte zum Süßwasser-Bach um zu trinken, sortiert "
            "Schiffstrümmer am Strand, weist jemandem eine konkrete "
            "Aufgabe zu (\"Mira, schau ob du Wasser findest\"), oder "
            "verbindet eine Wunde. SIE HANDELT. Verbote: KEINE "
            "Wiederholung von 'starre auf Sand', 'Salz brennt', "
            "'Kopf dröhnt' — die anderen schreiben das schon. KEIN "
            "passives Sitzen-und-Beobachten."
        ),
        "state": {
            "clothing": "dünnes Nachthemd, knielang, halb durchsichtig",
        },
    },
    {
        "name": "Lena",
        "age_stage": "teen",
        "color": "#f7768e",
        "persona": (
            "Lena, sechzehn, Schülerin, Erste-Klasse-Passagier mit ihren "
            "Eltern (die noch verschollen sind). Klammert sich emotional "
            "an die anderen, weint schnell, hat aber unter der Panik "
            "einen weichen Humor. Ist nie länger als zwei Stunden ohne "
            "ihr Handy gewesen. Hat immer Hunger. Spricht oft erst, wenn "
            "jemand sie anschaut." + ACTION_DISCIPLINE
        ),
        "scenario": SCENARIO,
        "greeting": (
            "*Lena sitzt zusammengekauert im warmen Sand, Knie an die "
            "Brust gezogen, und schniefte leise.* \"Mein… mein Handy "
            "ist weg.\""
        ),
        "example_dialog": "",
        "tags": ["castaway", "youngest"],
        "proactive_pulse_pattern": "every 6m",
        "proactive_pulse_prompt": (
            "Lena REAGIERT auf einen anderen Charakter NAMENTLICH. "
            "Wähle EINS: spricht Sandra direkt an (\"Sandra, ich hab "
            "Hunger\"), folgt Mira zum Wald, fragt Yara was zu tun "
            "ist, klammert sich an Sandras Arm. Sie ist 16 und "
            "ängstlich, ABER KEINE STATUE — sie macht etwas, sie "
            "spricht jemanden an. Verbote: KEIN passives Sand-"
            "Anstarren, KEIN 'mein Kopf dröhnt'-Loop. Sie nutzt einen "
            "Namen einer anderen Person."
        ),
        "state": {
            "clothing": "übergroßes T-Shirt + weißer Slip",
        },
    },
    {
        "name": "Mira",
        "age_stage": "adult",
        "color": "#9ece6a",
        "persona": (
            "Mira, Anfang zwanzig, Surf-Lehrerin auf Sylt. Sportlich, "
            "gebräunt, immer in Bewegung. Schläft am liebsten ohne "
            "alles. Ihr erster Reflex bei jedem Problem ist es zu "
            "probieren statt zu reden — sie klettert, taucht, sucht. "
            "Hält die Stimmung leicht durch trockenen Humor und macht "
            "Sandra's Pläne in Tat um, sobald die ausgesprochen sind."
            + ACTION_DISCIPLINE
        ),
        "scenario": SCENARIO,
        "greeting": (
            "*Mira spuckt Salzwasser aus, schüttelt sich wie ein Hund "
            "und steht auf.* \"Das war knapp. Wer kann schwimmen? Bin "
            "gleich wieder da, ich schau ob da Trümmer am Riff sind.\""
        ),
        "example_dialog": "",
        "tags": ["castaway", "explorer"],
        "proactive_pulse_pattern": "every 7m",
        "proactive_pulse_prompt": (
            "Mira ist SCHON UNTERWEGS oder kommt gerade zurück. "
            "Wähle EINS und beschreibe es konkret: kommt mit einer "
            "Banane / Mango / Kokosnuss aus dem Wald zurück, klettert "
            "gerade auf eine Palme, taucht im Riff nach Trümmern, "
            "läuft zum Süßwasser-Bach mit dem leeren Plastik-"
            "Container. SIE BEWEGT SICH. Sie sagt was sie gefunden "
            "hat oder gleich macht. Verbote: ABSOLUTES VERBOT von "
            "'starre auf den Sand', 'mein Kopf dröhnt', 'Salz auf "
            "der Haut'. Mira sitzt nicht. Mira handelt."
        ),
        "state": {
            "clothing": "nackt",
        },
    },
    {
        "name": "Yara",
        "age_stage": "adult",
        "color": "#bb9af7",
        "persona": (
            "Yara, Ende zwanzig, Architektin aus Wien. Ruhig, beobachtend, "
            "redet selten ohne nachzudenken. Hat die Gabe, eine Situation "
            "in zwei Sätzen zusammenzufassen, die die anderen drei sich "
            "anhören. Schlief am Sonnendeck im Liegestuhl, weil die "
            "Kabine zu eng war. Reagiert auf Stress mit Stille — das "
            "wird oft fälschlich als Kälte gelesen." + ACTION_DISCIPLINE
        ),
        "scenario": SCENARIO,
        "greeting": (
            "*Yara sitzt aufrecht im Sand und beobachtet die anderen, "
            "eine Hand schützend über den Augen gegen die Morgensonne.* "
            "\"Okay. Wir leben. Was wir wissen: Insel, Sonne im Osten, "
            "Wellen aus Süden. Was wir nicht wissen: alles andere.\""
        ),
        "example_dialog": "",
        "tags": ["castaway", "observer"],
        "proactive_pulse_pattern": "every 10m",
        "proactive_pulse_prompt": (
            "Yara macht etwas BEOBACHTBARES. Wähle EINS: zeigt mit "
            "dem Finger auf etwas Konkretes (Rauch, Trümmer im "
            "Wasser, Spur im Sand, ein Vogel über dem Hügel), fasst "
            "die Lage in EINEM Satz zusammen (\"Wir haben Wasser, "
            "aber kein Werkzeug.\"), oder stellt eine konkrete "
            "Frage die voranbringt (\"Wer hat das Süßwasser schon "
            "getestet?\"). KEIN SCHWEIGEN — wenn keiner antwortet, "
            "spricht sie aus was sie gerade sieht. Verbote: KEIN "
            "passives Sitzen, KEIN 'starre in den Sand'-Loop."
        ),
        "state": {
            "clothing": "dunkler Sport-BH + Bikini-Slip",
        },
    },
]

# Mutual relationships, written by character name. Resolved to char_ids
# after creation.
RELATIONSHIPS_BY_NAME: dict[str, dict[str, str]] = {
    "Sandra": {"Lena": "Schwester-Figur", "Mira": "Mit-Anführerin", "Yara": "Verbündete"},
    "Lena":   {"Sandra": "Beschützerin",  "Mira": "große Schwester", "Yara": "ruhige Tante"},
    "Mira":   {"Sandra": "Strategin",     "Lena": "kleine Schwester", "Yara": "Stratege-Buddy"},
    "Yara":   {"Sandra": "Pragmatikerin", "Lena": "Schützling",       "Mira": "Aktionspartnerin"},
}

# ─── Lorebook entries ───────────────────────────────────────────────

LORE_ENTRIES: list[dict[str, Any]] = [
    {
        "name": "Hintergrund: MS Kanaria",
        "always_on": True,
        "position": POSITION_BEFORE_SCENARIO,
        "priority": 10,
        "keys": [],
        "content": (
            "Die MS Kanaria, ein mittelgroßes Kreuzfahrtschiff, geriet "
            "in der Nacht in einen schweren Sturm. Wassereinbruch im "
            "Maschinenraum gegen 3 Uhr früh. Notfall-Evakuierung lief "
            "chaotisch — viele schliefen, manche schafften es zu den "
            "Rettungsbooten, viele nicht. Sandra, Lena, Mira und Yara "
            "fanden sich an einer Rettungsboje fest und trieben "
            "zusammen ab. Vom Schiff ist nichts mehr zu sehen."
        ),
    },
    {
        "name": "Insel-Geographie",
        "always_on": True,
        "position": POSITION_BEFORE_SCENARIO,
        "priority": 20,
        "keys": [],
        "content": (
            "Die Insel ist klein — etwa 2 km Durchmesser. Im Norden "
            "eine flache, sandige Bucht (dort wurden sie angespült). "
            "Im Inneren ein bewaldeter Hügel, ca. 80 m hoch. Im Süden "
            "felsiges Riff mit gefährlicher Brandung. Westhang offener "
            "Hang mit Palmen. Keine Anzeichen menschlicher Bewohnung. "
            "Keine Stromleitungen, keine Dörfer, keine Straßen."
        ),
    },
    {
        "name": "Tropisches Wetter",
        "always_on": True,
        "position": POSITION_AFTER_PERSONA,
        "priority": 30,
        "keys": [],
        "content": (
            "Tagsüber ~30°C, hohe Luftfeuchtigkeit, oft windstill. "
            "Mittagshitze macht körperliche Anstrengung schwer — "
            "kluge Charaktere ruhen zwischen 12-15 Uhr im Schatten. "
            "Nachts kühlt es auf ~22°C ab, ohne Decke fröstelt man. "
            "Etwa alle zwei Tage ein kurzer tropischer Regenschauer."
        ),
    },
    {
        "name": "Realismus-Disziplin",
        "always_on": True,
        "position": POSITION_AFTER_PERSONA,
        "priority": 40,
        "keys": [],
        "content": (
            "Keine Smartphones, kein Helikopter aus dem Himmel, kein "
            "magisches Auftauchen von Werkzeugen. Was die Frauen haben, "
            "müssen sie improvisieren — Steine, Holz, Lianen, was die "
            "Brandung anschwemmt. Verletzungen brauchen Tage zum Heilen. "
            "Ohne Werkzeug ist Feuer-Machen sehr schwer. Plot-Sprünge "
            "sind verboten — die Story bleibt am Boden der Tatsachen."
        ),
    },
    {
        "name": "Süßwasser-Fluss",
        "always_on": False,
        "position": POSITION_BEFORE_SCENARIO,
        "priority": 100,
        "keys": ["wasser", "fluss", "trinken", "durst", "bach", "quelle"],
        "content": (
            "Aus dem Inland kommt ein kleiner Bach — kühl, klar, sicher "
            "trinkbar. Mündet ~5 Minuten Fußmarsch östlich vom "
            "Strandungs-Punkt ins Meer. Quelle weiter oben am Hügel. "
            "Mira hat ihn auf einem ihrer ersten Erkundungs-Trips "
            "entdeckt."
        ),
    },
    {
        "name": "Kokospalmen",
        "always_on": False,
        "position": POSITION_BEFORE_SCENARIO,
        "priority": 110,
        "keys": ["kokos", "palme", "klettern", "baum"],
        "content": (
            "Ein Hain von etwa zwölf Kokospalmen am Westhang. Reife "
            "Nüsse hängen 8-12 m hoch — Klettern oder Werfen. Einige "
            "fallen von selbst (gut zu hören am Abend). Aufbrechen "
            "ohne Werkzeug ist mühsam — bester Trick: an einem "
            "spitzen Stein zerschmettern."
        ),
    },
    {
        "name": "Bananen-Stauden",
        "always_on": False,
        "position": POSITION_BEFORE_SCENARIO,
        "priority": 120,
        "keys": ["banane", "gelb", "staude", "frucht"],
        "content": (
            "Im bewaldeten Inneren stehen drei Bananen-Stauden, gemischt "
            "mit reifen und unreifen Früchten. Erreichbar ohne Klettern. "
            "Reife Früchte sind weich-gelb mit ein paar braunen Punkten — "
            "unreife (grün, hart) verursachen Bauchweh."
        ),
    },
    {
        "name": "Mango-Bäume",
        "always_on": False,
        "position": POSITION_BEFORE_SCENARIO,
        "priority": 130,
        "keys": ["mango", "frucht", "saft"],
        "content": (
            "Drei Mango-Bäume am Fluss-Lauf. Reife Mangos fallen von "
            "selbst — saftig, süß, klebrig. Mit einem Stein lassen sich "
            "höher hängende Früchte abwerfen. Die Steine vom Strand "
            "eignen sich gut."
        ),
    },
    {
        "name": "Strand & Lagune",
        "always_on": False,
        "position": POSITION_BEFORE_SCENARIO,
        "priority": 140,
        "keys": ["strand", "sand", "lagune", "meer", "ufer"],
        "content": (
            "Heller, feiner Sand entlang der Nordbucht — etwa 300 m "
            "lang. Im westlichen Teil wird die Bucht zu einer flachen "
            "Lagune (knietief, sicher zum Schwimmen). Im östlichen Teil "
            "felsige Kanten, dort hat die Brandung schon Schiffstrümmer "
            "abgelegt."
        ),
    },
    {
        "name": "Schiffstrümmer",
        "always_on": False,
        "position": POSITION_BEFORE_SCENARIO,
        "priority": 150,
        "keys": ["trümmer", "wrack", "schiff", "holz", "koffer"],
        "content": (
            "Verstreute Holzplanken im östlichen Strandabschnitt. Ein "
            "halb aufgebrochener Koffer (drinnen feuchte Kleidung, "
            "ruiniertes Buch). Ein leerer 20-Liter-Plastik-Container, "
            "intakt — wird Wasser-Speicher. Mehrere Seile, von Salz "
            "noch elastisch. KEIN Funkgerät, keine Pistole, keine "
            "Konserven — vor allem hat das Schiff ihre Snacks nicht "
            "ausgespuckt."
        ),
    },
    {
        "name": "Nacht auf der Insel",
        "always_on": False,
        "position": POSITION_BEFORE_SCENARIO,
        "priority": 160,
        "keys": ["nacht", "dunkel", "schlafen", "sterne", "müde"],
        "content": (
            "Nach Sonnenuntergang wird es schnell sehr dunkel. Sehr "
            "viele Sterne, kein Lichtschmutz. Tropische Insektengeräusche, "
            "Brandung im Hintergrund. Manchmal raschelt etwas im "
            "Unterholz — vermutlich Krabben, vielleicht Vögel, "
            "garantiert keine Raubtiere (die Insel ist zu klein). "
            "Ohne Feuer wird die Nacht kühl — die vier rücken instinktiv "
            "zusammen."
        ),
    },
]


# ─── Helpers ─────────────────────────────────────────────────────────


def _log(prefix: str, msg: str) -> None:
    print(f"[{prefix}] {msg}", flush=True)


async def _resolve_lorebook_id(
    lore: LorebookStore, name: str,
) -> str | None:
    books = await lore.list_lorebooks()
    for book in books:
        if book.name.strip().lower() == name.strip().lower():
            return book.id
    return None


async def _resolve_entry_names(
    lore: LorebookStore, lorebook_id: str,
) -> set[str]:
    entries = await lore.list_entries(lorebook_id=lorebook_id)
    return {e.name.strip().lower() for e in entries}


# ─── Reset ───────────────────────────────────────────────────────────


async def _reset(
    db: aiosqlite.Connection, store: CharacterStore, lore: LorebookStore,
    *, dry_run: bool,
) -> None:
    """Drop the four named characters and the named lorebook (incl. entries)."""
    _log("reset", "starting…")
    # Characters: by name, archived or not.
    for name in [c["name"] for c in CHARACTERS]:
        existing = await store.get_by_name(name, include_archived=True)
        if existing is None:
            _log("reset", f"character {name!r} not present, skip")
            continue
        if dry_run:
            _log("reset", f"WOULD delete character {name!r} (id={existing.id})")
        else:
            await db.execute(
                "DELETE FROM characters WHERE id = ?", (existing.id,),
            )
            _log("reset", f"deleted character {name!r}")
    # Lorebook: by name.
    book_id = await _resolve_lorebook_id(lore, LOREBOOK_NAME)
    if book_id is None:
        _log("reset", f"lorebook {LOREBOOK_NAME!r} not present, skip")
    else:
        if dry_run:
            _log("reset", f"WOULD delete lorebook {book_id} (incl. entries)")
        else:
            ok = await lore.delete_lorebook(book_id)
            _log("reset", f"deleted lorebook {book_id} → {ok}")
    if not dry_run:
        await db.commit()
    if MARKER_PATH.exists() and not dry_run:
        MARKER_PATH.unlink()
        _log("reset", "marker removed")
    _log("reset", "done")


# ─── Seed ────────────────────────────────────────────────────────────


async def _seed_characters(
    store: CharacterStore, *, dry_run: bool, update: bool,
) -> dict[str, str]:
    """Create or update characters; return ``{name: char_id}`` for ALL four.

    ``update=True`` overwrites every field of an existing character with
    the spec values. ``update=False`` (default) leaves existing
    characters untouched and only creates missing ones.
    """
    name_to_id: dict[str, str] = {}
    for spec in CHARACTERS:
        name = spec["name"]
        existing = await store.get_by_name(name)
        if existing is not None:
            name_to_id[name] = existing.id
            if not update:
                _log("=", f"character: {name:<8} (id={existing.id})  ← already exists, skip")
                continue
            if dry_run:
                _log("~", f"character: {name:<8} (id={existing.id})  ← WOULD update")
                continue
            await store.update(
                existing.id,
                persona=spec["persona"],
                greeting=spec["greeting"],
                scenario=spec["scenario"],
                example_dialog=spec["example_dialog"],
                color=spec["color"],
                age_stage=spec["age_stage"],
                tags=list(spec["tags"]),
                state=dict(spec["state"]),
                proactive_pulse_pattern=spec["proactive_pulse_pattern"],
                proactive_pulse_prompt=spec["proactive_pulse_prompt"],
            )
            _log("~", f"character: {name:<8} (id={existing.id})  ← updated")
            continue
        if dry_run:
            _log("+", f"character: {name:<8} (WOULD create)")
            name_to_id[name] = "<dry-run>"
            continue
        # Build CharacterCard with the full spec — pydantic will normalise.
        card = CharacterCard(
            name=name,
            persona=spec["persona"],
            greeting=spec["greeting"],
            scenario=spec["scenario"],
            example_dialog=spec["example_dialog"],
            color=spec["color"],
            age_stage=spec["age_stage"],
            tags=list(spec["tags"]),
            state=dict(spec["state"]),
            proactive_pulse_pattern=spec["proactive_pulse_pattern"],
            proactive_pulse_prompt=spec["proactive_pulse_prompt"],
        )
        created = await store.create(card)
        name_to_id[name] = created.id
        _log("+", f"character: {name:<8} (id={created.id})  ← created")
    return name_to_id


async def _seed_relationships(
    store: CharacterStore,
    name_to_id: dict[str, str],
    *,
    dry_run: bool,
) -> None:
    """Update each character's ``relationships`` dict with peer char_ids."""
    if dry_run:
        _log("i", "WOULD wire relationships (4 chars × 3 peers)")
        return
    for name, peers in RELATIONSHIPS_BY_NAME.items():
        char_id = name_to_id.get(name)
        if not char_id or char_id == "<dry-run>":
            continue
        # peer name → peer id
        rel_dict = {
            name_to_id[peer_name]: label
            for peer_name, label in peers.items()
            if peer_name in name_to_id
        }
        # Don't blow away existing relationships from other contexts —
        # merge defensively.
        current = await store.get(char_id)
        if current is None:
            continue
        merged = {**(current.relationships or {}), **rel_dict}
        if merged == (current.relationships or {}):
            continue  # no change
        await store.update(char_id, relationships=merged)
    _log("i", "relationships wired (4 chars × 3 peers)")


async def _seed_lorebook(
    lore: LorebookStore, *, dry_run: bool,
) -> tuple[str | None, int, int]:
    """Create the lorebook and entries; return (book_id, created, skipped)."""
    book_id = await _resolve_lorebook_id(lore, LOREBOOK_NAME)
    created_now = False
    if book_id is None:
        if dry_run:
            _log("+", f"lorebook: {LOREBOOK_NAME!r} (WOULD create, scope=global)")
            return None, 0, 0
        book = await lore.create_lorebook(
            name=LOREBOOK_NAME,
            description=(
                "Phase-13-Smoke-Test — Setup für ein Castaway-Szenario "
                "auf einer einsamen Karibik-Insel. Wird automatisch in "
                "RP-Sessions referenziert (scope=global)."
            ),
            scope=SCOPE_GLOBAL,
            token_budget=1500,
        )
        book_id = book.id
        created_now = True
        _log("+", f"lorebook: {LOREBOOK_NAME!r} (id={book_id})  ← created")
    else:
        _log("=", f"lorebook: {LOREBOOK_NAME!r} (id={book_id})  ← already exists")

    # Entries
    if book_id is None:
        return None, 0, 0
    existing_names = await _resolve_entry_names(lore, book_id)
    created_entries = 0
    skipped_entries = 0
    for spec in LORE_ENTRIES:
        name = spec["name"]
        if name.strip().lower() in existing_names:
            skipped_entries += 1
            continue
        if dry_run:
            _log("+", f"  entry: {name}  (WOULD create)")
            created_entries += 1
            continue
        await lore.create_entry(
            lorebook_id=book_id,
            name=name,
            keys=spec["keys"],
            content=spec["content"],
            position=spec["position"],
            priority=spec["priority"],
            always_on=spec["always_on"],
        )
        created_entries += 1
        _log("+", f"  entry: {name}")
    if not dry_run and not created_now:
        _log(
            "i",
            f"lorebook entries: {created_entries} new, {skipped_entries} kept",
        )
    return book_id, created_entries, skipped_entries


# ─── Main ────────────────────────────────────────────────────────────


def _print_postscript(stats_str: str) -> None:
    print(
        "\nDone. Now in the UI:\n"
        '  1. RP-tab → "Neue RP-Session"\n'
        '  2. Name: "Castaway-Test"\n'
        '  3. Szene: "Vier Frauen werden im Morgengrauen am Strand einer\n'
        '            Karibik-Insel angespült."\n'
        f'  4. State-Tracking: {stats_str}\n'
        "  5. Anhängen: Sandra, Lena, Mira, Yara\n"
        '  6. Lorebook: "Stranded — Karibik-Insel" ist global → wird\n'
        "     automatisch geladen, kein Pin nötig.\n"
        '  7. Character-Mode: "Nur Charaktere" (Mode 1)\n'
        "  8. Autonome Simulation: alle 5 min — Start\n"
    )


async def _main(args: argparse.Namespace) -> int:
    if not DB_PATH.exists():
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        # The plugin will create the file with full schema on first
        # boot, but we can pre-create empty + init schema right here.

    db = await aiosqlite.connect(str(DB_PATH))
    try:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.commit()

        store = CharacterStore(db)
        await store.init_schema()
        lore = LorebookStore(db)
        await lore.init_schema()

        if args.reset:
            await _reset(db, store, lore, dry_run=args.dry_run)
            if args.reset_only:
                return 0

        name_to_id = await _seed_characters(
            store, dry_run=args.dry_run, update=args.update,
        )
        await _seed_relationships(store, name_to_id, dry_run=args.dry_run)
        book_id, created, skipped = await _seed_lorebook(
            lore, dry_run=args.dry_run,
        )

        if not args.dry_run:
            payload = {
                "seeded_at": time.time(),
                "characters": name_to_id,
                "lorebook_id": book_id,
                "lorebook_entries_created": created,
                "lorebook_entries_skipped": skipped,
            }
            MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
            MARKER_PATH.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        stats_str = (
            "clothing; hunger=satt; durst=gestillt; erregung=neutral; "
            "mood=schock; energy=erschöpft; location=strand"
        )
        _print_postscript(stats_str)
    finally:
        await db.close()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Seed the Castaway Phase-13 smoke-test scenario.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen, don't write to the DB.",
    )
    p.add_argument(
        "--reset", action="store_true",
        help="Delete the four characters + the lorebook before seeding.",
    )
    p.add_argument(
        "--reset-only", action="store_true",
        help="Only reset (skip the re-seed). Implies --reset.",
    )
    p.add_argument(
        "--update", action="store_true",
        help=(
            "Overwrite existing characters' persona/greeting/state/etc "
            "with the script's specs. Default: skip existing names."
        ),
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.reset_only:
        args.reset = True
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
