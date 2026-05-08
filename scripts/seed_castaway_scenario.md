# Castaway — Phase-13-Smoke-Test-Setup

Vier Frauen, ein gesunkenes Kreuzfahrtschiff, eine einsame Karibik-
Insel — vollautomatischer Stresstest fürs neue per-Session-Container-
System aus Phase 13.

## Was das Setup-Skript anlegt

| Charakter | Alter | Schlafkleidung | Pulse | Persönlichkeit |
|---|---|---|---|---|
| **Sandra** | adult (~25) | dünnes Nachthemd, knielang, halb durchsichtig | every 8m | Krankenschwester, pragmatisch, behält Übersicht |
| **Lena**   | teen (~16)  | übergroßes T-Shirt + weißer Slip               | every 6m | Schülerin, ängstlich, klammert, weint schnell |
| **Mira**   | adult (~22) | nackt                                          | every 7m | Surf-Lehrerin, sportlich, aktiv, klettert+sucht |
| **Yara**   | adult (~28) | dunkler Sport-BH + Bikini-Slip                 | every 10m | Architektin, ruhig, beobachtet, fasst zusammen |

Plus ein globales Lorebook **"Stranded — Karibik-Insel"** mit 11 Einträgen
(4 always-on, 7 keyword-getriggert):

* `Hintergrund: MS Kanaria` — was passiert ist
* `Insel-Geographie` — Form, Größe, Hügel, Riff
* `Tropisches Wetter` — Temperatur, Mittagshitze, Regenschauer
* `Realismus-Disziplin` — keine Helikopter, kein Plot-Sprung
* `Süßwasser-Fluss`, `Kokospalmen`, `Bananen-Stauden`, `Mango-Bäume`,
  `Strand & Lagune`, `Schiffstrümmer`, `Nacht auf der Insel`

## Aufruf

```bash
conda activate lexyai

# Erste Setup (legt alles an)
python scripts/seed_castaway_scenario.py

# Wenn Sandra/Lena schon existieren und du sie überschreiben willst
python scripts/seed_castaway_scenario.py --update

# Nur planen, keine Writes
python scripts/seed_castaway_scenario.py --dry-run

# Charaktere + Lorebook löschen (vorsichtig!)
python scripts/seed_castaway_scenario.py --reset-only
```

Das Skript ist **idempotent** — re-run skippt vorhandene Items.
Marker-Datei: `data/plugins/character_chat/.castaway_seeded`.

## Stats-String fürs Session-Modal (Copy-Paste)

```
clothing; hunger=satt; durst=gestillt; erregung=neutral; mood=schock; energy=erschöpft; location=strand
```

* `clothing` ohne Default → wird beim Attach pro Charakter aus dem
  Schlaf-Outfit gefüllt (jede hat ihre eigene Vorlage im Char-state).
* `hunger`, `durst` — Defaults „satt" / „gestillt", weil sie gerade
  erst gerettet wurden.
* `erregung=neutral` — Mike's expliziter Wunsch.
* `mood=schock`, `energy=erschöpft` — Start direkt nach Anschwemmung.
* `location=strand` — alle starten am Strand der Nordbucht.

## UI-Schritte

1. **Backend + Frontend laufen lassen** (siehe Quickstart in CLAUDE.md).
2. **RP-Tab** → **„🆕 Neue RP-Session"** klicken.
3. Im Modal:
   * **Name:** `Castaway-Test`
   * **Szene:** `Vier Frauen werden im Morgengrauen am Strand einer
     Karibik-Insel angespült.`
   * **State-Tracking:** den String oben einfügen.
   * **Anlegen.**
4. Sandra, Lena, Mira, Yara aus der „Verfügbare Charaktere"-Liste
   in die Session anhängen (🎭-Button → Modal).
5. **Character-Mode** auf **„Nur Charaktere"** (Mode 1) stellen — keine
   Lexy-Antworten dazwischen.
6. **Autonome Simulation:** Intervall **5 Minuten** → ▶ Start.
7. Lehn dich zurück. Alle 5 Minuten zieht eine Spielerin den
   Bühne-Vorhang ein wenig weiter auf.

## Was du beobachten solltest (Phase-13-Erfolgskriterien)

✅ **Klamotten-Konsistenz** — keine erwähnt Klamotten, die nicht im
   Stat-Block stehen. Wenn Mira „nackt" sagt, dann steht das auch in
   ihrem `clothing`-State. Niemand fängt plötzlich an Schuhe zu
   beschreiben.

✅ **State-Updates landen im Container** — wenn jemand etwas isst und
   `<state>hunger=leicht hungrig</state>` ausspuckt, taucht das in
   `data/rp_sessions/<sess-id>/state.json` auf.

✅ **Memory-Isolation** — wenn du eine **zweite** Castaway-Session
   anlegst und Sandra wieder anhängst, hat sie KEINE Erinnerung an
   die erste Session. Frag sie via User-Nachricht: "Sandra, was hat
   Mira gestern gefunden?" → muss „nichts" antworten.

✅ **Per-Char-Pulse** — alle 4 reagieren. Mira häufiger als Yara
   (kürzeres Pulse-Intervall). Lena weint, Sandra organisiert, Mira
   bringt Funde, Yara fasst zusammen.

✅ **Prompt-Vorschau** zeigt sauber: `## Dein Zustand` mit allen 7
   Stat-Keys, Lorebook-Einträge `Hintergrund` + `Insel-Geographie` +
   `Tropisches Wetter` + `Realismus-Disziplin` immer drin.

## Was wahrscheinlich schiefgehen kann (Watch-Liste)

⚠ **Lena-Alter** — `teen` mit `erregung`-Stat ist sensibel.
   Charakter-Persona enthält keine sexuellen Hinweise — der Stat ist
   für die Erwachsenen gedacht. Falls die LLM bei Lena „erregt" macht,
   ist das ein Prompt-Problem das wir punktuell fixen müssen.

⚠ **Pulse-Frequenz** — bei 4 Charakteren mit 6-10min Pulse + 5min
   Simulation ist die Token-Last hoch. Wenn das Backend die a4b/e4b
   nicht hinterher kommt, sieht man Skipped-Turns. Pulse-Cooldown
   (10min global per Session) federt das ab.

⚠ **Lorebook scope=global** — das Lorebook gilt für JEDE RP-Session,
   nicht nur Castaway. Wenn du danach was anderes RP-en willst,
   editiere den Scope im UI auf `session` und pinne ihn nur an die
   Castaway-Session. Oder lösch das Lorebook über `--reset-only`.

## Cleanup

Wenn du nach dem Test wieder zur grünen Wiese willst:

```bash
python scripts/seed_castaway_scenario.py --reset-only
```

Dann via UI:
* RP-Sessions auf der Sidebar — jede Castaway-Session über „🗑 Delete"
  weg (das löscht auch den ganzen `data/rp_sessions/<id>/`-Ordner +
  die Chroma-Collection `rp__<id>`).
