# Lexy AI — Claude Code Instructions

> **Lexy AI** ist ein lokaler KI-Assistent mit Voice, Memory, Plugin-System und externen Channels.
> Alles läuft lokal (keine Cloud-APIs), basierend auf Gemma 4 LLMs via llama.cpp.

## Quickstart

```bash
# 1. Conda Environment aktivieren
conda activate lexy

# 2. Lexy starten
python -m lexy_core

# 3. Frontend starten (separates Terminal)
cd frontend && npm run dev
```

## Knowledge Base (Obsidian) — AKTIVES MEMORY SYSTEM

> **WICHTIG**: Die Knowledge Base ist dein EXTERNES GEDÄCHTNIS.
> Du LIEST daraus VOR jeder Implementierung.
> Du SCHREIBST hinein NACH jeder Implementierung.
> Das ist kein passives Archiv — es ist dein lebendes Wissensmanagement!

**Verzeichnis (voller Zugriff, lesen UND schreiben!):**
`G:\DATEN\Obsidian\KI-Memory\KI-Memory\10-projects\lexy\`

### Bei diesem Verzeichnis bitte eine Junction erstellen (einmalig):
```batch
mklink /J "G:\AI\LexyAI\docs" "G:\DATEN\Obsidian\KI-Memory\KI-Memory\10-projects\lexy"
```
Falls die Junction existiert, sind die Docs auch unter `docs/` erreichbar.
Falls nicht, nutze die absoluten Pfade unten.

### VOR jeder Code-Änderung — LIES:
- `architecture/overview.md` — Systemdiagramm, Schichten, Datenfluss
- `architecture/services.md` — Alle Services & Ports & Hardware
- `architecture/mvp.md` — MVP Definition, Phase 0-2, Requirements
- `contracts.md` — **SINGLE SOURCE OF TRUTH** für Parameter, Endpoints, Types
- `connections.md` — Aufrufketten (welche Funktion ruft was auf)
- `patterns/coding-style.md` — Naming, Logging, Error Handling

### NACH jeder Code-Änderung — SCHREIBE:
1. **Contracts updaten** (`contracts.md`) — Neue Parameter, Endpoints, Types eintragen
2. **Connections updaten** (`connections.md`) — Neue Aufrufketten dokumentieren
3. **Modul-Docs updaten** (`modules/<name>.md`) — Neue Module dokumentieren
4. **Changelog updaten** (`changelog.md`) — Was wurde geändert und warum
5. **Architecture updaten** — Falls sich die Architektur ändert

### Subsystem-Docs (lesen + bei Änderungen updaten):
- `architecture/plugin-system.md` — PluginAPI, Lifecycle, Manifest
- `architecture/event-system.md` — EventBus, HookManager, Signals
- `architecture/memory-system.md` — ChromaDB, HybridSearch, Dreaming
- `architecture/voice-system.md` — VoiceManager, STT/TTS Provider
- `architecture/channel-system.md` — WhatsApp, Discord, Telegram
- `contracts.md` — Parameter-Namen, API Endpunkte, Typen (**SINGLE SOURCE OF TRUTH**)
- `connections.md` — Vollständige Aufrufketten
- `patterns/coding-style.md` — Naming, Error Handling, Logging
- `_workspace/SYNTHESE-lexy-v2-final.md` — Gesamtplanung (700+ Zeilen)

## Lexy v1 Source Code (Portierungs-Referenz)

**Bewährter Code zum Portieren liegt in:**
`G:\AI\Lexy_OS\lexy_core\` (oder via Junction: `lexy_v1\lexy_core\`)

**Direkt portieren (nur print→structlog, Type Hints ergänzen):**
- `events/event_bus.py` → EventBus mit Wildcards
- `events/hooks.py` → 3-Typen HookManager (void/modifying/sync)
- `events/signals.py` → Thread-sicherer Shared State
- `plugin_system/base_plugin.py` → BasePlugin ABC
- `plugin_system/plugin_api.py` → PluginAPI Facade (Auto-Cleanup!)
- `plugin_system/plugin_loader.py` → Discovery + Topo-Sort
- `plugin_system/plugin_manifest.py` → YAML Manifest Parser
- `voice/voice_manager.py` → Provider-agnostisch
- `voice/stt_base.py` / `tts_base.py` → Provider ABCs
- `tools/tool_registry.py` → Tool Registration + Schema
- `tools/tool_caller.py` → LLM Output Parsing

**Plugins zum Portieren:**
- `G:\AI\Lexy_OS\plugins\voice_cosyvoice\` → CosyVoice TTS (Narrator+Emotion Pipeline!)
- `G:\AI\Lexy_OS\plugins\voice_canary\` → Canary STT (Fallback)
- `G:\AI\Lexy_OS\plugins\scheduler\` → Timer, Wecker, Impulse
- `G:\AI\Lexy_OS\plugins\autonomous_thinking\` → 4-Modi Denk-Engine
- `G:\AI\Lexy_OS\plugins\game_bridge\` → Minecraft/Factorio RCON
- `G:\AI\Lexy_OS\plugins\messaging_gateway\` → Channel-System Basis
- `G:\AI\Lexy_OS\plugins\web_crawler\` → Web Crawl + Knowledge
- `G:\AI\Lexy_OS\plugins\weather\` → Open-Meteo (kostenlos)

## Referenzprogramme (Analyse-Ergebnisse)

**Detaillierte Analysen liegen in Obsidian:**
`G:\DATEN\Obsidian\KI-Memory\KI-Memory\10-projects\lexy\_workspace\`
- `REF-agent-zero.md` — Monologue Loop, @extensible, DirtyJson, LiteLLM
- `REF-openclaw.md` — Channel-Abstraktion, SKILL.md, Dreaming, Security
- `REF-jarvis.md` — Floating Assistant, Function Calling 3-Layer
- `REF-holomat.md` — CSS Holographic UI, Boot Sequence, Voice Rings
- `REF-smart-mirror.md` — Widget Registry, Responsive Fonts, Sleep/Wake

**Quellcode der Referenzprogramme:**
`G:\AI\LexyAI\referenz\`

## Lokale Services

| Service | Host:Port | GPU | Beschreibung |
|---------|-----------|-----|-------------|
| **A4B Brain** | `127.0.0.1:5005` | Tesla P40 | Gemma4 27B — deep reasoning |
| **E4B Brain** | `127.0.0.1:5006` | RTX 3060 Ti | Gemma4 12B — schnelle Antworten |
| **Gemma4 STT** | `127.0.0.1:5007` | RTX 3060 Ti | Gemma4 4B Multimodal — Audio→Text |
| **CosyVoice** | `172.20.0.245:5500` | Ubuntu | TTS, Voice "referenz_mio" |
| **SearXNG** | `127.0.0.1:9001` | — | Lokale Metasuchmaschine |
| **ChromaDB** | `127.0.0.1:8000` | — | Vektor-DB (4 Collections) |
| **Lexy Backend** | `0.0.0.0:8765` | — | FastAPI + WebSocket |
| **Lexy Frontend** | `0.0.0.0:7900` | — | Next.js (HTTPS) |
| **Embedding** | In-Process | cuda:0 | Jina v5 Small (1024 dim) |

**llama.cpp**: Wird als Binary genutzt (`llama-server.exe`). Liegt in `G:\AI\llamaExe\`.
GGUF-Modelle liegen in `G:\AI\llamaExe\model\`.

## Fundamentale Regeln

1. **KEIN Stub-Code** — Jede Funktion hat eine vollständige Implementierung
2. **KEIN `print()`** — Nur `structlog`. Immer.
3. **KEIN `run_in_executor`** — Nutze `aiosqlite` oder native async
4. **KEIN `dict.get()` für Config** — Nutze Pydantic Models
5. **KEIN `api._core`** — Plugins nutzen NUR die PluginAPI
6. **Contracts sind Wahrheit** — Siehe Obsidian `contracts.md`
7. **Type Hints überall** — `mypy --strict` muss bestehen
8. **Tests** — pytest, jede neue Datei bekommt einen Test
9. **Docstrings** auf Englisch, Kommentare Deutsch OK

## Projekt-Setup

```bash
# Conda Environment erstellen
conda create -n lexy python=3.11 -y
conda activate lexy

# Dependencies installieren
pip install -r requirements.txt

# ChromaDB starten (separates Terminal)
chroma run --host 127.0.0.1 --port 8000

# Frontend Setup
cd frontend
npm install
npm run dev
```

## Ersteinrichtung

```batch
:: 1. Junctions erstellen (einmalig, als Admin)
scripts\setup_junction.bat

:: Danach existieren:
:: docs\      → Obsidian Knowledge Base (lesen + schreiben!)
:: lexy_v1\   → Lexy OS v1 Source (Portierungs-Referenz)
:: referenz\  → Referenzprogramme (read-only)
```

## Projektstruktur

```
G:\AI\LexyAI\
├── CLAUDE.md                    ← DU BIST HIER
├── requirements.txt
├── environment.yml
├── docs\                        ← JUNCTION → Obsidian Knowledge Base
│   ├── _CLAUDE.md               # Master Index
│   ├── architecture/            # Architektur-Docs (lesen + updaten!)
│   ├── contracts.md             # SINGLE SOURCE OF TRUTH (updaten!)
│   ├── connections.md           # Aufrufketten (updaten!)
│   ├── patterns/                # Coding Style
│   ├── modules/                 # Modul-Docs (erstellen bei neuem Modul!)
│   └── changelog.md             # Changelog (updaten!)
├── lexy_v1\                     ← JUNCTION → Lexy OS v1 Source
├── referenz\                    # Referenzprogramme
├── config/
│   ├── config.yaml              # Hauptkonfiguration
│   ├── plugins.yaml             # Plugin-Konfiguration
│   └── routing.yaml             # LLM-Routing
├── lexy_core/                   # Backend
│   ├── app.py                   # LexyApp Hauptklasse
│   ├── config/                  # Pydantic Config
│   ├── events/                  # EventBus, Hooks, Signals
│   ├── agent/                   # Agent Loop + Router
│   ├── memory/                  # ChromaDB + HybridSearch
│   ├── plugin_system/           # PluginAPI + Loader
│   ├── tools/                   # Tool Registry + Caller
│   ├── voice/                   # VoiceManager + ABCs
│   ├── channels/                # ChannelBase + Router
│   ├── llm/                     # LiteLLM + DirtyJson
│   └── websocket/               # FastAPI WS
├── plugins/                     # Alle Plugins
├── frontend/                    # Next.js Frontend
├── data/                        # Runtime-Daten
├── tests/                       # pytest Tests
├── scripts/                     # Setup/Start Scripts
└── referenz/                    # Referenzprogramme (read-only)
```

## Implementierungs-Reihenfolge

**Phase 0 — Foundation** (ZUERST!):
1. Conda env + requirements.txt
2. Config System (Pydantic)
3. structlog Logging
4. EventBus + HookManager + Signals (v1 portieren)
5. PluginAPI + PluginLoader (v1 portieren)
6. FastAPI Gateway + WebSocket

**Phase 1 — Core Brain**:
1. LLM Client (LiteLLM)
2. RepetitionDetector (v1 portieren)
3. Agent Loop (Think→Plan→Execute→Reflect)
4. Tool-System (v1 portieren)
5. DirtyJson Parser

**Phase 2 — Memory + Voice**:
1. ChromaDB 4 Collections
2. HybridSearch 70/30 (v1 portieren)
3. VoiceManager + ABCs (v1 portieren)
4. voice_gemma4 STT Plugin (NEU)
5. voice_cosyvoice TTS Plugin (v1 portieren)
6. voice_canary Fallback (v1 portieren)

**Phase 3 — Channels**:
1. ChannelBase + SessionRouter (v1 portieren)
2. channel_whatsapp (Baileys Bridge) ⭐
3. channel_discord (discord.py)
4. channel_telegram (v1 portieren)
