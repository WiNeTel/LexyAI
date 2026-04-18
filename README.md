# Lexy AI

> Local-first AI assistant with voice, persistent memory, plugin system,
> and external messaging channels. Everything runs on the local GPU via
> llama.cpp — no cloud APIs required.

**Status:** Private personal project — public nowhere by design. The
repository is used for version control and bootstrapping new machines,
not as a distribution channel. Expect rough edges, frequent refactors,
and assume code is under active development for the maintainer's own
use.

---

## Features

- **Two-brain routing** — fast Gemma 4 12B (E4B) for chat, deep
  Gemma 4 27B (A4B) for reasoning + RP. Router picks per turn.
- **Persistent memory** — ChromaDB with 4 collections + 70/30 hybrid
  search (semantic + BM25). Each character keeps isolated memory.
- **Plugin system** — hot-reloadable Python plugins with a clean
  `PluginAPI` facade (no reaching into core internals).
- **Voice pipeline** — Gemma 4 multimodal for STT, CosyVoice for TTS
  with per-character voices and narrator-emotion pipeline. Canary as
  fallback.
- **External channels** — WhatsApp (via Baileys bridge), Discord,
  Telegram. All converge into the same session/memory layer.
- **Character RP** — Silly-Tavern-lite with group turns, pulse timers
  (babies cry on their own), and an autonomous-simulation mode that
  self-plays scenes at a configurable interval.
- **Scheduler** — cron/natural-language timers that can invoke tools,
  send proactive messages, or trigger character pulses.

## Architecture

```
┌──────────── Frontend (vanilla HTML/JS/CSS via static server) ────────────┐
└───────────────────────────────┬──────────────────────────────────────────┘
                                │ WebSocket (0.0.0.0:8765)
┌───────────────────────────────▼──────────────────────────────────────────┐
│                          lexy_core (FastAPI)                             │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐  ┌─────────────┐   │
│  │  Agent  │  │ Plugins  │  │ Memory  │  │  Voice   │  │  Channels   │   │
│  │  Loop   │→ │ (hot-    │  │ Chroma  │  │ STT/TTS  │  │ WhatsApp    │   │
│  │ + Tools │  │ reload)  │  │  + HS   │  │ Provider │  │ Discord TG  │   │
│  └────┬────┘  └────┬─────┘  └────┬────┘  └────┬─────┘  └──────┬──────┘   │
│       │            │             │            │               │          │
│       └── EventBus + HookManager + Signals (thread-safe shared state) ───┤
└───────────┬─────────────┬───────────────────────────┬────────────────────┘
            │             │                           │
   ┌────────▼────┐   ┌────▼───────┐    ┌──────────────▼────────────┐
   │ A4B Brain   │   │ E4B Brain  │    │ CosyVoice TTS             │
   │ 127.0.0.1   │   │ 127.0.0.1  │    │ 172.20.0.245:5500 (Docker)│
   │ :5005 P40   │   │ :5006 3060 │    └───────────────────────────┘
   │ Gemma4 27B  │   │ Gemma4 12B │
   └─────────────┘   └────────────┘
```

Full architecture docs (contracts, connection graphs, module-level
references) live in a separate Obsidian vault and are **not** mirrored
into this repository.

## Quickstart

Prerequisites:

- Windows 10/11 (NVIDIA GPU, 24 GB+ VRAM recommended for A4B)
- Python 3.11 via Miniconda
- NVIDIA drivers + CUDA toolkit
- [llama.cpp](https://github.com/ggerganov/llama.cpp) binaries (not
  vendored here; place `llama-server.exe` in `G:\AI\llamaExe\` or
  adjust `scripts/start_llm_*.bat`)
- [ChromaDB](https://www.trychroma.com/) (`pip install chromadb`)
- Optional: Docker host for CosyVoice TTS

Setup:

```batch
:: 1. Conda env
conda env create -f environment.yml
conda activate lexy

:: 2. (Once) create Windows junctions for docs and v1 source
scripts\setup_junction.bat

:: 3. Start the local services (one per terminal)
scripts\start_chromadb.bat
scripts\start_llm_main.bat      :: A4B on port 5005
scripts\start_llm_fast.bat      :: E4B on port 5006

:: 4. Start Lexy
scripts\start_lexy.bat
:: or: python -m lexy_core

:: 5. Open the frontend
start https://localhost:7900
```

The WhatsApp bridge (optional) is in `bridges/` — see its own scripts.

## Local Services

| Service            | Host:Port         | GPU         | Purpose                            |
|--------------------|-------------------|-------------|------------------------------------|
| **A4B Brain**      | `127.0.0.1:5005`  | Tesla P40   | Gemma 4 27B — deep reasoning, RP   |
| **E4B Brain**      | `127.0.0.1:5006`  | RTX 3060 Ti | Gemma 4 12B — fast chat            |
| **Gemma 4 STT**    | `127.0.0.1:5007`  | RTX 3060 Ti | Gemma 4 4B Multimodal — audio→text |
| **CosyVoice TTS**  | `172.20.0.245:5500` | Ubuntu    | Per-character voice synthesis      |
| **SearXNG**        | `127.0.0.1:9001`  | —           | Local meta-search                  |
| **ChromaDB**       | `127.0.0.1:8000`  | —           | Vector DB, 4 collections           |
| **Lexy Backend**   | `0.0.0.0:8765`    | —           | FastAPI + WebSocket                |
| **Lexy Frontend**  | `0.0.0.0:7900`    | —           | Static HTTPS server                |
| **Embedding**      | in-process        | cuda:0      | Jina v5 Small (1024 dim)           |

## Project Layout

```
LexyAI/
├── lexy_core/              # Backend: FastAPI app, agent loop, memory,
│                           # plugin system, voice, channels, LLM client
├── plugins/                # Hot-reloadable plugins (see table below)
├── bridges/                # External-service bridges (WhatsApp Baileys)
├── frontend/static/        # Vanilla HTML/JS/CSS UI
├── config/                 # YAML config (routing, plugins, main)
├── scripts/                # Batch launchers, junction setup
├── tests/                  # pytest — 740+ tests, runs on every change
├── data/                   # Runtime data (gitignored: sessions, memory,
│                           # ChromaDB, character SQLite, avatars, certs)
├── llama.cpp/              # (gitignored) llama-server binaries + GGUF
├── docs/                   # (gitignored junction) Obsidian vault
└── lexy_v1/                # (gitignored junction) v1 source for porting
```

## Plugins

| Plugin                   | Purpose                                                    |
|--------------------------|------------------------------------------------------------|
| `character_chat`         | Silly-Tavern-lite: personas, group turns, pulses, sim mode |
| `autonomous_thinking`    | Background "thoughts" that surface in chat                 |
| `scheduler`              | Cron/NL timers, recurring tasks, proactive triggers        |
| `dreaming`               | Offline memory consolidation pass                          |
| `expert_panel`           | Multi-perspective deliberation for complex questions       |
| `knowledge_acquisition`  | Scrape → categorise → quality-score → persist              |
| `skill_writer`           | Auto-agent that can extend Lexy with new skills            |
| `orchestrator`           | Multi-brain orchestration patterns                         |
| `mcp_bridge`             | Model Context Protocol bridge                              |
| `dashboard`              | Ops dashboard (plugin status, memory health)               |
| `game_bridge`            | Minecraft / Factorio via RCON                              |
| `voice_canary`           | Canary STT (fallback)                                      |
| `voice_cosyvoice`        | CosyVoice TTS with narrator + emotion pipeline             |
| `voice_gemma4`           | Gemma 4 multimodal STT (primary)                           |
| `channel_whatsapp`       | WhatsApp (via Baileys bridge in `bridges/`)                |
| `channel_discord`        | Discord bot                                                |
| `channel_telegram`       | Telegram bot                                               |
| `weather`                | Open-Meteo (free, no API key)                              |
| `web_crawler`            | Crawl + knowledge ingestion                                |
| `youtube`                | YT transcript/metadata                                     |
| `spotify`                | Spotify control                                            |

## Development

```bash
conda activate lexy
pytest                          # full suite (~2 min)
pytest tests/test_character_chat_simulation.py -v   # targeted
mypy --strict lexy_core plugins # type-check (project rule)
```

Coding rules (enforced in reviews):

- **structlog** only — no `print()`.
- **Type hints everywhere**; `mypy --strict` must pass.
- **Pydantic** for config; no `dict.get()` on config dicts.
- **PluginAPI** only from plugins; never reach into `api._app` or
  similar private attrs.
- **aiosqlite** / native async — no `run_in_executor` for I/O.
- **Docstrings English**, comments may be German.

## Secrets & Config

- `config/config.yaml` ships with `api_key: "sk-lexy-local"` — that's
  just llama.cpp's local-auth placeholder, not a real credential.
- External service tokens (Discord, Telegram, etc.) are pulled from
  environment variables at runtime — the YAML only names the var:

  ```yaml
  channels:
    discord:
      token_env: "LEXY_DISCORD_TOKEN"
  ```

- Local overrides go in `config/config.local.yaml` (gitignored).

## License

[GPL v3](./LICENSE) — Copyleft. Forks and derivatives must also be
GPL v3.

Copyright © 2026 Mike (WiNeTel)
