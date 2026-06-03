"""
Phase 12.B + 12.C — memory_capture plugin tests.

Two layers under test:

1. **trigger_patterns** — pure-regex extraction. Tested as a plain
   function with a wide span of phrasings (German + English, fact-
   AFTER-trigger, fact-BEFORE-trigger, plus the failure modes:
   trigger-only ("merke dir das"), too-short fact, and noisy
   demonstratives that some sloppy patterns would otherwise capture).

2. **MemoryCapturePlugin** — the actual hook + storage path. We
   build a fake PluginAPI so we can intercept ``memory_store`` calls
   and assert they fire (or don't) for the right inputs. The
   implicit-capture layer uses a stubbed LLM that returns whatever
   JSON the test feeds — no real LLM call.

The tests deliberately don't boot a full LexyApp — the plugin's
seams are clean enough that we can drive it via stand-ins.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.memory_capture.plugin import MemoryCapturePlugin
from plugins.memory_capture.trigger_patterns import extract_fact


# ─── Layer 2: trigger_patterns ─────────────────────────────────────


class TestExtractFactGerman:
    """German fact-after-trigger and fact-before-trigger forms."""

    def test_merke_dir_comma_form(self) -> None:
        assert extract_fact(
            "merke dir, ich bin allergisch gegen Erdnüsse"
        ) == ("de", "ich bin allergisch gegen Erdnüsse")

    def test_merke_dir_dass_form(self) -> None:
        assert extract_fact(
            "merke dir dass mein Geburtstag der 1. Mai ist"
        ) == ("de", "mein Geburtstag der 1. Mai ist")

    def test_bitte_merken_colon_form(self) -> None:
        assert extract_fact(
            "Bitte merken: Mein Auto ist ein blauer Volvo"
        ) == ("de", "Mein Auto ist ein blauer Volvo")

    def test_vergiss_nicht_form(self) -> None:
        assert extract_fact(
            "vergiss nicht, dass ich Diabetes habe"
        ) == ("de", "ich Diabetes habe")

    def test_fact_before_trigger_with_address(self) -> None:
        """Mike's actual phrasing — the trigger sits at the END of
        the sentence after the fact, plus the message starts with
        an address ('Lexy ...') we strip."""
        assert extract_fact(
            "Lexy ich wohne am Nordpol, merke dir das"
        ) == ("de", "ich wohne am Nordpol")

    def test_fact_before_trigger_without_address(self) -> None:
        assert extract_fact(
            "ich heiße Mike, merke dir das"
        ) == ("de", "ich heiße Mike")


class TestExtractFactEnglish:
    def test_remember_that_form(self) -> None:
        assert extract_fact("remember that I am vegan") == (
            "en", "I am vegan",
        )

    def test_keep_in_mind_form(self) -> None:
        assert extract_fact(
            "keep in mind I prefer dark roast coffee"
        ) == ("en", "I prefer dark roast coffee")

    def test_dont_forget_form(self) -> None:
        assert extract_fact(
            "don't forget I'm allergic to peanuts"
        ) == ("en", "I'm allergic to peanuts")

    def test_fact_before_trigger_form(self) -> None:
        assert extract_fact(
            "I live at the north pole, remember this"
        ) == ("en", "I live at the north pole")


class TestExtractFactRejections:
    """Cases that must NOT match — too noisy, too short, no real fact."""

    @pytest.mark.parametrize(
        "msg",
        [
            "",
            "    ",
            "Lexy",
            "hi how are you",
            "what time is it?",
            "Lexy merke dir das",   # trigger only, no fact
            "merke dir das",         # trigger only
            "remember this",
            "vergiss nicht",          # trigger only, no comma + fact
        ],
    )
    def test_rejects_noise(self, msg: str) -> None:
        assert extract_fact(msg) is None

    def test_too_short_fact_rejected(self) -> None:
        # Single-word demonstrative fact gets filtered by the
        # multi-word guard. "das" is 3 chars — also fails the
        # length floor.
        assert extract_fact("merke dir das") is None
        assert extract_fact("remember this") is None


class TestExtractFactLanguageGate:
    """Disabling a language must drop matches in that language."""

    def test_disabled_german(self) -> None:
        assert extract_fact(
            "merke dir, ich wohne hier",
            enabled_languages=("en",),
        ) is None

    def test_disabled_english(self) -> None:
        assert extract_fact(
            "remember that I live here",
            enabled_languages=("de",),
        ) is None

    def test_empty_languages_disables_everything(self) -> None:
        assert extract_fact(
            "merke dir, ich wohne hier",
            enabled_languages=(),
        ) is None


# ─── Layer 2 + 3: MemoryCapturePlugin ──────────────────────────────


class _FakeAPI:
    """Minimal PluginAPI stub for the plugin tests.

    Tracks every memory_store call + every ws_broadcast so the test
    can assert the plugin made the calls it should and nothing it
    shouldn't.
    """

    def __init__(
        self,
        *,
        llm_response: str | None = None,
        relation_response: str = "UNABHAENGIG",
    ) -> None:
        self._cfg: dict[str, Any] = {
            "enabled_languages": ["de", "en"],
            "cooldown_seconds": 60,
            "implicit_capture_enabled": False,
            "implicit_confidence_min": 0.7,
            "implicit_brain": "e4b",
            "implicit_classify_cooldown": 0,  # disable for tests
        }
        self.stored: list[dict[str, Any]] = []
        self.broadcasts: list[dict[str, Any]] = []
        self.archived: list[dict[str, Any]] = []
        self.recall_responses: list[list[dict[str, Any]]] = []  # FIFO
        # Response the P2 relation classifier (api.llm_chat) returns:
        # DUPLIKAT / WIDERSPRUCH / UNABHAENGIG.
        self.relation_response = relation_response
        self.llm_chat_calls = 0

        # Stub LLM so the implicit-classifier path can be exercised
        # without a real network call.
        self._llm = MagicMock()
        self._llm.chat = AsyncMock(return_value=llm_response or "")
        self._app = MagicMock()
        self._app.llm = self._llm

    async def llm_chat(self, **kwargs: Any) -> str:
        self.llm_chat_calls += 1
        return self.relation_response

    async def memory_archive(
        self,
        collection: str,
        ids: list[str],
        reason: str,
        extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        self.archived.append(
            {
                "collection": collection,
                "ids": list(ids),
                "reason": reason,
                "extra_meta": dict(extra_meta or {}),
            }
        )
        return {"archived": len(ids), "fts": len(ids)}

    def get_config(self) -> dict[str, Any]:
        return dict(self._cfg)

    def update_config(self, **patch: Any) -> None:
        self._cfg.update(patch)

    async def memory_store(
        self,
        text: str,
        collection: str = "facts",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.stored.append({
            "text": text,
            "collection": collection,
            "metadata": dict(metadata or {}),
        })
        return f"stored-{len(self.stored)}"

    async def memory_recall(self, **kwargs: Any) -> list[dict[str, Any]]:
        if self.recall_responses:
            return self.recall_responses.pop(0)
        return []

    async def ws_broadcast(self, payload: dict[str, Any]) -> None:
        self.broadcasts.append(dict(payload))

    def register_hook(self, *args: Any, **kwargs: Any) -> None:
        # Tests call the handler directly; we don't need real hook
        # wiring.
        pass


def _build_plugin(api: _FakeAPI | None = None) -> MemoryCapturePlugin:
    api = api or _FakeAPI()
    plugin = MemoryCapturePlugin.__new__(MemoryCapturePlugin)
    plugin.api = api
    return plugin


# ── Layer 2 hook tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_explicit_hook_stores_on_trigger() -> None:
    api = _FakeAPI()
    plugin = _build_plugin(api)
    await plugin.on_load()

    ctx = {
        "text": "merke dir, ich wohne am Nordpol",
        "session_id": "sess-1",
    }
    await plugin._capture_explicit_facts(ctx)

    assert len(api.stored) == 1
    stored = api.stored[0]
    assert stored["collection"] == "facts"
    assert "Nordpol" in stored["text"]
    meta = stored["metadata"]
    assert meta["source"] == "trigger_phrase"
    assert meta["language"] == "de"
    assert meta["session_id"] == "sess-1"
    # Captured-flag set so Layer 3 doesn't reclassify the same msg.
    assert ctx.get("_memory_capture_explicit")
    # Broadcast happened.
    assert any(
        b["type"] == "memory_captured" and b["kind"] == "trigger_phrase"
        for b in api.broadcasts
    )


@pytest.mark.asyncio
async def test_explicit_hook_skips_non_trigger_messages() -> None:
    api = _FakeAPI()
    plugin = _build_plugin(api)
    await plugin.on_load()

    await plugin._capture_explicit_facts(
        {"text": "wie spät ist es?", "session_id": "sess-1"}
    )
    assert api.stored == []
    assert api.broadcasts == []


@pytest.mark.asyncio
async def test_explicit_hook_cooldown_prevents_double_store() -> None:
    api = _FakeAPI()
    plugin = _build_plugin(api)
    await plugin.on_load()

    msg = {"text": "merke dir, ich heiße Mike", "session_id": "sess-1"}
    await plugin._capture_explicit_facts(dict(msg))
    await plugin._capture_explicit_facts(dict(msg))  # duplicate, within cooldown

    assert len(api.stored) == 1


@pytest.mark.asyncio
async def test_explicit_hook_disabled_language() -> None:
    api = _FakeAPI()
    api.update_config(enabled_languages=["en"])  # German off
    plugin = _build_plugin(api)
    await plugin.on_load()

    await plugin._capture_explicit_facts(
        {"text": "merke dir, ich heiße Mike", "session_id": "sess-1"}
    )
    assert api.stored == []


# ── Layer 3 hook tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_implicit_hook_stores_when_classifier_says_yes() -> None:
    classifier_json = json.dumps({
        "is_fact": True,
        "fact": "User wohnt in Berlin",
        "confidence": 0.9,
    })
    api = _FakeAPI(llm_response=classifier_json)
    api.update_config(implicit_capture_enabled=True)
    plugin = _build_plugin(api)
    await plugin.on_load()

    ctx = {
        "text": "Ich bin gerade nach Berlin gezogen, fühlt sich gut an.",
        "session_id": "sess-2",
    }
    await plugin._capture_implicit_facts(ctx)

    assert len(api.stored) == 1
    stored = api.stored[0]
    assert "Berlin" in stored["text"]
    assert stored["metadata"]["source"] == "implicit_capture"
    assert abs(stored["metadata"]["confidence"] - 0.9) < 1e-6


@pytest.mark.asyncio
async def test_implicit_hook_skips_when_classifier_says_no() -> None:
    classifier_json = json.dumps({
        "is_fact": False,
        "fact": "",
        "confidence": 0.0,
    })
    api = _FakeAPI(llm_response=classifier_json)
    api.update_config(implicit_capture_enabled=True)
    plugin = _build_plugin(api)
    await plugin.on_load()

    await plugin._capture_implicit_facts(
        {"text": "Ich bin gerade so müde", "session_id": "sess-2"}
    )
    assert api.stored == []


@pytest.mark.asyncio
async def test_implicit_hook_skips_below_confidence_threshold() -> None:
    classifier_json = json.dumps({
        "is_fact": True,
        "fact": "User vielleicht etwas",
        "confidence": 0.4,  # below default threshold of 0.7
    })
    api = _FakeAPI(llm_response=classifier_json)
    api.update_config(implicit_capture_enabled=True)
    plugin = _build_plugin(api)
    await plugin.on_load()

    await plugin._capture_implicit_facts(
        {"text": "ich glaube ich tendiere zu Veganismus", "session_id": "s"}
    )
    assert api.stored == []


@pytest.mark.asyncio
async def test_implicit_hook_dedups_via_recall() -> None:
    """If the recall finds a near-identical fact (score ≥ 0.85) and the
    P2 relation classifier confirms it's a DUPLIKAT, we skip the store —
    prevents spamming the same fact 5x."""
    classifier_json = json.dumps({
        "is_fact": True,
        "fact": "User wohnt am Nordpol",
        "confidence": 0.95,
    })
    api = _FakeAPI(llm_response=classifier_json, relation_response="DUPLIKAT")
    api.update_config(implicit_capture_enabled=True)
    api.recall_responses.append([
        {"id": "old-np", "content": "User wohnt am Nordpol seit Jahren", "score": 0.91}
    ])
    plugin = _build_plugin(api)
    await plugin.on_load()

    await plugin._capture_implicit_facts(
        {"text": "Ich wohne übrigens am Nordpol", "session_id": "s"}
    )
    assert api.stored == []
    assert api.archived == []  # pure duplicate → nothing to supersede


@pytest.mark.asyncio
async def test_implicit_hook_skips_when_explicit_already_fired() -> None:
    """Layer 3 must not double-store after Layer 2 already grabbed
    the fact — the ``_memory_capture_explicit`` ctx flag signals it."""
    api = _FakeAPI()
    api.update_config(implicit_capture_enabled=True)
    plugin = _build_plugin(api)
    await plugin.on_load()

    ctx = {
        "text": "merke dir, ich wohne hier",
        "session_id": "s",
        "_memory_capture_explicit": "ich wohne hier",
    }
    await plugin._capture_implicit_facts(ctx)
    assert api.stored == []
    # And the LLM was never even called.
    assert api._llm.chat.await_count == 0


@pytest.mark.asyncio
async def test_implicit_hook_handles_garbage_classifier_output() -> None:
    """When the LLM returns nonsense, we log + skip, not crash."""
    api = _FakeAPI(llm_response="not valid json at all")
    api.update_config(implicit_capture_enabled=True)
    plugin = _build_plugin(api)
    await plugin.on_load()

    await plugin._capture_implicit_facts(
        {"text": "Ich liebe Jazz seit Jahren", "session_id": "s"}
    )
    assert api.stored == []


@pytest.mark.asyncio
async def test_implicit_hook_strips_markdown_code_fence() -> None:
    """LLMs sometimes wrap JSON in ```json``` fences. We strip them."""
    classifier_json = (
        "```json\n"
        + json.dumps({
            "is_fact": True,
            "fact": "User mag Jazz",
            "confidence": 0.85,
        })
        + "\n```"
    )
    api = _FakeAPI(llm_response=classifier_json)
    api.update_config(implicit_capture_enabled=True)
    plugin = _build_plugin(api)
    await plugin.on_load()

    await plugin._capture_implicit_facts(
        {"text": "Ich liebe Jazz seit Jahren", "session_id": "s"}
    )
    assert len(api.stored) == 1
    assert "Jazz" in api.stored[0]["text"]


# ── P2: contradiction resolution at the write choke-point ──────────


@pytest.mark.asyncio
async def test_contradiction_supersedes_old_fact() -> None:
    """A new fact that updates a close existing one stores the new entry
    (tagged supersedes) and recoverably archives the old one."""
    api = _FakeAPI(relation_response="WIDERSPRUCH")
    api.recall_responses.append(
        [{"id": "old-1", "content": "User nutzt Python 3.11", "score": 0.93}]
    )
    plugin = _build_plugin(api)
    await plugin.on_load()

    stored = await plugin._store_fact_with_dedup(
        fact="User nutzt Python 3.12",
        session_id="s1",
        source="trigger_phrase",
        metadata={"language": "de"},
    )

    assert stored is True
    assert len(api.stored) == 1
    assert api.stored[0]["metadata"].get("supersedes") == "old-1"
    assert len(api.archived) == 1
    arc = api.archived[0]
    assert arc["collection"] == "facts"
    assert arc["ids"] == ["old-1"]
    assert arc["reason"] == "superseded"
    assert arc["extra_meta"]["superseded_by"] == "stored-1"


@pytest.mark.asyncio
async def test_duplicate_close_fact_is_skipped() -> None:
    api = _FakeAPI(relation_response="DUPLIKAT")
    api.recall_responses.append(
        [{"id": "old-1", "content": "User wohnt in Berlin", "score": 0.97}]
    )
    plugin = _build_plugin(api)
    await plugin.on_load()

    stored = await plugin._store_fact_with_dedup(
        fact="User lebt in Berlin",
        session_id="s1",
        source="implicit_capture",
        metadata={},
    )

    assert stored is False
    assert api.stored == []
    assert api.archived == []


@pytest.mark.asyncio
async def test_unrelated_close_neighbor_still_stores() -> None:
    api = _FakeAPI(relation_response="UNABHAENGIG")
    api.recall_responses.append(
        [{"id": "old-1", "content": "User mag Jazz", "score": 0.88}]
    )
    plugin = _build_plugin(api)
    await plugin.on_load()

    stored = await plugin._store_fact_with_dedup(
        fact="User hat einen Hund namens Rex",
        session_id="s1",
        source="implicit_capture",
        metadata={},
    )

    assert stored is True
    assert len(api.stored) == 1
    assert "supersedes" not in api.stored[0]["metadata"]
    assert api.archived == []


@pytest.mark.asyncio
async def test_below_band_skips_relation_classifier() -> None:
    """A weak neighbour (< band) must NOT trigger the LLM relation call —
    the fact is just stored as new."""
    api = _FakeAPI(relation_response="WIDERSPRUCH")  # would supersede IF called
    api.recall_responses.append(
        [{"id": "old-1", "content": "etwas anderes", "score": 0.40}]
    )
    plugin = _build_plugin(api)
    await plugin.on_load()

    stored = await plugin._store_fact_with_dedup(
        fact="User heißt Mike",
        session_id="s1",
        source="trigger_phrase",
        metadata={},
    )

    assert stored is True
    assert len(api.stored) == 1
    assert api.llm_chat_calls == 0  # band gate prevented the LLM call
    assert api.archived == []


@pytest.mark.asyncio
async def test_contradiction_disabled_falls_back_to_skip() -> None:
    """With contradiction resolution off, a close neighbour means plain
    skip (pre-P2 dedup behaviour) — no LLM call, no supersede."""
    api = _FakeAPI(relation_response="WIDERSPRUCH")
    api._cfg["contradiction_resolution_enabled"] = False
    api.recall_responses.append(
        [{"id": "old-1", "content": "User nutzt Python 3.11", "score": 0.93}]
    )
    plugin = _build_plugin(api)
    await plugin.on_load()

    stored = await plugin._store_fact_with_dedup(
        fact="User nutzt Python 3.12",
        session_id="s1",
        source="implicit_capture",
        metadata={},
    )

    assert stored is False
    assert api.stored == []
    assert api.archived == []
    assert api.llm_chat_calls == 0
