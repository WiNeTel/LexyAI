"""Tests for Silly-Tavern PNG card import (Phase 9.9).

Mike's question: *"beim charakter import, da sollte doch beides
funktionieren, json und Silly Tavern Cards als png, oder?"*. Yes — but
only after this fix; the previous import path only accepted JSON.

Three layers under test:

1. **parse_silly_tavern_png** — extracts the embedded ``chara``
   tEXt chunk, base64-decodes it, parses the JSON, returns the
   resulting CharacterCard plus the raw PNG bytes (for use as
   avatar).
2. **parse_silly_tavern_bytes** — auto-detects PNG vs JSON via
   magic bytes / extension / MIME and dispatches.
3. **store.import_silly_tavern_bytes** — persists the card AND the
   avatar PNG to disk under the avatar directory.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio


# ─── Helpers ────────────────────────────────────────────────────────


def _make_card_png(card_data: dict, *, color: str = "red", chunk: str = "chara") -> bytes:
    """Build a tiny PNG with a Silly-Tavern ``chara`` tEXt chunk."""
    from PIL import Image, PngImagePlugin
    img = Image.new("RGB", (16, 16), color=color)
    buf = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    payload = json.dumps(card_data, ensure_ascii=False)
    encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    info.add_text(chunk, encoded)
    img.save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def _make_plain_png() -> bytes:
    from PIL import Image
    img = Image.new("RGB", (8, 8), color="blue")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ─── 1. parse_silly_tavern_png ──────────────────────────────────────


class TestParsePNG:
    def test_v2_card_is_parsed(self) -> None:
        from plugins.character_chat.character_card import parse_silly_tavern_png
        data = _make_card_png({
            "spec": "chara_card_v2",
            "data": {
                "name": "Vyrkos",
                "description": "Alter Drache.",
                "first_mes": "Hallo.",
                "scenario": "Höhle.",
                "mes_example": "Mensch:\nDrache: *grollt*",
                "tags": "drache, alt",
            },
        })
        card, raw = parse_silly_tavern_png(data)
        assert card.name == "Vyrkos"
        assert card.persona == "Alter Drache."
        assert card.greeting == "Hallo."
        assert card.scenario == "Höhle."
        assert "drache" in card.tags
        assert raw == data  # bytes returned for avatar persistence

    def test_v1_flat_card_is_parsed(self) -> None:
        from plugins.character_chat.character_card import parse_silly_tavern_png
        data = _make_card_png({
            "name": "Mara",
            "description": "Captain.",
            "first_mes": "Aye.",
        })
        card, _ = parse_silly_tavern_png(data)
        assert card.name == "Mara"
        assert card.persona == "Captain."

    def test_ccv3_chunk_also_accepted(self) -> None:
        # Some forks write to ``ccv3`` instead of ``chara``.
        from plugins.character_chat.character_card import parse_silly_tavern_png
        data = _make_card_png(
            {"name": "X", "description": "Y"},
            chunk="ccv3",
        )
        card, _ = parse_silly_tavern_png(data)
        assert card.name == "X"

    def test_empty_bytes_rejected(self) -> None:
        from plugins.character_chat.character_card import (
            parse_silly_tavern_png, CharacterCardError,
        )
        with pytest.raises(CharacterCardError, match="empty"):
            parse_silly_tavern_png(b"")

    def test_non_png_rejected(self) -> None:
        from plugins.character_chat.character_card import (
            parse_silly_tavern_png, CharacterCardError,
        )
        with pytest.raises(CharacterCardError, match="not a valid PNG"):
            parse_silly_tavern_png(b"this is not a PNG at all")

    def test_png_without_chara_chunk_rejected(self) -> None:
        from plugins.character_chat.character_card import (
            parse_silly_tavern_png, CharacterCardError,
        )
        plain = _make_plain_png()
        with pytest.raises(CharacterCardError, match="no Silly-Tavern"):
            parse_silly_tavern_png(plain)

    def test_invalid_base64_rejected(self) -> None:
        # Hand-craft a PNG whose ``chara`` chunk holds non-base64 bytes.
        from PIL import Image, PngImagePlugin
        img = Image.new("RGB", (8, 8), color="red")
        info = PngImagePlugin.PngInfo()
        info.add_text("chara", "###not base64###")
        buf = io.BytesIO()
        img.save(buf, "PNG", pnginfo=info)
        from plugins.character_chat.character_card import (
            parse_silly_tavern_png, CharacterCardError,
        )
        # base64.b64decode is permissive (validate=False) — bad chars
        # may be silently skipped, but the resulting bytes won't be
        # valid utf-8 / JSON.
        with pytest.raises(CharacterCardError):
            parse_silly_tavern_png(buf.getvalue())

    def test_invalid_json_rejected(self) -> None:
        from PIL import Image, PngImagePlugin
        img = Image.new("RGB", (8, 8), color="red")
        info = PngImagePlugin.PngInfo()
        # Valid base64, but the decoded text isn't JSON.
        bad = base64.b64encode(b"this is NOT json").decode("ascii")
        info.add_text("chara", bad)
        buf = io.BytesIO()
        img.save(buf, "PNG", pnginfo=info)
        from plugins.character_chat.character_card import (
            parse_silly_tavern_png, CharacterCardError,
        )
        with pytest.raises(CharacterCardError, match="not valid JSON"):
            parse_silly_tavern_png(buf.getvalue())

    def test_missing_name_rejected(self) -> None:
        # JSON parses but name is missing → ``parse_silly_tavern_card``
        # rejects (the card model requires a non-empty name).
        from plugins.character_chat.character_card import (
            parse_silly_tavern_png, CharacterCardError,
        )
        data = _make_card_png({"description": "no name here"})
        with pytest.raises(CharacterCardError, match="missing 'name'"):
            parse_silly_tavern_png(data)


# ─── 2. parse_silly_tavern_bytes auto-detect ────────────────────────


class TestAutoDetect:
    def test_detects_png_by_magic_bytes(self) -> None:
        from plugins.character_chat.character_card import parse_silly_tavern_bytes
        data = _make_card_png({"name": "Mara", "description": "x"})
        card, png = parse_silly_tavern_bytes(data)
        assert card.name == "Mara"
        assert png == data

    def test_detects_png_by_filename(self) -> None:
        from plugins.character_chat.character_card import parse_silly_tavern_bytes
        # PNG with the right magic bytes still wins — but extension is
        # what we'd use if magic bytes were ambiguous (they aren't here).
        data = _make_card_png({"name": "X", "description": "y"})
        card, png = parse_silly_tavern_bytes(
            data, filename="weird.PNG", content_type="application/octet-stream",
        )
        assert card.name == "X"
        assert png is not None

    def test_detects_png_by_mime(self) -> None:
        from plugins.character_chat.character_card import parse_silly_tavern_bytes
        data = _make_card_png({"name": "X", "description": "y"})
        card, png = parse_silly_tavern_bytes(data, content_type="image/png")
        assert card.name == "X"
        assert png is not None

    def test_falls_back_to_json(self) -> None:
        from plugins.character_chat.character_card import parse_silly_tavern_bytes
        payload = json.dumps({"name": "Json-Char", "description": "xyz"})
        card, png = parse_silly_tavern_bytes(
            payload.encode("utf-8"), filename="card.json",
        )
        assert card.name == "Json-Char"
        assert png is None

    def test_v2_json_also_works(self) -> None:
        from plugins.character_chat.character_card import parse_silly_tavern_bytes
        payload = json.dumps({
            "spec": "chara_card_v2",
            "data": {"name": "v2-char", "description": "abc"},
        })
        card, png = parse_silly_tavern_bytes(payload.encode("utf-8"))
        assert card.name == "v2-char"
        assert png is None

    def test_empty_payload_rejected(self) -> None:
        from plugins.character_chat.character_card import (
            parse_silly_tavern_bytes, CharacterCardError,
        )
        with pytest.raises(CharacterCardError, match="empty"):
            parse_silly_tavern_bytes(b"")

    def test_garbage_rejected(self) -> None:
        from plugins.character_chat.character_card import (
            parse_silly_tavern_bytes, CharacterCardError,
        )
        with pytest.raises(CharacterCardError):
            parse_silly_tavern_bytes(b"\x00\x01\x02\x03not png nor json")


# ─── 3. Store integration: PNG import writes the avatar ─────────────


@pytest_asyncio.fixture
async def store_with_db():
    from plugins.character_chat.character_store import CharacterStore
    db = await aiosqlite.connect(":memory:")
    s = CharacterStore(db)
    await s.init_schema()
    yield s
    await db.close()


class TestStoreImportBytes:
    @pytest.mark.asyncio
    async def test_png_import_persists_avatar_to_disk(
        self, store_with_db, tmp_path: Path
    ) -> None:
        avatar_dir = tmp_path / "avatars"
        png = _make_card_png({
            "spec": "chara_card_v2",
            "data": {"name": "Vyrkos", "description": "Drache."},
        })
        saved = await store_with_db.import_silly_tavern_bytes(
            png,
            filename="vyrkos.png",
            content_type="image/png",
            color="#aa3333",
            age_stage="adult",
            avatar_dir=avatar_dir,
        )
        assert saved.name == "Vyrkos"
        assert saved.color == "#aa3333"
        # Avatar file written + URL points at /avatars/<id>.png.
        avatar_path = avatar_dir / f"{saved.id}.png"
        assert avatar_path.exists()
        assert avatar_path.stat().st_size == len(png)
        assert saved.avatar == f"/avatars/{saved.id}.png"

    @pytest.mark.asyncio
    async def test_json_import_does_not_touch_avatar_dir(
        self, store_with_db, tmp_path: Path
    ) -> None:
        avatar_dir = tmp_path / "avatars"
        payload = json.dumps({"name": "JsonChar", "description": "x"})
        saved = await store_with_db.import_silly_tavern_bytes(
            payload.encode("utf-8"),
            filename="card.json",
            content_type="application/json",
            avatar_dir=avatar_dir,
        )
        assert saved.name == "JsonChar"
        assert saved.avatar == ""  # no avatar from JSON
        # Directory may exist (mkdir is idempotent) but no PNG inside.
        if avatar_dir.exists():
            files = list(avatar_dir.glob("*.png"))
            assert files == []

    @pytest.mark.asyncio
    async def test_png_without_avatar_dir_still_persists_card(
        self, store_with_db
    ) -> None:
        # Mike might call the import path without giving us an
        # avatar_dir (CLI script, custom integration, etc.). The card
        # must still land — only the avatar persistence is skipped.
        png = _make_card_png({"name": "Skipped", "description": "x"})
        saved = await store_with_db.import_silly_tavern_bytes(
            png,
            filename="x.png",
            content_type="image/png",
            avatar_dir=None,
        )
        assert saved.name == "Skipped"
        assert saved.avatar == ""

    @pytest.mark.asyncio
    async def test_age_stage_carried_through(
        self, store_with_db, tmp_path: Path
    ) -> None:
        avatar_dir = tmp_path / "avatars"
        png = _make_card_png({"name": "Baby", "description": "kleines Wesen"})
        saved = await store_with_db.import_silly_tavern_bytes(
            png, age_stage="baby", avatar_dir=avatar_dir,
        )
        assert saved.age_stage == "baby"
