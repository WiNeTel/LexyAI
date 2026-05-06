"""Tests for how the agent stitches uploads into the LLM message.

Two surfaces matter:

1. The brain router — when an attachment list contains an image, the
   route MUST land on ``multi`` (the multimodal brain on :5007).
2. The user-message construction — non-image attachments fold into the
   text body, images become ``image_url`` content blocks. We cover the
   two helper functions directly instead of building a full LexyApp.
"""

from __future__ import annotations

from lexy_core.agent.agent import (
    _build_image_blocks,
    _render_non_image_attachments,
)
from lexy_core.agent.router import BrainRouter
from lexy_core.config import RoutingConfig


# ─── Image blocks ────────────────────────────────────────────────────


class TestBuildImageBlocks:
    def test_empty_input(self) -> None:
        assert _build_image_blocks([]) == []

    def test_only_images_kept(self) -> None:
        blocks = _build_image_blocks(
            [
                {"kind": "image", "data_url": "data:image/png;base64,A"},
                {"kind": "document", "url": "/uploads/x/y.pdf"},
                {"kind": "code", "url": "/uploads/x/z.py"},
            ]
        )
        assert blocks == [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,A"}}
        ]

    def test_url_fallback_when_no_data_url(self) -> None:
        blocks = _build_image_blocks(
            [{"kind": "image", "url": "/uploads/s/abc.jpg"}]
        )
        assert blocks == [
            {"type": "image_url", "image_url": {"url": "/uploads/s/abc.jpg"}}
        ]

    def test_skips_image_without_url(self) -> None:
        # Defensive: the upload always returns at least a url, but if
        # something later strips both fields we shouldn't emit a broken
        # image_url block.
        blocks = _build_image_blocks([{"kind": "image"}])
        assert blocks == []

    def test_skips_non_dict_entries(self) -> None:
        blocks = _build_image_blocks(["nonsense", None, 42])  # type: ignore[list-item]
        assert blocks == []


# ─── Non-image attachment text ──────────────────────────────────────


class TestRenderNonImageAttachments:
    def test_images_skipped(self) -> None:
        # Pure image list → empty text block (images go via blocks).
        assert _render_non_image_attachments(
            [{"kind": "image", "data_url": "data:..."}]
        ) == ""

    def test_document_renders_with_excerpt_and_chunk_count(self) -> None:
        out = _render_non_image_attachments(
            [
                {
                    "kind": "document",
                    "filename": "spec.pdf",
                    "size": 2048,
                    "excerpt": "Section 1: Goals",
                    "chunks_indexed": 3,
                }
            ]
        )
        assert "Anhänge dieser Nachricht" in out
        assert "spec.pdf" in out
        assert "3 chunks indexed" in out
        assert "Section 1: Goals" in out

    def test_code_renders_with_language(self) -> None:
        out = _render_non_image_attachments(
            [
                {
                    "kind": "code",
                    "filename": "main.rs",
                    "size": 512,
                    "language": "rust",
                    "lines": 20,
                    "excerpt": "fn main() { println!(\"hi\"); }",
                }
            ]
        )
        assert "main.rs" in out
        assert "rust" in out
        assert "20 lines" in out
        assert "fn main()" in out

    def test_audio_renders_transcript_when_available(self) -> None:
        out = _render_non_image_attachments(
            [
                {
                    "kind": "audio",
                    "filename": "memo.mp3",
                    "size": 100_000,
                    "transcript": "Reminder to buy milk.",
                }
            ]
        )
        assert "Audio" in out
        assert "memo.mp3" in out
        assert "Reminder to buy milk." in out

    def test_audio_without_transcript_shows_placeholder(self) -> None:
        out = _render_non_image_attachments(
            [{"kind": "audio", "filename": "memo.wav", "size": 100}]
        )
        assert "no transcript" in out

    def test_unknown_kind_rendered_as_generic_attachment(self) -> None:
        out = _render_non_image_attachments(
            [{"kind": "blob", "filename": "thing.bin", "size": 42}]
        )
        assert "thing.bin" in out
        assert "blob" in out


# ─── Router vision override ─────────────────────────────────────────


class TestRouterVisionOverride:
    def _router(self, default: str = "a4b") -> BrainRouter:
        cfg = RoutingConfig(default_brain=default, rules=[])
        return BrainRouter(cfg)

    def test_no_image_uses_default(self) -> None:
        router = self._router("a4b")
        brain, reason = router.route("Hi there!", has_images=False)
        # "Hi there" matches an _E4B_STRONG hint → drops to e4b. The
        # important assertion is that we did NOT route to multi.
        assert brain != "multi"

    def test_image_forces_multi(self) -> None:
        router = self._router("a4b")
        brain, reason = router.route(
            "What is on this picture?", has_images=True
        )
        assert brain == "multi"
        assert reason == "image_attached"

    def test_explicit_brain_wins_over_image_override(self) -> None:
        # If the user explicitly picks A4B in the brain dropdown and
        # also drops an image, we honour the user choice.
        router = self._router("a4b")
        brain, reason = router.route(
            "What is on this picture?",
            requested="a4b",
            has_images=True,
        )
        assert brain == "a4b"
        assert reason == "explicit"
