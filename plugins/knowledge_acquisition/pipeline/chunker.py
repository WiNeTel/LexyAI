"""
Lexy AI - Knowledge Acquisition: Content Chunker.

Smart paragraph-level text splitting with overlap.
Splits at natural boundaries (double-newline > single-newline > sentence)
to keep semantic units together.
"""

from __future__ import annotations

import re

from lexy_core.utils.logging import get_logger

log = get_logger(module="knowledge.chunker")

# Sentence boundary heuristic: period/question/exclamation followed by space+uppercase
_RE_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u00DC])")


class ContentChunker:
    """Smart paragraph-level text splitting with overlap."""

    def chunk(
        self,
        text: str,
        chunk_size: int = 800,
        overlap: int = 100,
    ) -> list[str]:
        """Split *text* into chunks at paragraph boundaries.

        Priority:
        1. Split at double-newline (paragraphs)
        2. If a paragraph exceeds *chunk_size*, split at single newlines
        3. If still too large, split at sentence boundaries
        4. Last resort: hard split at *chunk_size*

        Each chunk is approximately *chunk_size* characters.
        The last *overlap* characters of the previous chunk are prepended
        to the next chunk for context continuity.

        Returns an empty list when *text* is blank or only whitespace.
        """
        text = text.strip()
        if not text:
            return []

        # Schritt 1: In Paragraphen aufteilen
        paragraphs = self._split_paragraphs(text)

        # Schritt 2: Zu grosse Paragraphen weiter aufbrechen
        segments: list[str] = []
        for para in paragraphs:
            if len(para) <= chunk_size:
                segments.append(para)
            else:
                segments.extend(self._split_large_block(para, chunk_size))

        # Schritt 3: Segmente in Chunks mit Zielgroesse zusammenfassen
        chunks = self._merge_segments(segments, chunk_size)

        # Schritt 4: Overlap hinzufuegen
        if overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks, overlap)

        log.debug(
            "chunker.done",
            input_length=len(text),
            chunk_count=len(chunks),
            avg_chunk_size=sum(len(c) for c in chunks) // max(len(chunks), 1),
        )
        return chunks

    # ── Interne Helfer ──────────────────────────────────────────────

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Split on double-newlines, strip each part, drop empties."""
        parts = re.split(r"\n\s*\n", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _split_large_block(block: str, max_size: int) -> list[str]:
        """Break an oversized paragraph by single-newlines, then sentences."""
        # Erst an Zeilenumbruechen versuchen
        lines = block.split("\n")
        if len(lines) > 1:
            segments: list[str] = []
            for line in lines:
                stripped = line.strip()
                if stripped:
                    segments.append(stripped)
            # Pruefen ob jedes Segment klein genug ist
            all_small = all(len(s) <= max_size for s in segments)
            if all_small:
                return segments
            # Immer noch zu grosse Segmente: an Saetzen splitten
            result: list[str] = []
            for seg in segments:
                if len(seg) <= max_size:
                    result.append(seg)
                else:
                    result.extend(
                        ContentChunker._split_by_sentences(seg, max_size)
                    )
            return result

        # Keine Zeilenumbrueche: an Saetzen splitten
        return ContentChunker._split_by_sentences(block, max_size)

    @staticmethod
    def _split_by_sentences(text: str, max_size: int) -> list[str]:
        """Split text at sentence boundaries. Hard-split as last resort."""
        sentences = _RE_SENTENCE_END.split(text)
        if len(sentences) <= 1 and len(text) > max_size:
            # Kein Satzende gefunden: harter Schnitt
            return ContentChunker._hard_split(text, max_size)

        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = (current + " " + sentence).strip() if current else sentence
            if len(candidate) <= max_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Satz allein zu lang? Hart splitten
                if len(sentence) > max_size:
                    chunks.extend(ContentChunker._hard_split(sentence, max_size))
                    current = ""
                else:
                    current = sentence
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _hard_split(text: str, max_size: int) -> list[str]:
        """Split text at *max_size* boundaries, preferring word boundaries."""
        pieces: list[str] = []
        while len(text) > max_size:
            # Am letzten Leerzeichen vor max_size abschneiden
            split_at = text.rfind(" ", 0, max_size)
            if split_at < max_size // 2:
                split_at = max_size  # Kein brauchbares Leerzeichen
            pieces.append(text[:split_at].rstrip())
            text = text[split_at:].lstrip()
        if text:
            pieces.append(text)
        return pieces

    @staticmethod
    def _merge_segments(
        segments: list[str], target_size: int
    ) -> list[str]:
        """Merge small consecutive segments until they approach *target_size*."""
        if not segments:
            return []
        chunks: list[str] = []
        current = segments[0]
        for segment in segments[1:]:
            candidate = current + "\n\n" + segment
            if len(candidate) <= target_size:
                current = candidate
            else:
                chunks.append(current)
                current = segment
        chunks.append(current)
        return chunks

    @staticmethod
    def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
        """Prepend the last *overlap* characters of the previous chunk."""
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            # Overlap-Text vom Ende des vorherigen Chunks
            overlap_text = prev[-overlap:] if len(prev) >= overlap else prev
            # Am Wortanfang abschneiden
            space_idx = overlap_text.find(" ")
            if space_idx >= 0:
                overlap_text = overlap_text[space_idx + 1 :]
            if overlap_text:
                result.append(overlap_text + " " + chunks[i])
            else:
                result.append(chunks[i])
        return result
