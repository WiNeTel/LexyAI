"""
Lexy AI - Knowledge Acquisition: Training Data Exporter.

Reads stored knowledge chunks from SQLite, generates instruction-response
pairs via the LLM, and writes them as JSONL files suitable for LoRA
fine-tuning (standard and Alpaca format).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import aiosqlite

from lexy_core.utils.logging import get_logger

log = get_logger(module="knowledge.exporter")

_INSTRUCTION_PROMPT = (
    "Given the following text, generate a clear, natural instruction or question "
    "that this text would be a good answer to. The instruction should be specific "
    "enough that the text fully answers it.\n\n"
    "Rules:\n"
    "- Write ONLY the instruction/question, nothing else.\n"
    "- Do not start with 'What is' every time — vary the phrasing.\n"
    "- The instruction must be in the same language as the text.\n\n"
    "Text:\n{chunk}"
)


class TrainingDataExporter:
    """Export knowledge chunks as JSONL for LoRA fine-tuning."""

    async def export_jsonl(
        self,
        db: aiosqlite.Connection,
        export_path: str | Path,
        category: str | None = None,
        min_quality: int = 3,
        api: Any = None,
    ) -> dict[str, Any]:
        """Read chunks from SQLite, generate instruction-response JSONL.

        Each line: ``{"instruction": "...", "input": "", "output": "...",
        "category": "...", "source_url": "..."}``.

        Returns stats about the export.
        """
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)

        chunks = await self._fetch_chunks(db, category, min_quality)
        if not chunks:
            log.info("exporter.no_chunks", category=category, min_quality=min_quality)
            return {"exported": 0, "file": None, "error": None}

        timestamp = int(time.time())
        suffix = f"_{category}" if category else "_all"
        filename = f"knowledge{suffix}_{timestamp}.jsonl"
        filepath = export_dir / filename

        exported = 0
        errors = 0

        with open(filepath, "w", encoding="utf-8") as fh:
            for chunk_row in chunks:
                chunk_id, text, cat, source_url, quality = chunk_row

                instruction = await self._generate_instruction(text, api)
                if not instruction:
                    errors += 1
                    continue

                record = {
                    "instruction": instruction,
                    "input": "",
                    "output": text,
                    "category": cat,
                    "source_url": source_url or "",
                    "quality_score": quality,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1

        log.info(
            "exporter.jsonl_done",
            file=str(filepath),
            exported=exported,
            errors=errors,
        )
        return {
            "exported": exported,
            "errors": errors,
            "file": str(filepath),
            "format": "jsonl",
        }

    async def export_alpaca(
        self,
        db: aiosqlite.Connection,
        export_path: str | Path,
        category: str | None = None,
        min_quality: int = 3,
        api: Any = None,
    ) -> dict[str, Any]:
        """Alpaca format: ``{"instruction": "...", "input": "...", "output": "..."}``.

        Writes a JSON array (not JSONL) for compatibility with Alpaca
        training scripts.  Returns stats about the export.
        """
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)

        chunks = await self._fetch_chunks(db, category, min_quality)
        if not chunks:
            log.info("exporter.no_chunks", category=category, min_quality=min_quality)
            return {"exported": 0, "file": None, "error": None}

        timestamp = int(time.time())
        suffix = f"_{category}" if category else "_all"
        filename = f"alpaca{suffix}_{timestamp}.json"
        filepath = export_dir / filename

        records: list[dict[str, str]] = []
        errors = 0

        for chunk_row in chunks:
            chunk_id, text, cat, source_url, quality = chunk_row

            instruction = await self._generate_instruction(text, api)
            if not instruction:
                errors += 1
                continue

            records.append(
                {
                    "instruction": instruction,
                    "input": "",
                    "output": text,
                }
            )

        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(records, fh, ensure_ascii=False, indent=2)

        log.info(
            "exporter.alpaca_done",
            file=str(filepath),
            exported=len(records),
            errors=errors,
        )
        return {
            "exported": len(records),
            "errors": errors,
            "file": str(filepath),
            "format": "alpaca",
        }

    # ── Interne Helfer ──────────────────────────────────────────────

    @staticmethod
    async def _fetch_chunks(
        db: aiosqlite.Connection,
        category: str | None,
        min_quality: int,
    ) -> list[tuple[str, str, str, str, int]]:
        """Fetch qualifying chunks from SQLite."""
        sql = (
            "SELECT id, content, category, source_url, quality_score "
            "FROM chunks WHERE quality_score >= ?"
        )
        params: list[Any] = [min_quality]

        if category:
            sql += " AND category = ?"
            params.append(category)

        sql += " ORDER BY quality_score DESC, created_at DESC"

        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return rows  # type: ignore[return-value]

    async def _generate_instruction(
        self,
        chunk: str,
        api: Any,
    ) -> str:
        """Ask LLM to generate an instruction that *chunk* answers."""
        if api is None:
            # Ohne LLM: generische Instruktion
            preview = chunk[:80].replace("\n", " ").strip()
            return f"Explain the following: {preview}..."

        prompt = _INSTRUCTION_PROMPT.format(chunk=chunk[:1500])

        try:
            response = await api.llm_chat(
                messages=[{"role": "user", "content": prompt}],
                brain="e4b",
                max_tokens=100,
                temperature=0.7,
            )
            instruction = response.strip()
            if instruction:
                return instruction
        except Exception as exc:
            log.warning("exporter.instruction_error", error=str(exc))

        # Fallback: generische Instruktion
        preview = chunk[:80].replace("\n", " ").strip()
        return f"Explain the following: {preview}..."
