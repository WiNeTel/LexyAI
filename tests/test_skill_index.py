"""
Tests for the Phase-P6 skill-catalogue injection (``build_catalog_block``).

Pure-function tests — verify that the catalogue lists active skills (name +
description only), excludes archived/disabled ones, bounds itself to
``inline_max`` preferring most-recently-used, and returns ``None`` when empty.
"""

from __future__ import annotations

from types import SimpleNamespace

from plugins.skill_writer.skill_index import build_catalog_block


def _skill(
    name: str,
    *,
    description: str = "does a thing",
    status: str = "active",
    state: str = "active",
    last_used_at: float | None = None,
    created_at: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=description,
        status=status,
        state=state,
        last_used_at=last_used_at,
        created_at=created_at,
    )


def test_empty_returns_none() -> None:
    assert build_catalog_block([]) is None


def test_lists_active_skills_with_descriptions() -> None:
    block = build_catalog_block(
        [
            _skill("parse-pdf", description="extract tables from PDFs"),
            _skill("fetch-weather", description="get the forecast"),
        ]
    )
    assert block is not None
    assert "**parse-pdf**: extract tables from PDFs" in block
    assert "**fetch-weather**: get the forecast" in block
    assert "run_skill" in block  # tells the model how to use them


def test_excludes_archived_and_disabled() -> None:
    block = build_catalog_block(
        [
            _skill("live-one"),
            _skill("archived-one", state="archived"),
            _skill("disabled-one", status="disabled"),
        ]
    )
    assert block is not None
    assert "live-one" in block
    assert "archived-one" not in block
    assert "disabled-one" not in block


def test_all_archived_returns_none() -> None:
    assert (
        build_catalog_block([_skill("a", state="archived")]) is None
    )


def test_bounds_to_inline_max_preferring_recent() -> None:
    skills = [
        _skill(f"s{i}", last_used_at=float(i)) for i in range(20)
    ]
    block = build_catalog_block(skills, inline_max=3)
    assert block is not None
    # The 3 highest last_used_at (s19, s18, s17) win.
    assert "**s19**" in block
    assert "**s18**" in block
    assert "**s17**" in block
    assert "**s0**" not in block
    # And the user is told more exist.
    assert "list_skills" in block


def test_long_description_is_truncated() -> None:
    block = build_catalog_block(
        [_skill("x", description="A" * 500)], max_desc_len=40
    )
    assert block is not None
    assert "…" in block
    # The bullet line stays bounded.
    bullet = next(ln for ln in block.splitlines() if ln.startswith("- **x**"))
    assert len(bullet) < 80
