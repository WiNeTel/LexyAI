"""
Phase 11 — Tests for the agentskills.io frontmatter parser/validator.

The spec is small but rigid: ``name`` has a precise regex, ``description``
has a length cap, the parent folder MUST equal ``name``, and a few
optional fields round-trip. These tests pin every rule from
agentskills.io/specification so a future YAML-parser swap or the
introduction of a new field can't silently relax constraints.
"""

from __future__ import annotations

import pytest

from plugins.skill_writer.skill_spec import (
    DESCRIPTION_MAX_LEN,
    NAME_MAX_LEN,
    SkillDescriptionError,
    SkillFrontmatter,
    SkillFrontmatterError,
    SkillNameError,
    SkillSpecError,
    parse_skill_md,
    render_skill_md,
    validate_skill_compatibility,
    validate_skill_description,
    validate_skill_name,
)


# ─── Name validation ─────────────────────────────────────────────────


class TestSkillNameValidation:
    """Spec rules for the ``name`` field."""

    @pytest.mark.parametrize(
        "name",
        [
            "pdf-extract",
            "data-analysis",
            "code-review",
            "a",                         # 1 char
            "skill42",
            "a-b-c",
            "skill-with-many-hyphens-but-no-doubles",
            "x" * NAME_MAX_LEN,           # exactly at the limit
        ],
    )
    def test_valid_names_accepted(self, name: str) -> None:
        validate_skill_name(name)  # must not raise

    @pytest.mark.parametrize(
        "name",
        [
            "PDF-Extract",              # uppercase
            "pdf_extract",              # underscore not allowed
            "-pdf",                     # leading hyphen
            "pdf-",                     # trailing hyphen
            "pdf--extract",             # consecutive hyphens
            "pdf extract",              # space
            "skäill",                   # non-ascii
            "1" * (NAME_MAX_LEN + 1),    # too long
            "",                          # empty
        ],
    )
    def test_invalid_names_rejected(self, name: str) -> None:
        with pytest.raises(SkillNameError):
            validate_skill_name(name)

    def test_parent_dir_mismatch_raises(self) -> None:
        with pytest.raises(SkillNameError):
            validate_skill_name(
                "pdf-extract", parent_dir_name="something-else"
            )

    def test_parent_dir_match_passes(self) -> None:
        validate_skill_name("pdf-extract", parent_dir_name="pdf-extract")

    def test_non_string_name_rejected(self) -> None:
        with pytest.raises(SkillNameError):
            validate_skill_name(42)  # type: ignore[arg-type]


# ─── Description validation ──────────────────────────────────────────


class TestSkillDescriptionValidation:
    def test_valid_description(self) -> None:
        validate_skill_description("Extract text from PDFs.")

    def test_empty_description_rejected(self) -> None:
        with pytest.raises(SkillDescriptionError):
            validate_skill_description("")

    def test_whitespace_only_rejected(self) -> None:
        with pytest.raises(SkillDescriptionError):
            validate_skill_description("   \n\t   ")

    def test_too_long_rejected(self) -> None:
        with pytest.raises(SkillDescriptionError):
            validate_skill_description("x" * (DESCRIPTION_MAX_LEN + 1))

    def test_at_limit_passes(self) -> None:
        validate_skill_description("x" * DESCRIPTION_MAX_LEN)

    def test_non_string_rejected(self) -> None:
        with pytest.raises(SkillDescriptionError):
            validate_skill_description(123)  # type: ignore[arg-type]


# ─── Compatibility validation ────────────────────────────────────────


class TestCompatibilityValidation:
    def test_none_is_fine(self) -> None:
        validate_skill_compatibility(None)

    def test_short_string_passes(self) -> None:
        validate_skill_compatibility("Requires Python 3.11")

    def test_too_long_rejected(self) -> None:
        with pytest.raises(SkillSpecError):
            validate_skill_compatibility("y" * 501)


# ─── Parse minimal example from the spec ─────────────────────────────


class TestParseSkillMd:
    def test_minimal_example(self) -> None:
        text = (
            "---\n"
            "name: skill-name\n"
            "description: A description of what this skill does and when to use it.\n"
            "---\n"
        )
        fm = parse_skill_md(text, parent_dir_name="skill-name")
        assert fm.name == "skill-name"
        assert "when to use" in fm.description
        assert fm.body == ""
        assert fm.license is None
        assert fm.compatibility is None
        assert fm.metadata == {}
        assert fm.allowed_tools is None

    def test_full_example_with_optionals(self) -> None:
        """Phase-11 spec example reproduced from agentskills.io."""
        text = (
            "---\n"
            "name: pdf-processing\n"
            "description: Extract PDF text, fill forms, merge files. "
            "Use when handling PDFs.\n"
            "license: Apache-2.0\n"
            "metadata:\n"
            "  author: example-org\n"
            "  version: \"1.0\"\n"
            "---\n"
            "Body text.\n"
        )
        fm = parse_skill_md(text, parent_dir_name="pdf-processing")
        assert fm.license == "Apache-2.0"
        assert fm.metadata == {"author": "example-org", "version": "1.0"}
        assert "Body text" in fm.body

    def test_parses_compatibility(self) -> None:
        text = (
            "---\n"
            "name: needs-python\n"
            "description: Demo.\n"
            "compatibility: Requires Python 3.11+ and httpx.\n"
            "---\n"
        )
        fm = parse_skill_md(text, parent_dir_name="needs-python")
        assert fm.compatibility == "Requires Python 3.11+ and httpx."

    def test_allowed_tools_string(self) -> None:
        text = (
            "---\n"
            "name: shell-helper\n"
            "description: Demo.\n"
            "allowed-tools: Bash(git:*) Read\n"
            "---\n"
        )
        fm = parse_skill_md(text, parent_dir_name="shell-helper")
        assert fm.allowed_tools == "Bash(git:*) Read"

    def test_allowed_tools_yaml_list_flattened(self) -> None:
        """Some authors yaml-list it; we coerce to space-separated."""
        text = (
            "---\n"
            "name: shell-helper\n"
            "description: Demo.\n"
            "allowed-tools:\n"
            "  - Bash(git:*)\n"
            "  - Read\n"
            "---\n"
        )
        fm = parse_skill_md(text, parent_dir_name="shell-helper")
        assert fm.allowed_tools == "Bash(git:*) Read"

    def test_metadata_coerces_non_string_values(self) -> None:
        """YAML scalars like ``version: 1.0`` are common — we coerce."""
        text = (
            "---\n"
            "name: ok\n"
            "description: D.\n"
            "metadata:\n"
            "  version: 1.0\n"
            "  count: 7\n"
            "---\n"
        )
        fm = parse_skill_md(text, parent_dir_name="ok")
        assert fm.metadata["version"] == "1.0"
        assert fm.metadata["count"] == "7"

    def test_no_frontmatter_raises(self) -> None:
        with pytest.raises(SkillFrontmatterError):
            parse_skill_md("Just markdown body.\n", parent_dir_name="x")

    def test_unterminated_frontmatter_raises(self) -> None:
        with pytest.raises(SkillFrontmatterError):
            parse_skill_md("---\nname: ok\ndescription: D.\n", parent_dir_name="ok")

    def test_yaml_garbage_raises(self) -> None:
        text = "---\n: : not valid\n---\n"
        with pytest.raises(SkillFrontmatterError):
            parse_skill_md(text, parent_dir_name="x")

    def test_missing_name_raises(self) -> None:
        text = "---\ndescription: D.\n---\n"
        with pytest.raises(SkillNameError):
            parse_skill_md(text, parent_dir_name="x")

    def test_missing_description_raises(self) -> None:
        text = "---\nname: x\n---\n"
        with pytest.raises(SkillDescriptionError):
            parse_skill_md(text, parent_dir_name="x")

    def test_parent_dir_mismatch_raises(self) -> None:
        text = "---\nname: a\ndescription: D.\n---\n"
        with pytest.raises(SkillNameError):
            parse_skill_md(text, parent_dir_name="different")


# ─── Render + roundtrip ──────────────────────────────────────────────


class TestRenderRoundtrip:
    def test_render_then_parse_idempotent(self) -> None:
        original = SkillFrontmatter(
            name="round-trip",
            description="Test roundtrip.",
            body="Body here.\n",
            license="MIT",
            compatibility="Linux/macOS",
            metadata={"version": "2.0", "author": "lexy"},
            allowed_tools="Read Write",
        )
        text = render_skill_md(original)
        reparsed = parse_skill_md(text, parent_dir_name="round-trip")
        assert reparsed.name == original.name
        assert reparsed.description == original.description
        assert reparsed.license == original.license
        assert reparsed.compatibility == original.compatibility
        assert reparsed.metadata == original.metadata
        assert reparsed.allowed_tools == original.allowed_tools
        assert "Body here" in reparsed.body

    def test_render_validates_invalid_name(self) -> None:
        bad = SkillFrontmatter(name="UPPER", description="D.")
        with pytest.raises(SkillNameError):
            render_skill_md(bad)

    def test_render_emits_minimal_form(self) -> None:
        fm = SkillFrontmatter(name="ok", description="D.")
        text = render_skill_md(fm)
        # No license/metadata/compatibility lines for a minimal skill.
        assert "license" not in text
        assert "metadata" not in text
        assert "compatibility" not in text
        assert text.startswith("---\nname: ok\ndescription: D.\n---")

    def test_to_public_omits_unset_optionals(self) -> None:
        fm = SkillFrontmatter(name="ok", description="D.")
        public = fm.to_public()
        assert public == {"name": "ok", "description": "D."}

    def test_to_public_includes_metadata(self) -> None:
        fm = SkillFrontmatter(
            name="ok", description="D.", metadata={"k": "v"}
        )
        public = fm.to_public()
        assert public["metadata"] == {"k": "v"}
