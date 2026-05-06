"""
Code-file parser — really just plain-text decode + language detection
based on file extension. We deliberately don't run a syntax parser:

* The LLM doesn't need an AST, it can read source directly.
* Different languages need different parsers — fragile and heavy.
* Detection by extension is 99 % accurate and trivially extendable.

If a future feature wants AST-aware processing (e.g. function-level
chunking for big repos), that lives in a separate module. This one is
the dumb, fast common case.
"""

from __future__ import annotations

from .text_parser import parse_text


# Extension → language slug. The slug is what we put in the prompt as
# "code (python)" so the LLM picks the right syntax highlighter mentally.
# Order doesn't matter; lookup is by exact extension.
_EXT_TO_LANG: dict[str, str] = {
    "py":   "python",
    "pyi":  "python",
    "js":   "javascript",
    "mjs":  "javascript",
    "cjs":  "javascript",
    "jsx":  "javascript",
    "ts":   "typescript",
    "tsx":  "typescript",
    "go":   "go",
    "rs":   "rust",
    "java": "java",
    "kt":   "kotlin",
    "kts":  "kotlin",
    "swift": "swift",
    "c":    "c",
    "h":    "c",
    "cpp":  "cpp",
    "cxx":  "cpp",
    "cc":   "cpp",
    "hpp":  "cpp",
    "hxx":  "cpp",
    "cs":   "csharp",
    "rb":   "ruby",
    "php":  "php",
    "sh":   "shell",
    "bash": "shell",
    "zsh":  "shell",
    "ps1":  "powershell",
    "lua":  "lua",
    "r":    "r",
    "jl":   "julia",
    "ex":   "elixir",
    "exs":  "elixir",
    "erl":  "erlang",
    "hs":   "haskell",
    "ml":   "ocaml",
    "fs":   "fsharp",
    "vue":  "vue",
    "svelte": "svelte",
    "css":  "css",
    "scss": "scss",
    "sass": "scss",
    "less": "less",
    "html": "html",
    "htm":  "html",
    "xml":  "xml",
    "yaml": "yaml",
    "yml":  "yaml",
    "toml": "toml",
    "json": "json",
    "ini":  "ini",
    "sql":  "sql",
    "graphql": "graphql",
    "gql":  "graphql",
    "proto": "protobuf",
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "mk":   "makefile",
    "tf":   "terraform",
    "hcl":  "hcl",
}


def detect_language(filename: str) -> str:
    """Return the language slug for a filename, or "" if unknown."""
    if not filename:
        return ""
    lower = filename.lower()
    # Handle filename-as-extension (Dockerfile, Makefile).
    base = lower.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if base in _EXT_TO_LANG:
        return _EXT_TO_LANG[base]
    if "." in base:
        ext = base.rsplit(".", 1)[-1]
        return _EXT_TO_LANG.get(ext, "")
    return ""


def parse_code(data: bytes, filename: str = "") -> tuple[str, str, int]:
    """Decode source, return (text, language, line_count)."""
    text = parse_text(data)
    lang = detect_language(filename)
    line_count = text.count("\n") + (0 if not text or text.endswith("\n") else 1)
    return text, lang, line_count
