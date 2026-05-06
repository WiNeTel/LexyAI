"""Document parsers used by the uploads pipeline."""

from .text_parser import parse_text
from .pdf_parser import parse_pdf
from .html_parser import parse_html
from .code_parser import parse_code, detect_language

__all__ = [
    "parse_text",
    "parse_pdf",
    "parse_html",
    "parse_code",
    "detect_language",
]
