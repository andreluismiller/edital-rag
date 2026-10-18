"""Camada de parsing: PDF → ``ParsedDocument`` limpo e tipado."""

from .document import parse_document
from .models import (
    ParseOptions,
    ParsedDocument,
    ParsedPage,
    ParsedTable,
    Row,
    Table,
)
from .pdf_tables import (
    coalesce_rows,
    extract_tables,
    forward_fill_rows,
)
from .pdf_text import (
    clean_text,
    collapse_whitespace,
    extract_text_pages,
    fix_hyphenation,
    normalize_unicode,
    remove_mojibake,
    strip_repeated_headers_footers,
)

__all__ = [
    # tipos
    "ParseOptions",
    "ParsedDocument",
    "ParsedPage",
    "ParsedTable",
    "Row",
    "Table",
    # texto
    "clean_text",
    "collapse_whitespace",
    "extract_text_pages",
    "fix_hyphenation",
    "normalize_unicode",
    "remove_mojibake",
    "strip_repeated_headers_footers",
    # tabelas
    "coalesce_rows",
    "extract_tables",
    "forward_fill_rows",
    # orquestração
    "parse_document",
]