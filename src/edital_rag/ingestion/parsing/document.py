"""Orquestração de parsing de um documento inteiro."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .models import ParseOptions, ParsedDocument
from .pdf_tables import extract_tables
from .pdf_text import extract_text_pages


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def parse_document(
    pdf_path: str | Path,
    options: ParseOptions | None = None,
    *,
    password: str | None = None,
) -> ParsedDocument:
    """Executa texto + tabelas e devolve um ``ParsedDocument`` imutável."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF não encontrado: {path}")
    options = options or ParseOptions()

    sha = _sha256_file(path)
    size = path.stat().st_size

    pages = extract_text_pages(path, options, password=password)
    tables = (
        extract_tables(path, options, password=password)
        if options.extract_tables
        else ()
    )

    return ParsedDocument(
        path=path,
        sha256=sha,
        size_bytes=size,
        page_count=len(pages),
        pages=pages,
        tables=tables,
    )