"""Extração e limpeza de texto corrido (PyMuPDF).

Funções puras sempre que possível para poderem ser testadas sem PDF.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Sequence

import fitz  # PyMuPDF

from .models import ParseOptions, ParsedPage

# --- padrões -------------------------------------------------------------

# Run de 20+ letras Latin-1 Supplement contíguas. É exatamente o artefato
# "ÀÁÂÃÄÅÆÇÈÉÊË...ÿ" que aparece em cabeçalhos de PDFs mal gerados.
# Português legítimo nunca produz uma sequência assim.
_MOJIBAKE_RUN = re.compile(r"[\u00C0-\u00FF]{20,}")

# Controles C1 (0x80–0x9F): nunca legítimos em texto.
_C1_CONTROLS = re.compile(r"[\u0080-\u009F]")

# Hífen de quebra de linha: "trans-\nformador" → "transformador".
# A faixa [À-ÖØ-öø-ÿ] exclui × (U+00D7) e ÷ (U+00F7) dos "letters".
_HYPHEN_BREAK = re.compile(
    r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ])-[ \t]*\n[ \t]*(?=[a-zà-öø-ÿ])"
)

# Espaços/tabs internos (não consome \n).
_MULTISPACE = re.compile(r"[ \t]+")

# Espaços antes/depois de \n (limpa linhas em branco com sujeira).
_EDGE_OF_LINE = re.compile(r"[ \t]*\n[ \t]*")

# 3+ quebras viram 2 (parágrafo).
_BLANK_LINES = re.compile(r"\n{3,}")


# --- limpeza -------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Normaliza para NFKC (compõe acentos, uniformiza formas)."""
    return unicodedata.normalize("NFKC", text)


def remove_mojibake(text: str) -> str:
    """Remove o run clássico de Latin-1 Supplement usado como cabeçalho."""
    return _MOJIBAKE_RUN.sub("", text)


def fix_hyphenation(text: str) -> str:
    """Junta palavras quebradas por hífen no fim de linha."""
    return _HYPHEN_BREAK.sub("", text)


def strip_repeated_headers_footers(text: str, patterns: Sequence[str]) -> str:
    """Remove linhas cujo conteúdo casa com ``patterns`` (regex, MULTILINE)."""
    if not patterns:
        return text
    compiled = [re.compile(p, re.MULTILINE) for p in patterns]
    keep: list[str] = []
    for line in text.splitlines():
        if any(p.search(line) for p in compiled):
            continue
        keep.append(line)
    return "\n".join(keep)


def collapse_whitespace(text: str) -> str:
    """Compacta espaços/tabs internos e normaliza quebras de linha."""
    text = _EDGE_OF_LINE.sub("\n", text)
    text = _MULTISPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def clean_text(raw: str, options: ParseOptions) -> str:
    """Aplica a cadeia completa de limpeza na ordem correta."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00A0", " ")  # NBSP → espaço
    text = _C1_CONTROLS.sub("", text)

    if options.remove_mojibake:
        text = remove_mojibake(text)

    if options.normalize_unicode:
        text = normalize_unicode(text)

    if options.strip_headers_footers:
        text = strip_repeated_headers_footers(text, options.header_footer_patterns)

    if options.fix_hyphenation:
        text = fix_hyphenation(text)

    if options.collapse_whitespace:
        text = collapse_whitespace(text)

    return text


# --- extração ------------------------------------------------------------

def extract_text_pages(
    pdf_path: str | Path,
    options: ParseOptions | None = None,
    *,
    password: str | None = None,
) -> tuple[ParsedPage, ...]:
    """Extrai e limpa o texto página a página.

    Retorna tupla imutável na ordem das páginas (``number`` começa em 1).
    """
    options = options or ParseOptions()
    path = Path(pdf_path)

    pages: list[ParsedPage] = []
    with fitz.open(str(path)) as doc:
        if doc.is_encrypted and (password is None or not doc.authenticate(password)):
            raise RuntimeError(f"PDF protegido por senha: {path}")
        for idx, page in enumerate(doc, start=1):
            raw = page.get_text("text")
            cleaned = clean_text(raw, options)
            rect = page.rect
            pages.append(
                ParsedPage(
                    number=idx,
                    raw_text=raw,
                    text=cleaned,
                    width=float(rect.width),
                    height=float(rect.height),
                )
            )
    return tuple(pages)