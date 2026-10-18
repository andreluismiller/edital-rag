"""Extração de tabelas (pdfplumber).

Estratégia:

1. ``page.find_tables()`` — devolve ``Table`` com ``bbox`` e ``extract()``.
   É a API mais útil porque preserva a geometria.
2. Fallback para ``page.extract_tables()`` quando ``find_tables`` falha
   ou devolve vazio em página que claramente tem conteúdo tabular.
3. Limpeza de células, remoção de linhas vazias e de colunas vazias.
4. Detecção heurística de cabeçalho.
5. Forward-fill opcional de colunas identificadoras (ITEM, CÓDIGO, LOTE),
   porque TRs costumam mesclar essas células ao longo de várias linhas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pdfplumber

from .models import ParseOptions, ParsedTable, Row, Table

_NULL_LIKE = frozenset({"", "-", "—", "–", "n/a", "na", "n/d"})

# Termos que, se aparecerem juntos no início de uma linha, sugerem cabeçalho.
_HEADER_TERMS = (
    "ITEM",
    "CÓDIGO",
    "CODIGO",
    "ESPECIFICA",
    "QUANT",
    "VALOR",
    "UNIDADE",
    "DESCRI",
    "PRODUTO",
    "LOTE",
    "TOTAL",
)


# --- limpeza de células -------------------------------------------------

def _clean_cell(cell: object, min_chars: int) -> str:
    if cell is None:
        return ""
    s = str(cell).replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    if s.strip().lower() in _NULL_LIKE:
        return ""
    if len(s) < min_chars:
        return ""
    return s


def _clean_row(row: Sequence[object], min_chars: int) -> Row:
    return tuple(_clean_cell(c, min_chars) for c in row)


def _pad_and_trim(rows: list[Row]) -> list[Row]:
    """Padroniza largura e remove colunas inteiramente vazias."""
    if not rows:
        return rows
    n = max(len(r) for r in rows)
    padded = [r + ("",) * (n - len(r)) for r in rows]
    keep = [any(r[i] for r in padded) for i in range(n)]
    return [tuple(c for c, k in zip(r, keep) if k) for r in padded]


# --- cabeçalho e forward-fill ------------------------------------------

def _detect_header(rows: list[Row]) -> int | None:
    """Retorna o índice da linha de cabeçalho, ou ``None``.

    Considera apenas as 3 primeiras linhas e exige ao menos 2 termos
    típicos de cabeçalho.
    """
    for i, row in enumerate(rows[:3]):
        joined = " ".join(row).upper()
        hits = sum(1 for k in _HEADER_TERMS if k in joined)
        if hits >= 2:
            return i
    return None


def forward_fill_rows(
    rows: list[Row],
    header: Row,
    fill_headers: Sequence[str],
) -> list[Row]:
    """Preenche células vazias em colunas identificadoras com o último valor visto.

    Essencial para TRs em que ITEM/CÓDIGO aparecem só na primeira linha física
    de cada item e a especificação se estende por várias linhas.
    """
    if not fill_headers:
        return rows
    targets_upper = {h.upper() for h in fill_headers}
    targets = {i for i, h in enumerate(header) if h.strip().upper() in targets_upper}
    if not targets:
        return rows

    out: list[Row] = []
    last = [""] * len(header)
    for row in rows:
        row = row + ("",) * (len(header) - len(row))
        new = list(row)
        for i in targets:
            if not new[i]:
                new[i] = last[i]
            else:
                last[i] = new[i]
        out.append(tuple(new))
    return out


def coalesce_rows(
    rows: list[Row],
    header: Row,
    id_headers: Sequence[str],
    concat_headers: Sequence[str] | None = None,
) -> list[Row]:
    """Funde linhas que compartilham as mesmas colunas identificadoras.

    Útil quando o chunker precisa de **um item = uma linha lógica**, com a
    especificação concatenada. A concatenação é feita com espaço nas colunas
    de ``concat_headers`` (por padrão, todas exceto as identificadoras).

    .. note::
       A camada de chunking pode preferir fazer sua própria fusão por razões
       de layout; por isso esta função é exposta publicamente.
    """
    targets_upper = {h.upper() for h in id_headers}
    id_idx = {i for i, h in enumerate(header) if h.strip().upper() in targets_upper}
    if not id_idx:
        return rows

    if concat_headers is None:
        concat_idx = {i for i in range(len(header)) if i not in id_idx}
    else:
        concat_upper = {h.upper() for h in concat_headers}
        concat_idx = {i for i, h in enumerate(header) if h.strip().upper() in concat_upper}

    out: list[Row] = []
    by_key: dict[tuple[str, ...], list[str]] = {}
    order: list[tuple[str, ...]] = []

    for row in rows:
        row = row + ("",) * (len(header) - len(row))
        key = tuple(row[i] for i in sorted(id_idx))
        if not any(key):
            continue
        if key not in by_key:
            by_key[key] = list(row)
            order.append(key)
        else:
            merged = by_key[key]
            for i in concat_idx:
                if not row[i]:
                    continue
                merged[i] = f"{merged[i]} {row[i]}".strip() if merged[i] else row[i]

    for key in order:
        out.append(tuple(by_key[key]))
    return out


# --- extração -----------------------------------------------------------

def _tables_from_page(page: pdfplumber.page.Page) -> list[tuple[list[list[object | None]], tuple[float, float, float, float] | None]]:
    """Tenta ``find_tables`` e cai para ``extract_tables``.

    Devolve pares ``(raw_rows, bbox)``.
    """
    result: list[tuple[list[list[object | None]], tuple[float, float, float, float] | None]] = []

    try:
        finder = page.find_tables(
            table_settings={
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 3,
                "intersection_tolerance": 5,
            }
        )
        for table in finder.tables:
            try:
                data = table.extract()
            except Exception:
                continue
            if not data:
                continue
            bbox = tuple(float(v) for v in table.bbox) if table.bbox else None
            result.append((data, bbox))
    except Exception:
        result = []

    if result:
        return result

    try:
        raw = page.extract_tables() or []
    except Exception:
        return []
    return [(t, None) for t in raw]


def extract_tables(
    pdf_path: str | Path,
    options: ParseOptions | None = None,
    *,
    password: str | None = None,
) -> tuple[ParsedTable, ...]:
    """Extrai todas as tabelas relevantes do PDF."""
    options = options or ParseOptions()
    if not options.extract_tables:
        return ()

    tables: list[ParsedTable] = []
    with pdfplumber.open(str(pdf_path), password=password) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            for table_idx, (raw, bbox) in enumerate(_tables_from_page(page)):
                rows = [
                    _clean_row(r, options.table_min_cell_chars)
                    for r in raw
                ]
                rows = [r for r in rows if any(r)]
                if len(rows) < options.table_min_rows:
                    continue
                rows = _pad_and_trim(rows)
                if not rows or len(rows[0]) < options.table_min_cols:
                    continue

                h_idx = _detect_header(rows)
                header: Row | None = rows[h_idx] if h_idx is not None else None

                data = rows[h_idx + 1 :] if h_idx is not None else rows
                if options.table_forward_fill and header is not None:
                    data = forward_fill_rows(
                        data, header, options.table_forward_fill_headers
                    )

                full_rows: list[Row] = ([header] if header is not None else []) + data
                tables.append(
                    ParsedTable(
                        page=page_idx,
                        index=table_idx,
                        rows=tuple(full_rows),
                        header=header,
                        bbox=bbox,
                    )
                )

    return tuple(tables)