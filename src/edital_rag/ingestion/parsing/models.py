"""Tipos imutáveis da camada de parsing.

Nada aqui faz I/O. São contratos que viajam entre `pdf_text`, `pdf_tables`
e a camada de chunking. Manter este módulo sem dependências externas
facilita testes e evita import circular.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Uma linha de tabela é uma tupla de strings (células já limpas).
# Uma tabela é uma tupla de linhas.
Row = tuple[str, ...]
Table = tuple[Row, ...]


@dataclass(frozen=True, slots=True)
class ParsedPage:
    """Uma página do PDF.

    - ``raw_text``: saída crua do PyMuPDF (para debug).
    - ``text``: texto após ``clean_text`` (é o que segue para chunking).
    """

    number: int  # 1-based
    raw_text: str
    text: str
    width: float | None = None
    height: float | None = None


@dataclass(frozen=True, slots=True)
class ParsedTable:
    """Uma tabela extraída do PDF.

    ``rows`` já vem limpo (células sem ``\\n``, sem espaços duplicados,
    ``None`` → ``""``). O ``header`` é detectado heuristicamente; ``None``
    quando não há cabeçalho reconhecível.
    """

    page: int
    index: int  # índice da tabela dentro da página, 0-based
    rows: Table
    header: Row | None = None
    bbox: tuple[float, float, float, float] | None = None

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.rows[0]) if self.rows else 0

    @property
    def data_rows(self) -> Table:
        """Linhas de dados (exclui o cabeçalho, quando detectado)."""
        if self.header is None:
            return self.rows
        return self.rows[1:]

    def column_index(self, name: str) -> int | None:
        """Índice da coluna cujo nome de cabeçalho casa com ``name`` (case-insensitive)."""
        if self.header is None:
            return None
        target = name.strip().upper()
        for i, h in enumerate(self.header):
            if h.strip().upper() == target:
                return i
        return None

    def column(self, name: str) -> tuple[str, ...]:
        """Valores de uma coluna de dados, na ordem das linhas."""
        idx = self.column_index(name)
        if idx is None:
            return ()
        return tuple(r[idx] if idx < len(r) else "" for r in self.data_rows)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Resultado completo de ``parse_document``."""

    path: Path
    sha256: str
    size_bytes: int
    page_count: int
    pages: tuple[ParsedPage, ...]
    tables: tuple[ParsedTable, ...] = ()

    @property
    def pages_text(self) -> str:
        """Concatena o texto limpo de todas as páginas com um separador duplo."""
        return "\n\n".join(p.text for p in self.pages if p.text)

    def tables_on_page(self, page: int) -> tuple[ParsedTable, ...]:
        return tuple(t for t in self.tables if t.page == page)


@dataclass(frozen=True, slots=True)
class ParseOptions:
    """Configuração do parsing. Todos os defaults visam editais em PT-BR."""

    # --- limpeza de texto -------------------------------------------------
    normalize_unicode: bool = True
    remove_mojibake: bool = True
    fix_hyphenation: bool = True
    collapse_whitespace: bool = True
    strip_headers_footers: bool = True
    header_footer_patterns: tuple[str, ...] = (
        r"^\s*P[áa]gina\s+\d+(\s+de\s+\d+)?\s*$",
        r"^\s*Fls\.?\s*n?[ºo°]?\s*[_\s]*$",
        r"^\s*Rub\.?\s*[_\s]*$",
        r"^\s*_{5,}\s*$",
    )

    # --- extração de tabelas ---------------------------------------------
    extract_tables: bool = True
    table_min_rows: int = 2
    table_min_cols: int = 2
    table_min_cell_chars: int = 1
    table_forward_fill: bool = True
    table_forward_fill_headers: tuple[str, ...] = (
        "ITEM",
        "CÓDIGO",
        "CODIGO",
        "LOTE",
        "PRODUTO",
    )