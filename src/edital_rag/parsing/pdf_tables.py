"""
Extração de tabelas de PDFs usando pdfplumber.

Retorna uma estrutura TabelaExtraida que preserva:
- a página de origem
- os dados como matriz (list[list[str | None]])
- o cabeçalho inferido (primeira linha não vazia)
- o texto linearizado (pipe-separated) para uso direto em chunks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber


@dataclass
class TabelaExtraida:
    """Representa uma tabela extraída de uma página do PDF."""

    pagina: int  # 0-indexed
    indice_na_pagina: int  # múltiplas tabelas na mesma página
    dados: list[list[str | None]]
    cabecalho: list[str] = field(default_factory=list)
    linhas: list[list[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Normaliza None → "" e faz strip em cada célula
        self.dados = [
            [(str(c).strip() if c is not None else "") for c in linha]
            for linha in self.dados
        ]
        # Infere cabeçalho: primeira linha não totalmente vazia
        self.cabecalho = self._inferir_cabecalho()
        # Linhas de dados (excluindo cabeçalho)
        if self.cabecalho:
            self.linhas = self.dados[1:]
        else:
            self.linhas = self.dados

    def _inferir_cabecalho(self) -> list[str]:
        for linha in self.dados:
            if any(celula for celula in linha):
                return [str(c) for c in linha]
        return []

    @property
    def num_colunas(self) -> int:
        return max((len(l) for l in self.dados), default=0)

    @property
    def num_linhas_dados(self) -> int:
        return len(self.linhas)


def _normalizar_tabela(tabela: list[list[Any]]) -> list[list[str | None]]:
    """
    pdfplumber pode retornar linhas com número diferente de colunas.
    Aqui padronizamos todas ao maior comprimento, preenchendo com None.
    """
    if not tabela:
        return []
    max_cols = max(len(linha) for linha in tabela)
    normalizada: list[list[str | None]] = []
    for linha in tabela:
        linha_pad = list(linha) + [None] * (max_cols - len(linha))
        normalizada.append(linha_pad)
    return normalizada


def extrair_tabelas_pdf(
    caminho_pdf: str | Path,
    *,
    table_settings: dict | None = None,
) -> list[TabelaExtraida]:
    """
    Extrai todas as tabelas do PDF.

    Args:
        caminho_pdf: caminho do arquivo PDF.
        table_settings: dicionário opcional com parâmetros do pdfplumber
            (ex.: {"vertical_strategy": "lines", "horizontal_strategy": "lines"}).

    Returns:
        Lista de TabelaExtraida na ordem em que aparecem no documento.
    """
    caminho = Path(caminho_pdf)
    if not caminho.exists():
        raise FileNotFoundError(f"PDF não encontrado: {caminho}")

    settings = table_settings or {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 10,
    }

    tabelas: list[TabelaExtraida] = []

    with pdfplumber.open(caminho) as pdf:
        for idx_pagina, pagina in enumerate(pdf.pages):
            try:
                tabelas_brutas = pagina.extract_tables(table_settings=settings)
            except Exception:
                # PDFs malformados podem quebrar o extract_tables;
                # seguimos o processamento sem interromper o pipeline.
                tabelas_brutas = []

            for idx_tabela, tabela in enumerate(tabelas_brutas):
                if not tabela:
                    continue
                tabela_norm = _normalizar_tabela(tabela)
                tabelas.append(
                    TabelaExtraida(
                        pagina=idx_pagina,
                        indice_na_pagina=idx_tabela,
                        dados=tabela_norm,
                    )
                )

    return tabelas


def tabela_para_texto(
    tabela: TabelaExtraida,
    *,
    incluir_cabecalho: bool = True,
) -> str:
    """
    Converte uma TabelaExtraida em texto pipe-separated.

    Se incluir_cabecalho=True, prefixa cada linha de dados com o cabeçalho
    no formato "col1: val1 | col2: val2 | ...". Isso ajuda o LLM a entender
    o significado das colunas em chunks isolados.
    """
    if not tabela.dados:
        return ""

    linhas_saida: list[str] = []

    if incluir_cabecalho and tabela.cabecalho:
        for linha in tabela.linhas:
            pares = [
                f"{cab}: {val}"
                for cab, val in zip(tabela.cabecalho, linha)
                if val
            ]
            if pares:
                linhas_saida.append(" | ".join(pares))
    else:
        for linha in tabela.dados:
            linhas_saida.append(" | ".join(c for c in linha if c))

    return "\n".join(linhas_saida)