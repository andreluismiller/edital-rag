"""
Extração de texto corrido de PDFs usando PyMuPDF (fitz).

Fornece duas APIs:
- extrair_texto_paginas: retorna lista de dicts por página (útil para metadados).
- extrair_texto_pdf: retorna string única concatenada (útil para pipelines simples).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PaginaTexto:
    """Representa o texto extraído de uma página individual."""

    numero: int  # 0-indexed internamente; +1 na exposição
    texto: str
    largura: float = 0.0
    altura: float = 0.0
    blocos: list[dict] = field(default_factory=list)


def extrair_texto_paginas(
    caminho_pdf: str | Path,
    *,
    preservar_blocos: bool = False,
    ordenar_por_posicao: bool = True,
) -> list[PaginaTexto]:
    """
    Extrai o texto de cada página do PDF.

    Args:
        caminho_pdf: caminho para o arquivo PDF.
        preservar_blocos: se True, guarda os blocos brutos (x0, y0, x1, y1, texto).
        ordenar_por_posicao: se True, usa sort=True no get_text para respeitar
            a ordem de leitura (topo-esquerda para baixo-direita).

    Returns:
        Lista de PaginaTexto, uma por página, na ordem do documento.
    """
    caminho = Path(caminho_pdf)
    if not caminho.exists():
        raise FileNotFoundError(f"PDF não encontrado: {caminho}")

    paginas: list[PaginaTexto] = []

    with fitz.open(caminho) as doc:
        for idx, pagina in enumerate(doc):
            texto = pagina.get_text("text", sort=ordenar_por_posicao)

            blocos_brutos: list[dict] = []
            if preservar_blocos:
                # get_text("blocks") retorna tuplas: (x0, y0, x1, y1, texto, num_bloco, tipo)
                for bloco in pagina.get_text("blocks", sort=ordenar_por_posicao):
                    blocos_brutos.append(
                        {
                            "x0": bloco[0],
                            "y0": bloco[1],
                            "x1": bloco[2],
                            "y1": bloco[3],
                            "texto": bloco[4],
                            "tipo": bloco[6] if len(bloco) > 6 else 0,
                        }
                    )

            paginas.append(
                PaginaTexto(
                    numero=idx,
                    texto=texto,
                    largura=pagina.rect.width,
                    altura=pagina.rect.height,
                    blocos=blocos_brutos,
                )
            )

    return paginas


def extrair_texto_pdf(caminho_pdf: str | Path) -> str:
    """
    Extrai todo o texto do PDF como uma única string, com separador
    de página explícito para permitir rastreabilidade posterior.
    """
    paginas = extrair_texto_paginas(caminho_pdf)
    partes = [f"\n\n=== [PÁGINA {p.numero + 1}] ===\n\n{p.texto}" for p in paginas]
    return "".join(partes)