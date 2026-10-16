"""
Chunking de tabelas extraídas por pdf_tables.

Estratégia principal: "linha + cabeçalho" — cada linha de dados gera um
chunk no formato:
    "Tabela 3.1 (pág. 12) | Modelo: TR-500 | Potência: 500 kVA | ..."

Isso garante autossuficiência semântica: mesmo isolado, o chunk é
compreensível. Também suportamos modo "tabela inteira" para tabelas
pequenas, onde agrupar é mais eficiente.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_editais.chunking.strategies import (
    Chunk,
    EstrategiaChunking,
    contar_tokens,
    gerar_chunk_id,
)
from rag_editais.parsing.pdf_tables import TabelaExtraida, extrair_tabelas_pdf


@dataclass
class ChunkingTabelas:
    """
    Parâmetros do chunking de tabelas.

    modo:
        - "linha": uma linha por chunk, com cabeçalho prefixado (default).
        - "tabela": tabela inteira em um chunk (útil para tabelas pequenas).
        - "auto": usa "tabela" se a tabela inteira couber em max_tokens;
          caso contrário, "linha".

    max_tokens_tabela:
        Limite para o modo "tabela" ou "auto".
    incluir_cabecalho_linha:
        Prefixa o cabeçalho em cada linha no modo "linha".
    """

    modo: str = "auto"
    max_tokens_tabela: int = 500
    incluir_cabecalho_linha: bool = True
    estrategia: EstrategiaChunking = EstrategiaChunking.TABELAS

    # -----------------------------------------------------------------------
    # API
    # -----------------------------------------------------------------------

    def chunk(
        self,
        *,
        caminho_pdf: str | Path,
        fonte: str,
        edital_id: str = "",
        chunk_index_inicial: int = 0,
    ) -> list[Chunk]:
        tabelas = extrair_tabelas_pdf(caminho_pdf)
        chunks: list[Chunk] = []
        idx = chunk_index_inicial

        for tabela in tabelas:
            chunks_tabela = self._chunk_tabela(
                tabela=tabela,
                fonte=fonte,
                edital_id=edital_id,
                chunk_index_inicial=idx,
            )
            chunks.extend(chunks_tabela)
            idx += len(chunks_tabela)

        return chunks

    # -----------------------------------------------------------------------
    # Internos
    # -----------------------------------------------------------------------

    def _chunk_tabela(
        self,
        *,
        tabela: TabelaExtraida,
        fonte: str,
        edital_id: str,
        chunk_index_inicial: int,
    ) -> list[Chunk]:
        if not tabela.linhas:
            return []

        rotulo_tabela = self._rotulo(tabela)

        # Decide o modo efetivo
        modo = self.modo
        if modo == "auto":
            texto_completo = self._render_tabela_completa(tabela, rotulo_tabela)
            modo = (
                "tabela"
                if contar_tokens(texto_completo) <= self.max_tokens_tabela
                else "linha"
            )

        if modo == "tabela":
            return [
                self._chunk_tabela_inteira(
                    tabela=tabela,
                    rotulo=rotulo_tabela,
                    fonte=fonte,
                    edital_id=edital_id,
                    chunk_index=chunk_index_inicial,
                )
            ]

        return self._chunk_por_linha(
            tabela=tabela,
            rotulo=rotulo_tabela,
            fonte=fonte,
            edital_id=edital_id,
            chunk_index_inicial=chunk_index_inicial,
        )

    def _chunk_por_linha(
        self,
        *,
        tabela: TabelaExtraida,
        rotulo: str,
        fonte: str,
        edital_id: str,
        chunk_index_inicial: int,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        cabecalho = tabela.cabecalho or []

        for i, linha in enumerate(tabela.linhas):
            pares = []
            for j, valor in enumerate(linha):
                if not valor:
                    continue
                nome_col = cabecalho[j] if j < len(cabecalho) and cabecalho[j] else f"Coluna {j + 1}"
                pares.append(f"{nome_col}: {valor}")

            if not pares:
                continue

            prefixo = f"{rotulo} (pág. {tabela.pagina + 1})"
            texto = f"{prefixo} | " + " | ".join(pares)

            chunks.append(
                Chunk(
                    texto=texto,
                    fonte=fonte,
                    pagina=tabela.pagina + 1,
                    secao=rotulo,
                    tipo="tabela",
                    edital_id=edital_id,
                    chunk_index=chunk_index_inicial + i,
                    metadata={
                        "linha_tabela": i,
                        "colunas": cabecalho,
                        "tabela_indice_pagina": tabela.indice_na_pagina,
                        "modo": "linha",
                    },
                )
            )

        return chunks

    def _chunk_tabela_inteira(
        self,
        *,
        tabela: TabelaExtraida,
        rotulo: str,
        fonte: str,
        edital_id: str,
        chunk_index: int,
    ) -> Chunk:
        texto = self._render_tabela_completa(tabela, rotulo)
        return Chunk(
            texto=texto,
            fonte=fonte,
            pagina=tabela.pagina + 1,
            secao=rotulo,
            tipo="tabela",
            edital_id=edital_id,
            chunk_index=chunk_index,
            metadata={
                "tabela_indice_pagina": tabela.indice_na_pagina,
                "colunas": tabela.cabecalho,
                "modo": "tabela",
            },
        )

    @staticmethod
    def _render_tabela_completa(tabela: TabelaExtraida, rotulo: str) -> str:
        """Renderiza a tabela inteira em formato legível para o LLM."""
        linhas_render: list[str] = [f"{rotulo} (pág. {tabela.pagina + 1})"]
        if tabela.cabecalho:
            linhas_render.append(" | ".join(tabela.cabecalho))
            linhas_render.append("-" * 40)
        for linha in tabela.linhas:
            linhas_render.append(" | ".join(c for c in linha if c))
        return "\n".join(linhas_render)

    @staticmethod
    def _rotulo(tabela: TabelaExtraida) -> str:
        """
        Constrói um rótulo estável. Se a tabela tem cabeçalho, usamos a
        primeira coluna do cabeçalho como pista; caso contrário, geramos
        um identificador por posição.
        """
        if tabela.cabecalho and tabela.cabecalho[0]:
            return f"Tabela — {tabela.cabecalho[0]}"
        return f"Tabela (pág. {tabela.pagina + 1}, #{tabela.indice_na_pagina + 1})"


def chunking_tabelas(
    caminho_pdf: str | Path,
    *,
    fonte: str,
    edital_id: str = "",
    chunk_index_inicial: int = 0,
    **kwargs: Any,
) -> list[Chunk]:
    """Atalho funcional."""
    estrategia = ChunkingTabelas(**kwargs)
    return estrategia.chunk(
        caminho_pdf=caminho_pdf,
        fonte=fonte,
        edital_id=edital_id,
        chunk_index_inicial=chunk_index_inicial,
    )