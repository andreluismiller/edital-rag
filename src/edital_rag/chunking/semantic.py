"""
Chunking semântico de texto corrido.

Estratégia:
1. Segmenta o texto em seções via layout.detectar_secoes/segmentar_por_secoes.
2. Para cada seção, acumula parágrafos até atingir max_tokens.
3. Quando estoura, emite o chunk e começa um novo com overlap de
   parágrafos finais (contexto de continuidade).
4. Se um parágrafo sozinho excede max_tokens, ele é cortado em sentenças.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from edital_rag.chunking.strategies import (
    Chunk,
    EstrategiaChunking,
    contar_tokens,
)
from edital_rag.parsing.cleaner import limpar_texto, remover_headers_footers
from edital_rag.parsing.layout import SecaoDetectada, segmentar_por_secoes
from edital_rag.parsing.pdf_text import extrair_texto_paginas


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

@dataclass
class ChunkingSemantico:
    """
    Parâmetros do chunking semântico.

    max_tokens: tamanho-alvo do chunk (padrão 400).
    min_tokens: chunks menores que isso são fundidos com o próximo (padrão 60).
    overlap_tokens: sobreposição entre chunks adjacentes (padrão 60).
    incluir_preambulo: mantém o texto antes da primeira seção numerada.
    limpar: aplica pipeline de limpeza do cleaner.
    """

    max_tokens: int = 400
    min_tokens: int = 60
    overlap_tokens: int = 60
    incluir_preambulo: bool = True
    limpar: bool = True

    estrategia: EstrategiaChunking = EstrategiaChunking.SEMANTICO

    # -----------------------------------------------------------------------
    # API principal
    # -----------------------------------------------------------------------

    def chunk(
        self,
        *,
        caminho_pdf: str | Path,
        fonte: str,
        edital_id: str = "",
    ) -> list[Chunk]:
        """
        Executa o pipeline completo de chunking sobre um PDF.
        """
        paginas = extrair_texto_paginas(caminho_pdf)

        # 1. Limpeza por página + remoção de headers/footers repetidos
        textos_pagina = [p.texto for p in paginas]
        if self.limpar:
            textos_pagina = remover_headers_footers(textos_pagina)
            textos_pagina = [limpar_texto(t) for t in textos_pagina]

        # 2. Cada página é segmentada em seções independentemente.
        #    Seções que atravessam páginas serão tratadas como chunks
        #    separados — é uma limitação aceitável em editais, onde
        #    cada seção tende a caber em uma página.
        chunks: list[Chunk] = []
        idx = 0

        for numero_pagina, (pagina_obj, texto) in enumerate(
            zip(paginas, textos_pagina), start=1
        ):
            if not texto.strip():
                continue

            secoes = segmentar_por_secoes(
                texto, incluir_preambulo=self.incluir_preambulo
            )

            for secao in secoes:
                chunks_secao = self._chunk_secao(
                    secao=secao,
                    fonte=fonte,
                    edital_id=edital_id,
                    pagina=numero_pagina,
                    chunk_index_inicial=idx,
                )
                chunks.extend(chunks_secao)
                idx += len(chunks_secao)

        return chunks

    # -----------------------------------------------------------------------
    # Internos
    # -----------------------------------------------------------------------

    def _chunk_secao(
        self,
        *,
        secao: SecaoDetectada,
        fonte: str,
        edital_id: str,
        pagina: int,
        chunk_index_inicial: int,
    ) -> list[Chunk]:
        """Divide o texto de uma seção em chunks respeitando max_tokens."""
        paragrafos = self._dividir_paragrafos(secao.texto)
        if not paragrafos:
            return []

        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_tokens = 0

        for paragrafo in paragrafos:
            tokens_par = contar_tokens(paragrafo)

            # Parágrafo gigante: subdividir em sentenças
            if tokens_par > self.max_tokens:
                # Fecha o buffer atual antes de tratar o parágrafo grande
                if buffer:
                    chunks.append(
                        self._emitir(
                            buffer,
                            secao=secao,
                            fonte=fonte,
                            edital_id=edital_id,
                            pagina=pagina,
                            chunk_index=chunk_index_inicial + len(chunks),
                        )
                    )
                    buffer, buffer_tokens = self._overlap(buffer)

                for sub in self._dividir_paragrafo_grande(paragrafo):
                    chunks.append(
                        self._emitir(
                            [sub],
                            secao=secao,
                            fonte=fonte,
                            edital_id=edital_id,
                            pagina=pagina,
                            chunk_index=chunk_index_inicial + len(chunks),
                        )
                    )
                continue

            # Cabe no buffer?
            if buffer_tokens + tokens_par <= self.max_tokens:
                buffer.append(paragrafo)
                buffer_tokens += tokens_par
            else:
                # Emite o buffer e inicia novo com overlap
                if buffer:
                    chunks.append(
                        self._emitir(
                            buffer,
                            secao=secao,
                            fonte=fonte,
                            edital_id=edital_id,
                            pagina=pagina,
                            chunk_index=chunk_index_inicial + len(chunks),
                        )
                    )
                    buffer, buffer_tokens = self._overlap(buffer)
                buffer.append(paragrafo)
                buffer_tokens += tokens_par

        # Fecha o último buffer
        if buffer and buffer_tokens >= self.min_tokens:
            chunks.append(
                self._emitir(
                    buffer,
                    secao=secao,
                    fonte=fonte,
                    edital_id=edital_id,
                    pagina=pagina,
                    chunk_index=chunk_index_inicial + len(chunks),
                )
            )
        elif buffer and chunks:
            # Buffer final muito pequeno: funde ao último chunk
            ultimo = chunks[-1]
            ultimo.texto = ultimo.texto + "\n\n" + "\n\n".join(buffer)
            ultimo.num_tokens = contar_tokens(ultimo.texto)
            from edital_rag.chunking.strategies import gerar_chunk_id
            ultimo.chunk_id = gerar_chunk_id(ultimo)
        elif buffer:
            # Caso raro: seção muito curta sem chunks anteriores
            chunks.append(
                self._emitir(
                    buffer,
                    secao=secao,
                    fonte=fonte,
                    edital_id=edital_id,
                    pagina=pagina,
                    chunk_index=chunk_index_inicial,
                )
            )

        return chunks

    def _emitir(
        self,
        paragrafos: list[str],
        *,
        secao: SecaoDetectada,
        fonte: str,
        edital_id: str,
        pagina: int,
        chunk_index: int,
    ) -> Chunk:
        """Constrói um Chunk a partir de um buffer de parágrafos."""
        texto = "\n\n".join(p.strip() for p in paragrafos if p.strip())
        return Chunk(
            texto=texto,
            fonte=fonte,
            pagina=pagina,
            secao=secao.rotulo,
            tipo="preambulo" if secao.tipo == "preambulo" else "texto",
            edital_id=edital_id,
            chunk_index=chunk_index,
            metadata={
                "tipo_secao": secao.tipo,
                "numero_secao": secao.numero,
                "titulo_secao": secao.titulo,
            },
        )

    def _overlap(self, paragrafos: list[str]) -> tuple[list[str], int]:
        """
        Extrai os parágrafos finais que cabem em overlap_tokens.
        Retorna (paragrafos_de_overlap, tokens_totais).
        """
        if self.overlap_tokens <= 0 or not paragrafos:
            return [], 0

        selecionados: list[str] = []
        total = 0
        for p in reversed(paragrafos):
            t = contar_tokens(p)
            if total + t > self.overlap_tokens and selecionados:
                break
            selecionados.insert(0, p)
            total += t
        return selecionados, total

    @staticmethod
    def _dividir_paragrafos(texto: str) -> list[str]:
        """Divide o texto em parágrafos por linhas em branco."""
        bruto = re.split(r"\n\s*\n", texto)
        return [p.strip() for p in bruto if p.strip()]

    def _dividir_paragrafo_grande(self, paragrafo: str) -> list[str]:
        """
        Um parágrafo que excede max_tokens é dividido em sentenças
        e reagrupado em blocos de tamanho controlado.
        """
        sentencas = re.split(r"(?<=[.!?])\s+", paragrafo)
        blocos: list[str] = []
        buffer: list[str] = []
        tokens = 0
        for s in sentencas:
            t = contar_tokens(s)
            if tokens + t > self.max_tokens and buffer:
                blocos.append(" ".join(buffer))
                buffer, tokens = [], 0
            buffer.append(s)
            tokens += t
        if buffer:
            blocos.append(" ".join(buffer))
        return blocos


def chunking_semantico(
    caminho_pdf: str | Path,
    *,
    fonte: str,
    edital_id: str = "",
    **kwargs: Any,
) -> list[Chunk]:
    """Atalho funcional para uso direto."""
    estrategia = ChunkingSemantico(**kwargs)
    return estrategia.chunk(caminho_pdf=caminho_pdf, fonte=fonte, edital_id=edital_id)