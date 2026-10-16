"""
Definições base para estratégias de chunking.

Fornece:
- Chunk: dataclass com texto + metadados (página, seção, tipo, fonte).
- ChunkStrategy: Protocol que toda estratégia deve implementar.
- EstrategiaChunking: Enum com os tipos disponíveis.
- gerar_chunk_id: hash determinístico baseado em conteúdo + metadados.
- contar_tokens: estimativa leve sem dependências externas.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Estimativa de tokens (sem tiktoken)
# ---------------------------------------------------------------------------

_RE_PALAVRA = re.compile(r"\w+", re.UNICODE)


def contar_tokens(texto: str) -> int:
    """
    Estimativa leve de tokens.

    Regra prática para português: ~1.3 tokens por palavra em BPE.
    É apenas uma heurística para controle de tamanho — não substitui
    o tokenizer real, mas evita carregar dependências pesadas.
    """
    palavras = _RE_PALAVRA.findall(texto)
    return int(len(palavras) * 1.3)


# ---------------------------------------------------------------------------
# Enum de estratégias
# ---------------------------------------------------------------------------

class EstrategiaChunking(str, Enum):
    SEMANTICO = "semantico"
    TABELAS = "tabelas"
    MISTO = "misto"


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    Unidade atômica indexável.

    Atributos:
        texto: conteúdo textual do chunk.
        fonte: identificador do PDF de origem (ex.: "edital_001/termo_referencia").
        pagina: número de página (1-indexed) ou None se abranger várias.
        secao: rótulo da seção de origem (ex.: "3.2 Requisitos Elétricos").
        tipo: "texto" | "tabela" | "preambulo".
        edital_id: identificador do edital (para filtragem).
        chunk_index: posição sequencial dentro do documento.
        num_tokens: estimativa de tokens.
        metadata: campo livre para informações adicionais.
    """

    texto: str
    fonte: str
    pagina: int | None = None
    secao: str = ""
    tipo: str = "texto"
    edital_id: str = ""
    chunk_index: int = 0
    num_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self) -> None:
        self.texto = self.texto.strip()
        if not self.num_tokens:
            self.num_tokens = contar_tokens(self.texto)
        if not self.chunk_id:
            self.chunk_id = gerar_chunk_id(self)

    def to_dict(self) -> dict[str, Any]:
        """Serialização para JSONL."""
        return {
            "chunk_id": self.chunk_id,
            "texto": self.texto,
            "fonte": self.fonte,
            "pagina": self.pagina,
            "secao": self.secao,
            "tipo": self.tipo,
            "edital_id": self.edital_id,
            "chunk_index": self.chunk_index,
            "num_tokens": self.num_tokens,
            "metadata": self.metadata,
        }


def gerar_chunk_id(chunk: Chunk) -> str:
    """
    Gera um ID determinístico baseado em conteúdo + metadados-chave.
    Isso garante que reingerir o mesmo PDF gere os mesmos IDs,
    permitindo deduplicação incremental.
    """
    base = f"{chunk.fonte}|{chunk.pagina}|{chunk.secao}|{chunk.tipo}|{chunk.texto}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Protocol de estratégia
# ---------------------------------------------------------------------------

@runtime_checkable
class ChunkStrategy(Protocol):
    """Interface para estratégias de chunking."""

    def chunk(
        self,
        *,
        fonte: str,
        edital_id: str = "",
        **kwargs: Any,
    ) -> list[Chunk]:
        ...