"""
Módulo de chunking para editais de transformadores elétricos.

Expõe as estratégias principais e o tipo base Chunk.
"""

from edital_rag.chunking.strategies import (
    Chunk,
    ChunkStrategy,
    EstrategiaChunking,
    gerar_chunk_id,
)
from edital_rag.chunking.semantic import (
    ChunkingSemantico,
    chunking_semantico,
)
from edital_rag.chunking.table_chunker import (
    ChunkingTabelas,
    chunking_tabelas,
)

__all__ = [
    # strategies
    "Chunk",
    "ChunkStrategy",
    "EstrategiaChunking",
    "gerar_chunk_id",
    # semantic
    "ChunkingSemantico",
    "chunking_semantico",
    # table_chunker
    "ChunkingTabelas",
    "chunking_tabelas",
]