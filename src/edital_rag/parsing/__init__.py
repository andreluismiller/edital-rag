"""
Módulo de parsing de PDFs de editais de transformadores elétricos.

Expõe as funções principais de extração e limpeza para uso pelo pipeline.
"""

from edital_rag.parsing.pdf_text import extrair_texto_paginas, extrair_texto_pdf
from edital_rag.parsing.pdf_tables import (
    TabelaExtraida,
    extrair_tabelas_pdf,
    tabela_para_texto,
)
from edital_rag.parsing.cleaner import (
    limpar_texto,
    remover_headers_footers,
    normalizar_espacos,
)
from edital_rag.parsing.layout import (
    SecaoDetectada,
    detectar_secoes,
    segmentar_por_secoes,
)

__all__ = [
    # pdf_text
    "extrair_texto_paginas",
    "extrair_texto_pdf",
    # pdf_tables
    "TabelaExtraida",
    "extrair_tabelas_pdf",
    "tabela_para_texto",
    # cleaner
    "limpar_texto",
    "remover_headers_footers",
    "normalizar_espacos",
    # layout
    "SecaoDetectada",
    "detectar_secoes",
    "segmentar_por_secoes",
]