"""
Limpeza e normalização de texto extraído de PDFs.

Cobre problemas típicos de editais:
- hifenização quebrada ao final de linha ("transfor-\nmador")
- linhas quebradas no meio de frases
- cabeçalhos e rodapés repetidos página a página
- espaços múltiplos, tabs e caracteres invisíveis
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# ---------------------------------------------------------------------------
# Normalizações básicas
# ---------------------------------------------------------------------------

_RE_ESPACOS = re.compile(r"[ \t\u00a0]+")
_RE_LINHAS_VAZIAS = re.compile(r"\n{3,}")
_RE_HIFEN_QUEBRA = re.compile(r"(\w)-\n(\w)")
_RE_QUEBRA_MEIO_FRASE = re.compile(r"([a-záéíóúâêôãõç,])\n([a-záéíóúâêôãõç])")


def normalizar_espacos(texto: str) -> str:
    """Colapsa espaços/tabs múltiplos e remove espaços nas pontas de cada linha."""
    linhas = [linha.rstrip() for linha in texto.splitlines()]
    texto = "\n".join(linhas)
    texto = _RE_ESPACOS.sub(" ", texto)
    texto = _RE_LINHAS_VAZIAS.sub("\n\n", texto)
    return texto.strip()


def corrigir_hifenizacao(texto: str) -> str:
    """Une palavras quebradas com hífen no final da linha."""
    return _RE_HIFEN_QUEBRA.sub(r"\1\2", texto)


def unir_quebras_de_frase(texto: str) -> str:
    """
    Junta linhas quando a quebra ocorreu no meio de uma frase
    (linha anterior termina com letra minúscula ou vírgula, e a próxima
    começa com letra minúscula).
    """
    return _RE_QUEBRA_MEIO_FRASE.sub(r"\1 \2", texto)


def remover_caracteres_controle(texto: str) -> str:
    """Remove caracteres Unicode de controle, exceto \n e \t."""
    return "".join(
        c for c in texto
        if unicodedata.category(c) != "Cc" or c in "\n\t"
    )


def limpar_texto(
    texto: str,
    *,
    corrigir_hifen: bool = True,
    unir_linhas: bool = True,
) -> str:
    """
    Pipeline completo de limpeza para um bloco de texto.

    Ordem importa: hifenização antes de unir linhas, para não perder
    palavras compostas.
    """
    texto = remover_caracteres_controle(texto)
    if corrigir_hifen:
        texto = corrigir_hifenizacao(texto)
    if unir_linhas:
        texto = unir_quebras_de_frase(texto)
    texto = normalizar_espacos(texto)
    return texto


# ---------------------------------------------------------------------------
# Remoção de cabeçalhos/rodapés repetidos
# ---------------------------------------------------------------------------

def _normalizar_linha_para_hash(linha: str) -> str:
    """Normaliza uma linha para comparação entre páginas."""
    linha = re.sub(r"\d+", "#", linha.strip())
    return linha.lower()


def detectar_headers_footers(
    paginas: list[str],
    *,
    max_linhas_analisadas: int = 4,
    min_repeticoes: float = 0.6,
) -> tuple[set[str], set[str]]:
    """
    Detecta linhas que aparecem no topo/rodapé de >= min_repeticoes das páginas.

    Returns:
        (headers, footers) como conjuntos de linhas normalizadas.
    """
    contador_topo: Counter[str] = Counter()
    contador_rodape: Counter[str] = Counter()

    for texto in paginas:
        linhas = [l for l in texto.splitlines() if l.strip()]
        if not linhas:
            continue
        topo = linhas[:max_linhas_analisadas]
        rodape = linhas[-max_linhas_analisadas:]
        for linha in topo:
            contador_topo[_normalizar_linha_para_hash(linha)] += 1
        for linha in rodape:
            contador_rodape[_normalizar_linha_para_hash(linha)] += 1

    total_paginas = max(len(paginas), 1)
    limite = min_repeticoes * total_paginas

    headers = {k for k, v in contador_topo.items() if v >= limite}
    footers = {k for k, v in contador_rodape.items() if v >= limite}
    return headers, footers


def remover_headers_footers(
    paginas: list[str],
    *,
    max_linhas_analisadas: int = 4,
    min_repeticoes: float = 0.6,
) -> list[str]:
    """
    Remove cabeçalhos e rodapés repetidos de uma lista de páginas.
    """
    headers, footers = detectar_headers_footers(
        paginas,
        max_linhas_analisadas=max_linhas_analisadas,
        min_repeticoes=min_repeticoes,
    )
    if not headers and not footers:
        return paginas

    resultado: list[str] = []
    for texto in paginas:
        linhas = texto.splitlines()
        filtradas: list[str] = []
        for i, linha in enumerate(linhas):
            chave = _normalizar_linha_para_hash(linha)
            no_topo = i < max_linhas_analisadas and chave in headers
            no_rodape = i >= len(linhas) - max_linhas_analisadas and chave in footers
            if not (no_topo or no_rodape):
                filtradas.append(linha)
        resultado.append("\n".join(filtradas))

    return resultado