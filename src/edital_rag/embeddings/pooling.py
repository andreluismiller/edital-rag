"""
Estratégias de pooling para extrair um único vetor de uma sequência
de hidden states do transformer.

Suporta:
- CLS: usa o token [CLS] (posição 0) — comum em BERT e alguns BGE.
- MEAN: média dos tokens reais, ignorando padding via attention_mask.
- WEIGHTED_MEAN: média ponderada pelas posições (aproximação barata do
  que o sentence-transformers faz).

Também fornece normalização L2, indispensável para similaridade de
cosseno via produto interno no FAISS (IndexFlatIP).
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class EstrategiaPooling(str, Enum):
    CLS = "cls"
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"


def _mascarar(
    hidden: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    """
    Aplica a máscara de atenção aos hidden states.

    Args:
        hidden: shape (batch, seq_len, dim).
        attention_mask: shape (batch, seq_len), valores 0/1.

    Returns:
        Hidden states com posições de padding zeradas.
    """
    # (batch, seq_len, 1) para broadcast
    mask_expandida = attention_mask[..., None].astype(hidden.dtype)
    return hidden * mask_expandida


def pooling_cls(
    hidden: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    """
    Usa o vetor do token [CLS] (posição 0) como representação da sequência.
    Retorna shape (batch, dim).
    """
    return hidden[:, 0, :]


def pooling_mean(
    hidden: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    """
    Média dos tokens reais (não-padding). Retorna shape (batch, dim).
    """
    mascarado = _mascarar(hidden, attention_mask)
    soma = mascarado.sum(axis=1)  # (batch, dim)
    contagem = attention_mask.sum(axis=1, keepdims=True).astype(hidden.dtype)
    # Evita divisão por zero (não deve ocorrer, pois há sempre ao menos CLS/SEP)
    contagem = np.clip(contagem, a_min=1.0, a_max=None)
    return soma / contagem


def pooling_weighted_mean(
    hidden: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    """
    Média ponderada: tokens mais próximos do início têm peso maior
    (decaimento linear). Aproximação barata do que o sentence-transformers
    faz com pesos aprendidos.

    Retorna shape (batch, dim).
    """
    batch, seq_len, dim = hidden.shape
    # Pesos lineares: seq_len, seq_len-1, ..., 1
    pesos = np.arange(seq_len, 0, -1, dtype=hidden.dtype)  # (seq_len,)
    pesos = pesos[None, :] * attention_mask.astype(hidden.dtype)  # (batch, seq_len)

    pesos_expandidos = pesos[..., None]  # (batch, seq_len, 1)
    numerador = (hidden * pesos_expandidos).sum(axis=1)  # (batch, dim)
    denominador = pesos.sum(axis=1, keepdims=True)  # (batch, 1)
    denominador = np.clip(denominador, a_min=1e-9, a_max=None)
    return numerador / denominador


_MAPA_POOLING = {
    EstrategiaPooling.CLS: pooling_cls,
    EstrategiaPooling.MEAN: pooling_mean,
    EstrategiaPooling.WEIGHTED_MEAN: pooling_weighted_mean,
}


def aplicar_pooling(
    hidden: np.ndarray,
    attention_mask: np.ndarray,
    estrategia: EstrategiaPooling | str = EstrategiaPooling.CLS,
) -> np.ndarray:
    """
    Dispatcher de pooling.

    Args:
        hidden: (batch, seq_len, dim).
        attention_mask: (batch, seq_len).
        estrategia: enum ou string ("cls", "mean", "weighted_mean").

    Returns:
        Array (batch, dim) com o vetor pooled por amostra.
    """
    if isinstance(estrategia, str):
        estrategia = EstrategiaPooling(estrategia)
    return _MAPA_POOLING[estrategia](hidden, attention_mask)


def normalizar_l2(vetores: np.ndarray) -> np.ndarray:
    """
    Normaliza cada linha para norma L2 = 1. Isso faz o produto interno
    equivaler à similaridade de cosseno — pré-requisito para IndexFlatIP.
    """
    normas = np.linalg.norm(vetores, axis=1, keepdims=True)
    normas = np.clip(normas, a_min=1e-12, a_max=None)
    return vetores / normas