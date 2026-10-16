"""
Detecção de seções e estrutura de layout em editais.

Reconhece padrões comuns em termos de referência:
- Numeração hierárquica: "1.", "1.1", "1.1.1", "2.3.4"
- Anexos: "ANEXO I", "ANEXO A", "Anexo II"
- Tabelas: "Tabela 1", "TABELA 2.1"
- Seções nomeadas: "REQUISITOS TÉCNICOS", "ESPECIFICAÇÕES"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Padrões
# ---------------------------------------------------------------------------

# Ex.: "1.", "1.1", "1.1.1", "2.3.4.5" — no início da linha
_RE_NUMERACAO = re.compile(
    r"^(?P<num>\d+(?:\.\d+)*)\.?\s+(?P<titulo>.{3,120})$",
    re.MULTILINE,
)

# Ex.: "ANEXO I", "ANEXO A", "Anexo II — Especificações"
_RE_ANEXO = re.compile(
    r"^(?P<rotulo>ANEXO\s+(?:[IVXLCDM]+|[A-Z])(?:\s*[-—:]\s*.*)?)$",
    re.MULTILINE | re.IGNORECASE,
)

# Ex.: "Tabela 1", "TABELA 2.1 — Características"
_RE_TABELA = re.compile(
    r"^(?P<rotulo>TABELA\s+\d+(?:\.\d+)*)(?:\s*[-—:]\s*(?P<titulo>.*))?$",
    re.MULTILINE | re.IGNORECASE,
)

# Seções em CAIXA ALTA com pelo menos 2 palavras (heurística)
_RE_SECAO_MAIUSCULA = re.compile(
    r"^(?P<titulo>[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-ZÁÉÍÓÚÂÊÔÃÕÇ\s\-]{5,80})$",
    re.MULTILINE,
)


@dataclass
class SecaoDetectada:
    """Representa uma seção do documento com sua posição no texto."""

    tipo: str  # "numerada" | "anexo" | "tabela" | "maiuscula"
    numero: str  # "1.2.3" ou "I" ou "1"
    titulo: str
    inicio: int  # posição no texto original (índice do match)
    fim: int = 0  # preenchido por segmentar_por_secoes
    texto: str = ""  # conteúdo da seção

    @property
    def rotulo(self) -> str:
        """Rótulo legível para uso em metadados de chunk."""
        if self.tipo == "anexo":
            return f"ANEXO {self.numero}"
        if self.tipo == "tabela":
            return f"Tabela {self.numero}"
        if self.tipo == "numerada":
            return f"{self.numero} {self.titulo}".strip()
        return self.titulo


def detectar_secoes(texto: str) -> list[SecaoDetectada]:
    """
    Detecta todas as seções candidatas no texto, em ordem de aparição.

    A ordem de prioridade entre padrões é resolvida por posição: se dois
    padrões casam no mesmo offset, escolhemos o mais específico
    (anexo > tabela > numerada > maiúscula).
    """
    candidatas: dict[int, SecaoDetectada] = {}

    for match in _RE_NUMERACAO.finditer(texto):
        num = match.group("num")
        titulo = match.group("titulo").strip()
        # Filtra falsos positivos: linhas de tabela costumam ter muitos pipes
        if titulo.count("|") >= 2:
            continue
        candidatas.setdefault(
            match.start(),
            SecaoDetectada(
                tipo="numerada",
                numero=num,
                titulo=titulo,
                inicio=match.start(),
            ),
        )

    for match in _RE_ANEXO.finditer(texto):
        rotulo = match.group("rotulo").strip()
        # Extrai o identificador romano/letra
        partes = rotulo.split()
        numero = partes[1] if len(partes) > 1 else ""
        candidatas[match.start()] = SecaoDetectada(
            tipo="anexo",
            numero=numero,
            titulo=rotulo,
            inicio=match.start(),
        )

    for match in _RE_TABELA.finditer(texto):
        rotulo = match.group("rotulo").strip()
        numero = rotulo.split()[-1]
        titulo = (match.group("titulo") or "").strip()
        candidatas.setdefault(
            match.start(),
            SecaoDetectada(
                tipo="tabela",
                numero=numero,
                titulo=titulo or rotulo,
                inicio=match.start(),
            ),
        )

    for match in _RE_SECAO_MAIUSCULA.finditer(texto):
        titulo = match.group("titulo").strip()
        # Evita capturar linhas que já são anexos/tabelas
        if re.match(r"^ANEXO\s|^TABELA\s", titulo, re.IGNORECASE):
            continue
        candidatas.setdefault(
            match.start(),
            SecaoDetectada(
                tipo="maiuscula",
                numero="",
                titulo=titulo,
                inicio=match.start(),
            ),
        )

    return sorted(candidatas.values(), key=lambda s: s.inicio)


def segmentar_por_secoes(
    texto: str,
    *,
    incluir_preambulo: bool = True,
) -> list[SecaoDetectada]:
    """
    Segmenta o texto em seções com base nas marcações detectadas.

    Args:
        texto: conteúdo completo (uma página ou todo o documento).
        incluir_preambulo: se True, o texto antes da primeira seção é
            retornado como uma seção do tipo "preambulo".

    Returns:
        Lista de SecaoDetectada com o campo `texto` preenchido.
    """
    secoes = detectar_secoes(texto)
    if not secoes:
        if incluir_preambulo and texto.strip():
            return [
                SecaoDetectada(
                    tipo="preambulo",
                    numero="",
                    titulo="Preâmbulo",
                    inicio=0,
                    fim=len(texto),
                    texto=texto.strip(),
                )
            ]
        return []

    resultado: list[SecaoDetectada] = []

    if incluir_preambulo and secoes[0].inicio > 0:
        preambulo = texto[: secoes[0].inicio].strip()
        if preambulo:
            resultado.append(
                SecaoDetectada(
                    tipo="preambulo",
                    numero="",
                    titulo="Preâmbulo",
                    inicio=0,
                    fim=secoes[0].inicio,
                    texto=preambulo,
                )
            )

    for i, secao in enumerate(secoes):
        fim = secoes[i + 1].inicio if i + 1 < len(secoes) else len(texto)
        secao.fim = fim
        secao.texto = texto[secao.inicio : fim].strip()
        if secao.texto:
            resultado.append(secao)

    return resultado