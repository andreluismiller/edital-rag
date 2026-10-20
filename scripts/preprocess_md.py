#!/usr/bin/env python3
"""
Pre-processamento do Termo de Referencia (Markdown) para pipeline RAG.

Etapas:
  1. Remove placeholders de imagem (<!-- image -->), preservando metadados no log.
  2. Reconstroi tabelas fragmentadas (merge de fragmentos separados por ruido).
  3. Corrige quebras de linha dentro de celulas (linhas de continuacao).
  4. Normaliza titulos e numeracao (ex.: "## 5 - TITULO" -> "## 5. TITULO").
  5. Preserva valores monetarios, potencias, prazos e normas (validacao final).

Uso:
    python scripts/preprocess_md.py
    python scripts/preprocess_md.py --input ../data/processed/edital_aracruz_es.md \
                                    --output ../data/processed/edital_aracruz_es_clean.md
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


# ----------------------- Regex / padroes -----------------------

IMAGE_PLACEHOLDER_RE = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TABLE_LINE_RE = re.compile(r"^\s*\|")
SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")
HTML_COMMENT_RE = re.compile(r"^\s*<!--.*-->\s*$")

# Headings que aparecem como "ruido" entre fragmentos de tabela (cabecalhos de documento)
NOISE_HEADING_RE = re.compile(
    r"^#{1,3}\s+(ANEXO\b|TERMO\s+DE\s+REFER[EÊ]NCIA\b)",
    re.IGNORECASE,
)

# Numeracao de titulos: "5 - TITULO" -> "5. TITULO"
NUM_HEADING_DASH_RE = re.compile(r"^(\d+(?:\.\d+)*)\s*[-\u2013]\s+(.+)$")


# ----------------------- Helpers -----------------------

def is_table_line(line: str) -> bool:
    return bool(TABLE_LINE_RE.match(line))


def is_separator_row(cells: list[str]) -> bool:
    non_empty = [c for c in cells if c.strip()]
    return bool(non_empty) and all(SEPARATOR_CELL_RE.match(c.strip()) for c in non_empty)


def is_noise_line(line: str) -> bool:
    """Linha que pode separar fragmentos de uma mesma tabela (ruido de paginacao)."""
    s = line.strip()
    if s == "":
        return True
    if HTML_COMMENT_RE.match(s):
        return True
    if NOISE_HEADING_RE.match(s):
        return True
    return False


def parse_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def render_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def parse_table_block(lines: list[str], start: int) -> tuple[list[str], int]:
    """Retorna (linhas da tabela, indice do primeiro elemento apos a tabela)."""
    end = start
    rows: list[str] = []
    n = len(lines)
    while end < n and is_table_line(lines[end]):
        rows.append(lines[end])
        end += 1
    return rows, end


# ----------------------- Merge de tabelas -----------------------

def row_looks_like_continuation(cells: list[str], header_cells: list[str]) -> bool:
    """Uma linha parece continuacao da anterior se a maioria das celulas esta vazia
    e/ou a primeira celula (identificador) esta vazia enquanto o header tem identificador."""
    if not cells:
        return False
    non_empty = [c for c in cells if c.strip()]
    if not non_empty:
        return False
    if len(non_empty) * 2 <= len(cells):
        return True
    if header_cells and header_cells[0].strip() != "" and cells[0].strip() == "":
        return True
    return False


def can_merge(table_a: list[str], table_b: list[str]) -> bool:
    if not table_a or not table_b:
        return False
    header_a = parse_row(table_a[0])
    b_first = parse_row(table_b[0])
    if row_looks_like_continuation(b_first, header_a):
        return True
    if b_first == header_a:
        return True
    return False


def merge_two_tables(table_a: list[str], table_b: list[str]) -> list[str]:
    if not table_a:
        return list(table_b)
    if not table_b:
        return list(table_a)

    merged = list(table_a)
    header_a = parse_row(table_a[0])
    b_rows = list(table_b)
    b_first = parse_row(b_rows[0])

    # Caso 1: continuacao da ultima linha
    if row_looks_like_continuation(b_first, header_a):
        start = 1
        if len(b_rows) > 1 and is_separator_row(parse_row(b_rows[1])):
            start = 2
        if merged:
            last_cells = parse_row(merged[-1])
            while len(last_cells) < len(b_first):
                last_cells.append("")
            for ci, cv in enumerate(b_first):
                if not cv.strip():
                    continue
                if last_cells[ci].strip():
                    last_cells[ci] = last_cells[ci].rstrip() + " " + cv.strip()
                else:
                    last_cells[ci] = cv.strip()
            merged[-1] = render_row(last_cells)
        for k in range(start, len(b_rows)):
            cells = parse_row(b_rows[k])
            if not is_separator_row(cells):
                merged.append(b_rows[k])
        return merged

    # Caso 2: cabecalho identico (mesma tabela quebrada em duas)
    if b_first == header_a and len(b_rows) > 1 and is_separator_row(parse_row(b_rows[1])):
        for k in range(2, len(b_rows)):
            cells = parse_row(b_rows[k])
            if not is_separator_row(cells):
                merged.append(b_rows[k])
        return merged

    return merged + b_rows


# ----------------------- Etapas de pre-processamento -----------------------

def remove_image_placeholders(text: str) -> tuple[str, list[int]]:
    positions = [m.start() for m in IMAGE_PLACEHOLDER_RE.finditer(text)]
    return IMAGE_PLACEHOLDER_RE.sub("", text), positions


def normalize_headings(text: str) -> str:
    out: list[str] = []
    for line in text.split("\n"):
        m = HEADING_RE.match(line)
        if not m:
            out.append(line.rstrip())
            continue
        hashes, title = m.group(1), m.group(2).strip()
        dash = NUM_HEADING_DASH_RE.match(title)
        if dash:
            title = f"{dash.group(1)}. {dash.group(2).strip()}"
        out.append(f"{hashes} {title}".rstrip())
    return "\n".join(out)


def merge_table_fragments(lines: list[str]) -> tuple[list[str], list[str]]:
    """Funde fragmentos de tabela separados por ruido. Retorna (linhas, headings diferidos)."""
    result: list[str] = []
    deferred_headings: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if is_table_line(lines[i]):
            table_lines, end = parse_table_block(lines, i)
            while True:
                j = end
                noise: list[str] = []
                while j < n and is_noise_line(lines[j]):
                    noise.append(lines[j])
                    j += 1
                if j < n and is_table_line(lines[j]) and noise:
                    next_table, next_end = parse_table_block(lines, j)
                    if can_merge(table_lines, next_table):
                        for nl in noise:
                            s = nl.strip()
                            if HEADING_RE.match(s) and NOISE_HEADING_RE.match(s):
                                deferred_headings.append(nl.rstrip())
                        table_lines = merge_two_tables(table_lines, next_table)
                        end = next_end
                        continue
                break
            result.extend(table_lines)
            i = end
        else:
            result.append(lines[i])
            i += 1

    # Deduplicar headings preservando ordem
    if deferred_headings:
        seen: set[str] = set()
        unique: list[str] = []
        for h in deferred_headings:
            if h not in seen:
                seen.add(h)
                unique.append(h)
        result = unique + [""] + result

    return result, deferred_headings


# ----------------------- Verificacoes / relatorios -----------------------

def count_pattern(text: str, pattern: str) -> Counter:
    return Counter(m.group(0) for m in re.finditer(pattern, text))


def preservation_report(original: str, cleaned: str) -> list[str]:
    patterns = {
        "valores_monetarios": r"R\$\s*[\d\.]+,\d{2}",
        "normas": r"NBR[\s\-]*[\d\-]+|ES\.DT\.PDN\.[\d\.]+|IEC[\s\-]*\d+|ISO[\s\-]*\d+",
        "potencias_kVA": r"\d+(?:[.,]\d+)?\s*kVA",
        "prazos": r"\b\d+\s*(?:\([^)]+\)\s*)?(?:dias?|meses?|m[eê]s(?:es)?|anos?|horas?)\b",
        "tensoes_kV": r"\b\d+(?:[.,]\d+)?\s*kV\b",
    }
    lines: list[str] = []
    for name, pat in patterns.items():
        orig = count_pattern(original, pat)
        new = count_pattern(cleaned, pat)
        missing = orig - new
        if missing:
            lines.append(f"[AVISO] {name}: perdidos {dict(missing)}")
        else:
            lines.append(f"[OK]    {name}: {sum(orig.values())} preservados")
    return lines


def post_merge_anomalies(lines: list[str]) -> list[str]:
    """Reporta possiveis problemas pos-merge para revisao manual."""
    warnings: list[str] = []

    # 1. Colunas inconsistentes em tabelas
    i = 0
    while i < len(lines):
        if is_table_line(lines[i]):
            block: list[str] = []
            j = i
            while j < len(lines) and is_table_line(lines[j]):
                block.append(lines[j])
                j += 1
            if len(block) >= 2:
                ncols = len(parse_row(block[0]))
                for k, row in enumerate(block[1:], start=1):
                    cells = parse_row(row)
                    if not is_separator_row(cells) and len(cells) != ncols:
                        warnings.append(
                            f"  linha ~{i + k + 1}: {len(cells)} colunas (esperado {ncols}): {row[:90]}..."
                        )
            i = j
        else:
            i += 1

    # 2. Celula iniciada por "NBR " quando linha anterior termina em "conforme"
    for idx, row in enumerate(lines):
        if not is_table_line(row):
            continue
        cells = parse_row(row)
        if not cells:
            continue
        desc = cells[1] if len(cells) > 1 else ""
        if desc.startswith("NBR "):
            prev = lines[idx - 1].rstrip() if idx > 0 else ""
            if prev.endswith("conforme |") or prev.endswith("conforme|"):
                warnings.append(
                    f"  linha {idx + 1}: possivel 'NBR' deslocada do item anterior "
                    f"(descricao comeca com 'NBR')"
                )

    # 3. Blocos duplicados obvios (mesmo texto em niveis diferentes)
    seen_sections: dict[str, list[str]] = {}
    for row in lines:
        m = HEADING_RE.match(row)
        if not m:
            continue
        title = m.group(2).strip().lower()
        if title in {"enrolamentos e isolamentos", "enrolamentos e isolamento"}:
            seen_sections.setdefault(title, []).append(row)
    for title, rows in seen_sections.items():
        if len(rows) > 1:
            warnings.append(
                f"  secao '{title}' aparece {len(rows)} vezes; verificar duplicidade"
            )

    return warnings


def report_ensaios(lines: list[str]) -> list[str]:
    """Sinaliza que as listas de ensaios foram mantidas (nao convertidas em tabela)."""
    msgs: list[str] = []
    for label in ("ensaios de tipo", "ensaios de rotina"):
        if any(label in ln.lower() for ln in lines):
            msgs.append(f"[INFO] Lista '{label}' mantida como bullet-list (nao convertida em tabela).")
    return msgs


# ----------------------- Main -----------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-processa o Termo de Referencia em Markdown para pipeline RAG."
    )
    default_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--input",
        default=str(default_root / "data" / "processed" / "edital_aracruz_es.md"),
        help="Caminho do markdown de entrada",
    )
    parser.add_argument(
        "--output",
        default=str(default_root / "data" / "processed" / "edital_aracruz_es_clean.md"),
        help="Caminho do markdown de saida",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise SystemExit(f"[ERRO] Arquivo nao encontrado: {input_path}")

    text_original = input_path.read_text(encoding="utf-8")
    text = text_original

    # 0. Normalizar finais de linha
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1. Remover placeholders de imagem
    text, img_positions = remove_image_placeholders(text)
    print(f"[INFO] Placeholders de imagem removidos: {len(img_positions)}")

    # 2. Normalizar headings
    text = normalize_headings(text)

    # 3. Reconstruir tabelas fragmentadas
    lines = text.split("\n")
    lines, deferred_headings = merge_table_fragments(lines)
    text = "\n".join(lines)

    if deferred_headings:
        print(f"[INFO] Headings de documento reposicionados no topo: {deferred_headings}")

    # 4. Colapsar linhas em branco excessivas
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Garantir newline final
    text = text.rstrip() + "\n"

    # 6. Relatorios
    print("\n[INFO] Verificacao de preservacao:")
    for line in preservation_report(text_original, text):
        print("   ", line)

    print("\n[INFO] Anomalias pos-merge (revisao manual recomendada):")
    anomalies = post_merge_anomalies(text.split("\n"))
    if anomalies:
        for a in anomalies:
            print(a)
    else:
        print("    Nenhuma.")

    print("\n[INFO] Listas estruturadas:")
    for m in report_ensaios(text.split("\n")):
        print("   ", m)

    # 7. Gravar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"\n[OK] Arquivo gerado: {output_path}")
    print(f"[OK] Tamanho: {len(text)} caracteres, {text.count(chr(10))} linhas")


if __name__ == "__main__":
    main()