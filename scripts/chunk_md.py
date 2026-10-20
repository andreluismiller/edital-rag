#!/usr/bin/env python3
"""
Etapa 2 do pipeline RAG: parse hierarquico + chunking do Termo de Referencia.

Entrada: data/processed/edital_aracruz_es_clean.md
Saida:   data/processed/edital_aracruz_es_chunks.jsonl

Cada linha e um chunk no schema:
{
  "chunk_id", "doc_id", "doc_title", "section_path", "item_number",
  "chunk_type", "parent_id", "child_ids", "table_id", "row_range",
  "entities", "normas_citadas", "valores", "referencias_resolvidas",
  "source_anchor", "texto"
}
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------- Regex ----------------------------

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
NUMERIC_TITLE_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$")
CLAUSE_LINE_RE = re.compile(r"^(?:-\s*)?(\d+(?:\.\d+)+)\.?\s+(.+)$")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*]\s+|[a-zA-Z]\)\s+|[IVX]+[.)]\s+|\d+[.)]\s+)(.+)$")
TABLE_LINE_RE = re.compile(r"^\s*\|")

# Normas / referencias externas
NBR_RE = re.compile(r"NBR[\s\-]*\d+(?:[\.\-]\d+)*")
IEC_RE = re.compile(r"\bIEC[\s\-]*\d+(?:[\.\-]\d+)*")
ISO_RE = re.compile(r"\bISO[\s\-]*\d+(?:[\.\-]\d+)*")
EDP_NORM_RE = re.compile(r"ES\.DT\.PDN\.[\d\.]+")
ANP_RE = re.compile(
    r"(?:Resolu[çc][ãa]o|Regulamento\s+T[ée]cnico)\s+(?:ANP\s+)?N[ºo°]\s*\d+/\d+"
)

# Valores
MONEY_RE = re.compile(r"R\$\s*[\d\.]+,\d{2}")
KVA_RE = re.compile(r"\d+(?:[.,]\d+)?\s*kVA", re.IGNORECASE)
KV_RE = re.compile(r"\d+(?:[.,]\d+)?\s*kV\b", re.IGNORECASE)
PERCENT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%")
TIME_RE = re.compile(
    r"\b\d+\s*(?:\([^)]+\)\s*)?(?:dias?|meses?|m[eê]s(?:es)?|anos?|horas?)\b"
)

# Referencias internas (a resolver contra section_map)
INT_REF_RE = re.compile(
    r"(?:conforme|ver|vide|consultar|de\s+acordo\s+com)\s+(?:o\s+)?(?:item|se[çc][ãa]o|cl[áa]usula)\s+(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

# ---------------------------- Dominio ----------------------------

DOMAIN_TERMS = [
    "óleo isolante", "óleo vegetal", "óleo mineral",
    "comutador de tap", "comutador", "tap",
    "bucha", "buchas", "tanque", "tampa",
    "radiador", "radiadores", "núcleo", "enrolamento", "enrolamentos",
    "para-raios", "alça de suspensão", "placa de identificação",
    "impedância", "curto-circuito", "isolamento", "estanqueidade",
    "corrosão", "pintura", "zincagem",
    "garantia", "reajuste", "homologação",
    "recebimento provisório", "recebimento definitivo",
    "nota fiscal", "fatura", "penalidade", "multa", "sanção",
]

STOP_ACRONYMS = {"TR", "ES", "BR", "DE", "DO", "DA", "OK", "PDF", "AF", "IN"}

# ---------------------------- Dataclasses ----------------------------

@dataclass
class Block:
    type: str  # "paragraph" | "table" | "list" | "numbered_clause"
    raw: str
    meta: dict = field(default_factory=dict)


@dataclass
class Section:
    number: str
    title: str
    raw_content: str
    start_line: int = 0
    end_line: int = 0
    parent: Optional["Section"] = None
    children: list["Section"] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)


# ---------------------------- Parsing ----------------------------

def split_blocks(raw: str) -> list[Block]:
    """Divide o conteudo da secao em blocos (paragrafo/tabela/lista/clausula)."""
    lines = raw.split("\n")
    blocks: list[Block] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        if TABLE_LINE_RE.match(line):
            tbl_lines = []
            while i < n and TABLE_LINE_RE.match(lines[i]):
                tbl_lines.append(lines[i])
                i += 1
            blocks.append(Block("table", "\n".join(tbl_lines),
                                {"lines": tbl_lines}))
            continue

        if LIST_ITEM_RE.match(line):
            list_lines = [line]
            i += 1
            while i < n and lines[i].strip() and not TABLE_LINE_RE.match(lines[i]) \
                    and not HEADING_RE.match(lines[i]):
                if LIST_ITEM_RE.match(lines[i]):
                    list_lines.append(lines[i])
                elif lines[i][0] in " \t":
                    list_lines[-1] += " " + lines[i].strip()
                else:
                    break
                i += 1
            blocks.append(Block("list", "\n".join(list_lines),
                                {"lines": list_lines}))
            continue

        para_lines = [line]
        i += 1
        while i < n and lines[i].strip() and not TABLE_LINE_RE.match(lines[i]) \
                and not LIST_ITEM_RE.match(lines[i]) and not HEADING_RE.match(lines[i]):
            para_lines.append(lines[i])
            i += 1
        para = "\n".join(para_lines).strip()
        m = CLAUSE_LINE_RE.match(para)
        if m:
            blocks.append(Block("numbered_clause", para, {"number": m.group(1)}))
        else:
            blocks.append(Block("paragraph", para))
    return blocks


def parse_document(text: str) -> tuple[list[Section], dict[str, Section], str]:
    """Parse do markdown limpo em arvore de secoes + prefixo do documento."""
    lines = text.split("\n")
    headings: list[dict] = []
    doc_prefix_parts: list[str] = []

    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if not m or len(m.group(1)) != 2:
            continue
        title = m.group(2).strip()
        num_match = NUMERIC_TITLE_RE.match(title)
        if num_match:
            headings.append({"line": i,
                             "number": num_match.group(1),
                             "title": num_match.group(2).strip()})
        else:
            doc_prefix_parts.append(title)

    sections: list[Section] = []
    for idx, h in enumerate(headings):
        start = h["line"] + 1
        end = headings[idx + 1]["line"] if idx + 1 < len(headings) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        sec = Section(
            number=h["number"], title=h["title"], raw_content=content,
            start_line=start, end_line=end,
        )
        sec.blocks = split_blocks(content)
        sections.append(sec)

    section_map = {s.number: s for s in sections}
    roots: list[Section] = []
    for s in sections:
        parts = s.number.split(".")
        if len(parts) > 1:
            parent = section_map.get(".".join(parts[:-1]))
            if parent:
                s.parent = parent
                parent.children.append(s)
                continue
        roots.append(s)

    # Prefixo do documento (primeiro heading nao numerico)
    doc_prefix = doc_prefix_parts[0] if doc_prefix_parts else ""
    return roots, section_map, doc_prefix


# ---------------------------- Section path ----------------------------

def build_section_path(section: Section, doc_prefix: str,
                       compact: bool = True) -> str:
    """Constroi 'ANEXO I > Termo de Referência > 5. DEFINIÇÕES > 5.2.3.10 Líquido Isolante'."""
    ancestors: list[Section] = []
    cur = section.parent
    while cur is not None:
        ancestors.append(cur)
        cur = cur.parent
    ancestors.reverse()

    parts: list[str] = [doc_prefix] if doc_prefix else []
    if compact:
        if ancestors:
            top = ancestors[0]
            parts.append(f"{top.number}. {top.title}")
    else:
        parts.extend(f"{a.number}. {a.title}" for a in ancestors)
    parts.append(f"{section.number}. {section.title}")
    return " > ".join(p for p in parts if p)


# ---------------------------- Base chunk ----------------------------

def make_base_chunk(section: Section, doc_id: str, doc_title: str,
                    doc_prefix: str, compact_path: bool) -> dict:
    return {
        "chunk_id": "",
        "doc_id": doc_id,
        "doc_title": doc_title,
        "section_path": build_section_path(section, doc_prefix, compact=compact_path),
        "item_number": section.number,
        "chunk_type": "",
        "parent_id": (f"tr-{section.parent.number}-pai" if section.parent else None),
        "child_ids": [],
        "table_id": None,
        "row_range": None,
        "entities": [],
        "normas_citadas": [],
        "valores": [],
        "referencias_resolvidas": [],
        "source_anchor": section.number,
        "texto": "",
    }


# ---------------------------- Construtor de chunks por tipo ----------------------------

def build_parent_text(section: Section) -> str:
    """Texto do chunk pai: prosa propria (antes dos filhos) ou sumario de filhos."""
    own = [b.raw for b in section.blocks if b.type in
           ("paragraph", "numbered_clause", "table", "list")]
    if own:
        return "\n\n".join(own)
    titles = "; ".join(f"{c.number}. {c.title}" for c in section.children)
    return f"{section.number}. {section.title}\n\nSubseções: {titles}"


def make_table_chunk(base: dict, section: Section, block: Block,
                     idx: int, max_rows: int = 20) -> list[dict]:
    """Gera um ou mais chunks de tabela; split com cabecalho repetido se > max_rows."""
    lines = block.raw.split("\n")
    header = lines[:2] if len(lines) >= 2 else lines[:1]
    data = lines[2:] if len(lines) >= 2 else []

    if len(data) <= max_rows:
        chunk = dict(base)
        chunk["chunk_id"] = f"tr-{section.number}-tbl-{idx:02d}"
        chunk["chunk_type"] = "tabela"
        chunk["table_id"] = chunk["chunk_id"]
        chunk["row_range"] = [1, max(1, len(data))]
        chunk["texto"] = block.raw
        return [chunk]

    chunks: list[dict] = []
    for start in range(0, len(data), max_rows):
        group = data[start:start + max_rows]
        chunk = dict(base)
        chunk["chunk_id"] = f"tr-{section.number}-tbl-{idx:02d}-{start // max_rows + 1:02d}"
        chunk["chunk_type"] = "tabela"
        chunk["table_id"] = f"tr-{section.number}-tbl-{idx:02d}"
        chunk["row_range"] = [start + 1, start + len(group)]
        chunk["texto"] = "\n".join(header + group)
        chunks.append(chunk)
    return chunks


def make_list_chunks(base: dict, section: Section, block: Block,
                     counter: dict, max_items: int = 8) -> list[dict]:
    """Gera um ou mais chunks de lista; split com cabecalho repetido se > max_items."""
    lines = block.meta["lines"]
    groups = ([lines] if len(lines) <= max_items
              else [lines[i:i + max_items] for i in range(0, len(lines), max_items)])

    chunks: list[dict] = []
    for group in groups:
        counter["lista"] += 1
        chunk = dict(base)
        chunk["chunk_id"] = f"tr-{section.number}-lst-{counter['lista']:02d}"
        chunk["chunk_type"] = "lista"
        header = (f"[{section.number}. {section.title}]\n"
                  if len(groups) > 1 else "")
        chunk["texto"] = header + "\n".join(group)
        chunks.append(chunk)
    return chunks


# ---------------------------- Geracao de chunks ----------------------------

def generate_chunks(roots: list[Section], section_map: dict[str, Section],
                    doc_id: str, doc_title: str, doc_prefix: str,
                    compact_path: bool = True,
                    max_table_rows: int = 20,
                    max_list_items: int = 8) -> list[dict]:
    chunks: list[dict] = []

    def walk(section: Section) -> list[str]:
        base = make_base_chunk(section, doc_id, doc_title, doc_prefix, compact_path)

        if section.children:
            parent = dict(base)
            parent["chunk_id"] = f"tr-{section.number}-pai"
            parent["chunk_type"] = "secao"
            parent["texto"] = build_parent_text(section)
            chunks.append(parent)
            child_ids: list[str] = []
            for child in section.children:
                child_ids.extend(walk(child))
            parent["child_ids"] = child_ids
            return [parent["chunk_id"]]

        generated: list[str] = []
        counter = {"clausula": 0, "tabela": 0, "lista": 0}

        for block in section.blocks:
            if block.type == "table":
                counter["tabela"] += 1
                for chunk in make_table_chunk(base, section, block,
                                              counter["tabela"], max_table_rows):
                    chunks.append(chunk)
                    generated.append(chunk["chunk_id"])
            elif block.type == "list":
                list_chunks = make_list_chunks(base, section, block,
                                               counter, max_list_items)
                chunks.extend(list_chunks)
                generated.extend(c["chunk_id"] for c in list_chunks)
            else:
                counter["clausula"] += 1
                chunk = dict(base)
                chunk["chunk_id"] = f"tr-{section.number}-{counter['clausula']:03d}"
                chunk["chunk_type"] = "clausula"
                chunk["texto"] = block.raw
                chunks.append(chunk)
                generated.append(chunk["chunk_id"])

        if not generated:
            counter["clausula"] += 1
            chunk = dict(base)
            chunk["chunk_id"] = f"tr-{section.number}-{counter['clausula']:03d}"
            chunk["chunk_type"] = "clausula"
            chunk["texto"] = f"{section.number}. {section.title}"
            chunks.append(chunk)
            generated.append(chunk["chunk_id"])

        return generated

    for root in roots:
        walk(root)

    for chunk in chunks:
        enrich_chunk(chunk, section_map)
    return chunks


# ---------------------------- Enriquecimento ----------------------------

def enrich_chunk(chunk: dict, section_map: dict[str, Section]) -> None:
    text = chunk["texto"]
    chunk["normas_citadas"] = extract_normas(text)
    chunk["valores"] = extract_valores(text)
    chunk["entities"] = extract_entities(text, chunk["normas_citadas"])
    chunk["referencias_resolvidas"] = resolve_internal_refs(text, section_map)


def extract_normas(text: str) -> list[str]:
    found: set[str] = set()
    for rx in (NBR_RE, IEC_RE, ISO_RE, EDP_NORM_RE, ANP_RE):
        for m in rx.finditer(text):
            found.add(m.group(0).strip())
    return sorted(found)


def extract_valores(text: str) -> list[str]:
    out: list[str] = []
    for rx in (MONEY_RE, KVA_RE, KV_RE, PERCENT_RE, TIME_RE):
        for m in rx.finditer(text):
            out.append(m.group(0).strip())
    seen: set[str] = set()
    unique: list[str] = []
    for v in out:
        if v.lower() not in seen:
            seen.add(v.lower())
            unique.append(v)
    return unique


def extract_entities(text: str, normas: list[str]) -> list[str]:
    text_lower = text.lower()
    ents: set[str] = set()
    for term in DOMAIN_TERMS:
        if term.lower() in text_lower:
            ents.add(term)
    for m in re.finditer(r"\b[A-Z]{2,6}\b", text):
        acr = m.group(0)
        if acr in STOP_ACRONYMS:
            continue
        if any(acr in n for n in normas):
            continue
        ents.add(acr)
    return sorted(ents)


def resolve_internal_refs(text: str,
                          section_map: dict[str, Section]) -> list[dict]:
    refs: list[dict] = []
    for m in INT_REF_RE.finditer(text):
        num = m.group(1)
        # Descarta refs imediatamente precedidas por norma (ex.: "NBR 5440, item 6.1")
        window = text[max(0, m.start() - 40):m.start()]
        if re.search(r"(?:NBR|IEC|ISO|ES\.DT\.PDN\.)[\s\-]*[\d\.]+\s*[,;]?\s*$",
                     window, re.IGNORECASE):
            continue
        if num in section_map:
            target = section_map[num]
            refs.append({
                "texto": m.group(0).strip(),
                "item": num,
                "titulo": f"{target.number}. {target.title}",
            })
    return refs


# ---------------------------- Main ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Etapa 2: parse hierarquico + chunking do Termo de Referencia."
    )
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--input", default=str(root / "data" / "processed" / "edital_aracruz_es_clean.md"))
    parser.add_argument("--output", default=str(root / "data" / "processed" / "edital_aracruz_es_chunks.jsonl"))
    parser.add_argument("--doc-id", default="edital_aracruz_es")
    parser.add_argument("--doc-title", default="Termo de Referência — Aracruz/ES")
    parser.add_argument("--full-path", action="store_true",
                        help="section_path com TODOS os ancestrais (default: compacto, 2 niveis)")
    parser.add_argument("--max-table-rows", type=int, default=20,
                        help="Limiar para split de tabelas com cabecalho repetido")
    parser.add_argument("--max-list-items", type=int, default=8,
                        help="Limiar para split de listas com cabecalho repetido")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_path.exists():
        raise SystemExit(f"[ERRO] Arquivo nao encontrado: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    roots, section_map, doc_prefix = parse_document(text)

    chunks = generate_chunks(
        roots=roots,
        section_map=section_map,
        doc_id=args.doc_id,
        doc_title=args.doc_title,
        doc_prefix=doc_prefix,
        compact_path=not args.full_path,
        max_table_rows=args.max_table_rows,
        max_list_items=args.max_list_items,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c["chunk_type"]] = by_type.get(c["chunk_type"], 0) + 1

    print(f"[OK] {len(chunks)} chunks em {output_path}")
    print(f"[INFO] Tipos: {by_type}")
    print(f"[INFO] Secoes totais: {len(section_map)}")
    print(f"[INFO] Prefixo do documento: {doc_prefix!r}")


if __name__ == "__main__":
    main()