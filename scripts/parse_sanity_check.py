# scripts/parse_sanity_check.py
from pathlib import Path

from edital_rag.ingestion.parsing import parse_document

PDF = Path(__file__).resolve().parent.parent / "data/samples/sample_edital.pdf"
doc = parse_document(PDF)

print(f"tabelas totais:  {len(doc.tables)}")
print(f"tabelas na p.1:  {len(doc.tables_on_page(1))}")

for t in doc.tables_on_page(1):
    print(f"\n=== tabela idx={t.index} p={t.page} "
          f"n_rows={t.n_rows} n_cols={t.n_cols} bbox={t.bbox}")
    print(f"header: {t.header!r}")
    print(f"col_index('ITEM')   = {t.column_index('ITEM')}")
    print(f"col_index('CÓDIGO') = {t.column_index('CÓDIGO')}")
    print(f"col_index('CODIGO') = {t.column_index('CODIGO')}")
    print("primeiras 3 linhas de rows:")
    for i, row in enumerate(t.rows[:3]):
        print(f"  [{i}] {row}")
    print(f"coluna('CÓDIGO') = {t.column('CÓDIGO')!r}")
    print(f"coluna('CODIGO') = {t.column('CODIGO')!r}")

# print(f"páginas: {doc.page_count}, tabelas: {len(doc.tables)}")
# print(f"sha256: {doc.sha256[:16]}…, bytes: {doc.size_bytes}")

# # 1) Tabela de itens da página 1 tem os dois transformadores?
# tabelas_p1 = doc.tables_on_page(1)
# assert tabelas_p1, "esperava ao menos uma tabela na p.1"
# itens = next(t for t in tabelas_p1 if t.column_index("ITEM") is not None)
# codigos = itens.column("CÓDIGO")
# print(f"códigos na p.1: {codigos}")
# assert "170.001.288" in codigos and "170.001.289" in codigos

# # 2) Forward-fill aplicado: especificação deve estar presente nas 2 linhas
# esp = itens.column("ESPECIFICAÇÃO") or itens.column("ESPECIFICACAO")
# assert all(len(e) > 50 for e in esp if e), "especificação ficou vazia em alguma linha"

# # 3) Sem mojibake
# assert "ÀÁÂÃÄÅ" not in doc.pages[0].text
# assert "Página 58 de 84" not in doc.pages[13].text, "rodapé deveria ter sido removido"

# # 4) Hífen quebrado não sobrou
# assert "trans-\nformador" not in doc.pages_text
# print("OK — parsing válido para o TR de Primavera do Leste")