import argparse
import os
import time
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Prefetch,
    SparseVector,
    FusionQuery,
    Fusion,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)
from fastembed import TextEmbedding, SparseTextEmbedding
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = ROOT / "models"
DEFAULT_QDRANT_PATH = ROOT / "data" / "qdrant_db"

DEFAULT_COLLECTION = "edital_aracruz_es"
DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


class RAGSystem:
    """RAG para o Termo de Referência do edital de transformadores (Aracruz/ES)."""

    INSTRUCTIONS = """
Você é um assistente especialista em editais de compras públicas de transformadores elétricos,
atendendo colaboradores dos setores de Engenharia, Comercial e PMO de uma indústria de transformadores.
Utilize APENAS as informações presentes no "Contexto" abaixo para responder à pergunta do usuário.

REGRAS:
- Responda em Português do Brasil.
- Cite sempre o item do Termo de Referência em que a resposta se baseia (ex.: "Item 5.2.3.10").
- Quando a pergunta envolver uma lista (ensaios, documentos, obrigações, sanções), reproduza a lista completa.
- Preserve valores monetários, potências (kVA), tensões (kV), percentuais e prazos exatamente como no contexto.
- Se a informação não constar no contexto, diga claramente que não está no Termo de Referência fornecido.
- Retorne apenas a resposta direta e formatada em Markdown (sem comentários adicionais ou JSON).
""".strip()

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        qdrant_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        models_dir: Optional[str] = None,
        dense_model_name: str = DEFAULT_DENSE_MODEL,
        sparse_model_name: str = DEFAULT_SPARSE_MODEL,
        llm_model: str = "qwen/qwen3.8-27b",
    ):
        self.collection_name = collection_name

        # LLM (Groq via compatibilidade OpenAI)
        self.llm_client = OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        self.model_name = llm_model

        # Qdrant: local em disco por padrão, URL se explicitamente configurado
        if qdrant_url or os.environ.get("QDRANT_URL"):
            url = qdrant_url or os.environ["QDRANT_URL"]
            print(f"[INFO] Qdrant remoto: {url}")
            self.qdrant_client = QdrantClient(url=url)
        else:
            path = Path(qdrant_path or DEFAULT_QDRANT_PATH)
            path.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Qdrant local: {path}")
            self.qdrant_client = QdrantClient(path=str(path))

        # Embeddings: mesmos modelos da ingestão, carregados do cache local
        models_path = str(Path(models_dir or DEFAULT_MODELS_DIR).resolve())
        print(f"[INFO] Carregando embeddings de {models_path} ...")

        try:
            self.dense_model = TextEmbedding(
                model_name=dense_model_name,
                cache_dir=models_path,
            )
            self.sparse_model = SparseTextEmbedding(
                model_name=sparse_model_name,
                cache_dir=models_path,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Falha ao carregar embeddings a partir de '{models_path}'.\n"
                f"Exceção original: {type(exc).__name__}: {exc}\n"
                "Verifique se os modelos foram baixados com:\n"
                "  uv run python scripts/download_models.py"
            ) from exc        

        # Preços do Groq (aproximados, USD por 1M de tokens)
        self.pricing = {
            "llama-3.3-70b-versatile": {"prompt": 0.59, "completion": 0.79},
        }

    # ------------------------- Infraestrutura -------------------------

    def check_collection_exists(self) -> bool:
        try:
            return self.qdrant_client.collection_exists(self.collection_name)
        except Exception as exc:
            print(f"[ERRO] Ao verificar coleção: {exc}")
            return False

    def list_distinct(self, field: str, limit: int = 1000) -> list[str]:
        """Lista valores distintos de um campo keyword, útil para popular filtros na UI."""
        values: set[str] = set()
        next_offset = None
        try:
            while True:
                points, next_offset = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    with_payload=[field],
                    with_vectors=False,
                    limit=256,
                    offset=next_offset,
                )
                for p in points:
                    v = (p.payload or {}).get(field)
                    if v is None:
                        continue
                    if isinstance(v, list):
                        values.update(str(x) for x in v if x)
                    else:
                        values.add(str(v))
                if next_offset is None or len(values) >= limit:
                    break
        except Exception as exc:
            print(f"[ERRO] Ao listar '{field}': {exc}")
        return sorted(values)

    # ------------------------- Filtros -------------------------

    def _build_filter(
        self,
        chunk_type: Optional[str] = None,
        doc_id: Optional[str] = None,
        item_number: Optional[str] = None,
        entity: Optional[str] = None,
        norma: Optional[str] = None,
    ) -> Optional[Filter]:
        conditions = []
        if chunk_type:
            conditions.append(FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type)))
        if doc_id:
            conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        if item_number:
            conditions.append(FieldCondition(key="item_number", match=MatchValue(value=item_number)))
        if entity:
            conditions.append(FieldCondition(key="entities", match=MatchAny(any=[entity])))
        if norma:
            conditions.append(FieldCondition(key="normas_citadas", match=MatchAny(any=[norma])))
        return Filter(must=conditions) if conditions else None

    # ------------------------- Busca híbrida -------------------------

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        chunk_type: Optional[str] = None,
        doc_id: Optional[str] = None,
        item_number: Optional[str] = None,
        entity: Optional[str] = None,
        norma: Optional[str] = None,
    ):
        """Busca híbrida (denso + esparso) com RRF e filtros opcionais."""
        query_dense = list(self.dense_model.embed([query]))[0].tolist()

        sparse_gen = list(self.sparse_model.embed([query]))[0]
        query_sparse = SparseVector(
            indices=[int(i) for i in sparse_gen.indices],
            values=[float(v) for v in sparse_gen.values],
        )

        query_filter = self._build_filter(
            chunk_type=chunk_type,
            doc_id=doc_id,
            item_number=item_number,
            entity=entity,
            norma=norma,
        )

        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(query=query_dense, using="dense", limit=limit, filter=query_filter),
                Prefetch(query=query_sparse, using="sparse", limit=limit, filter=query_filter),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return results.points

    # ------------------------- Contexto e prompt -------------------------

    def build_context(self, search_results: list) -> str:
        """Monta o bloco de contexto a partir do payload dos chunks recuperados."""
        blocks: list[str] = []
        for i, point in enumerate(search_results, start=1):
            p = point.payload or {}
            header = (
                f"[{i}] Item {p.get('item_number', '?')} — "
                f"{p.get('chunk_type', '?')} (score={point.score:.4f})"
            )
            lines = [header, f"Caminho: {p.get('section_path', '')}"]
            if p.get("normas_citadas"):
                lines.append(f"Normas citadas: {', '.join(p['normas_citadas'])}")
            if p.get("valores"):
                lines.append(f"Valores/prazos: {', '.join(p['valores'])}")
            if p.get("row_range"):
                lines.append(f"Linhas da tabela: {p['row_range']}")
            lines.append("Conteúdo:")
            lines.append(p.get("texto", ""))
            blocks.append("\n".join(lines))
        return "\n\n---\n\n".join(blocks)

    def build_prompt(self, user_query: str, search_results: list) -> str:
        context = self.build_context(search_results)
        template = """
{instructions}

Contexto:
{context}

Pergunta do Usuário:
{question}

Resposta:
""".strip()
        return template.format(
            instructions=self.INSTRUCTIONS,
            context=context,
            question=user_query,
        )

    # ------------------------- LLM e métricas -------------------------

    def generate_response(self, prompt: str):
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        usage = response.usage
        return (
            response.choices[0].message.content,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        price = self.pricing.get(self.model_name)
        if not price:
            return 0.0
        return (
            (prompt_tokens / 1_000_000) * price["prompt"]
            + (completion_tokens / 1_000_000) * price["completion"]
        )

    # ------------------------- Pipeline -------------------------

    def execute_rag_pipeline(
        self,
        user_query: str,
        limit: int = 5,
        chunk_type: Optional[str] = None,
        doc_id: Optional[str] = None,
        item_number: Optional[str] = None,
        entity: Optional[str] = None,
        norma: Optional[str] = None,
    ) -> dict:
        start = time.time()

        if not self.check_collection_exists():
            raise RuntimeError(
                f"A coleção '{self.collection_name}' não foi encontrada no Qdrant."
            )

        results = self.hybrid_search(
            user_query,
            limit=limit,
            chunk_type=chunk_type,
            doc_id=doc_id,
            item_number=item_number,
            entity=entity,
            norma=norma,
        )

        prompt = self.build_prompt(user_query, results)
        answer, pt, ct, tt = self.generate_response(prompt)

        elapsed = round(time.time() - start, 4)
        cost = self.calculate_cost(pt, ct)

        sources = [
            {
                "score": round(p.score, 4),
                "item_number": (p.payload or {}).get("item_number"),
                "chunk_type": (p.payload or {}).get("chunk_type"),
                "section_path": (p.payload or {}).get("section_path"),
                "normas_citadas": (p.payload or {}).get("normas_citadas", []),
            }
            for p in results
        ]

        return {
            "question": user_query,
            "answer": answer,
            "filters": {
                "chunk_type": chunk_type,
                "doc_id": doc_id,
                "item_number": item_number,
                "entity": entity,
                "norma": norma,
            },
            "model_used": self.model_name,
            "instructions": self.INSTRUCTIONS,
            "prompt": prompt,
            "response_time_seconds": elapsed,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
            "cost_usd": cost,
            "context_sources": sources,
        }


# ------------------------- CLI -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG do Termo de Referência — Aracruz/ES")
    parser.add_argument("query", help="Pergunta em linguagem natural")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-type", choices=["clausula", "tabela", "lista", "secao"])
    parser.add_argument("--item-number", help="Ex.: 5.2.3.10 (match exato)")
    parser.add_argument("--doc-id", help="Ex.: edital_aracruz_es")
    parser.add_argument("--entity", help="Ex.: óleo isolante")
    parser.add_argument("--norma", help="Ex.: NBR 5440")
    parser.add_argument("--qdrant-path", default=str(DEFAULT_QDRANT_PATH))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    args = parser.parse_args()

    rag = RAGSystem(
        qdrant_path=args.qdrant_path,
        models_dir=args.models_dir,
        dense_model_name=args.dense_model,
    )
    out = rag.execute_rag_pipeline(
        args.query,
        limit=args.top_k,
        chunk_type=args.chunk_type,
        doc_id=args.doc_id,
        item_number=args.item_number,
        entity=args.entity,
        norma=args.norma,
    )

    print("\n=== RESPOSTA ===")
    print(out["answer"])
    print("\n=== FONTES ===")
    for s in out["context_sources"]:
        print(f"  [{s['score']}] {s['item_number']} ({s['chunk_type']}) — {s['section_path']}")
    print("\n=== TELEMETRIA ===")
    print(f"Modelo: {out['model_used']}")
    print(f"Tempo: {out['response_time_seconds']}s")
    print(f"Tokens (prompt/compl/total): {out['prompt_tokens']}/{out['completion_tokens']}/{out['total_tokens']}")
    print(f"Custo estimado: ${out['cost_usd']:.6f}")


if __name__ == "__main__":
    main()