#!/usr/bin/env python3
"""Etapa 3: embedding + Qdrant local, otimizado para Codespaces."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Iterator

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

COLLECTION = "edital_aracruz_es"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384d
SPARSE_MODEL = "Qdrant/bm25"

PAYLOAD_INDEXES = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "item_number": models.PayloadSchemaType.KEYWORD,
    "chunk_type": models.PayloadSchemaType.KEYWORD,
    "section_path": models.PayloadSchemaType.TEXT,
    "parent_id": models.PayloadSchemaType.KEYWORD,
    "source_anchor": models.PayloadSchemaType.KEYWORD,
    "entities": models.PayloadSchemaType.KEYWORD,
    "normas_citadas": models.PayloadSchemaType.KEYWORD,
}


def iter_chunks(path: Path, batch_size: int) -> Iterator[list[dict]]:
    """Le o JSONL em lotes, sem carregar o arquivo inteiro na memoria."""
    batch: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def embedding_text(chunk: dict) -> str:
    """Texto enriquecido para o embedding (caminho + conteudo)."""
    path = chunk.get("section_path", "")
    body = chunk.get("texto", "")
    return f"{path}\n\n{body}" if body not in path else path


def point_id(chunk: dict) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"]))


def ensure_collection(client: QdrantClient, dense_dim: int, recreate: bool) -> None:
    if recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    for field, schema in PAYLOAD_INDEXES.items():
        try:
            client.create_payload_index(COLLECTION, field, schema)
        except Exception:
            pass  # indice ja existe


def process_batch(
    batch: list[dict],
    dense_model: TextEmbedding,
    sparse_model: SparseTextEmbedding,
    client: QdrantClient,
) -> None:
    texts = [embedding_text(c) for c in batch]
    dense_vecs = list(dense_model.embed(texts))
    sparse_vecs = list(sparse_model.embed(texts))

    points = []
    for chunk, dv, sv in zip(batch, dense_vecs, sparse_vecs):
        payload = {k: v for k, v in chunk.items() if k != "texto"}
        payload["texto"] = chunk["texto"]
        points.append(
            models.PointStruct(
                id=point_id(chunk),
                vector={
                    "dense": dv.tolist(),
                    "sparse": models.SparseVector(
                        indices=[int(i) for i in sv.indices],
                        values=[float(v) for v in sv.values],
                    ),
                },
                payload=payload,
            )
        )
    client.upsert(COLLECTION, points, wait=True)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default=str(root / "data" / "processed" / "edital_aracruz_es_chunks.jsonl")
    )
    parser.add_argument("--qdrant-path", default=str(root / "data" / "qdrant_db"))
    parser.add_argument("--models-dir", default=str(root / "models"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"[ERRO] Nao encontrado: {input_path}")

    models_dir = str(Path(args.models_dir).resolve())
    dense_model = TextEmbedding(
        model_name=DENSE_MODEL, cache_dir=models_dir, local_files_only=True
    )
    sparse_model = SparseTextEmbedding(
        model_name=SPARSE_MODEL, cache_dir=models_dir, local_files_only=True
    )
    dense_dim = dense_model.embedding_size
    print(f"[INFO] Modelo denso: {DENSE_MODEL} ({dense_dim}d), batch={args.batch_size}")

    qdrant_path = Path(args.qdrant_path)
    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))
    ensure_collection(client, dense_dim, recreate=args.recreate)

    total = 0
    for batch in iter_chunks(input_path, args.batch_size):
        process_batch(batch, dense_model, sparse_model, client)
        total += len(batch)
        print(f"[INFO] Inseridos {total} chunks", end="\r")

    print(f"\n[OK] Total: {client.count(COLLECTION).count} pontos")
    client.close()


if __name__ == "__main__":
    main()