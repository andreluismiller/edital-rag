#!/usr/bin/env python3
"""Baixa modelos do FastEmbed para root/models/ (executar uma vez)."""

import argparse
from pathlib import Path
from fastembed import TextEmbedding, SparseTextEmbedding

DEFAULT_DENSE = "sentence-transformers/all-MiniLM-L6-v2"  # 384d, leve para Codespaces
DEFAULT_SPARSE = "Qdrant/bm25"


def download(models_dir: Path, dense_model: str, sparse_model: str) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/2] Baixando denso: {dense_model} -> {models_dir}")
    TextEmbedding(model_name=dense_model, cache_dir=str(models_dir))
    print(f"[2/2] Baixando esparso: {sparse_model} -> {models_dir}")
    SparseTextEmbedding(model_name=sparse_model, cache_dir=str(models_dir))
    print("[OK] Modelos prontos para uso offline.")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default=str(root / "models"))
    parser.add_argument("--dense-model", default=DEFAULT_DENSE)
    parser.add_argument("--sparse-model", default=DEFAULT_SPARSE)
    args = parser.parse_args()
    download(Path(args.models_dir), args.dense_model, args.sparse_model)


if __name__ == "__main__":
    main()