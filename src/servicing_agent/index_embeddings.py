from __future__ import annotations

import argparse
import json

from .retrieval import EmbeddingUnavailableError, HybridRetrievalService
from .settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initialize or refresh the local document-embedding cache."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute embeddings even when the fingerprinted cache exists.",
    )
    args = parser.parse_args()
    settings = Settings.from_environment()
    retriever = HybridRetrievalService(
        mode="semantic",
        embedding_model=settings.retrieval_model,
        embedding_cache_dir=settings.retrieval_cache_dir,
        top_k=settings.retrieval_top_k,
        rrf_k=settings.retrieval_rrf_k,
        semantic_min_score=settings.retrieval_semantic_min_score,
        semantic_min_top_score=settings.retrieval_semantic_min_top_score,
    )
    try:
        cache_status = retriever.refresh_embeddings(force=args.force)
    except EmbeddingUnavailableError as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}))
        return 1
    print(
        json.dumps(
            {
                "status": "ok",
                "cache_status": cache_status,
                "model": settings.retrieval_model,
                "cache_dir": str(settings.retrieval_cache_dir),
                "document_count": len(retriever.documents),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
