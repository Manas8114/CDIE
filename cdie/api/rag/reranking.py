"""
CDIE v5 — RAG Reranking Module

Passage re-ranking via OPEA TEI Reranking service
with cosine-similarity fallback.
"""

from __future__ import annotations

from typing import Any

from cdie.observability import get_logger
from cdie.utils.circuit_breaker import opea_post

log = get_logger(__name__)


def rerank_opea(
    query: str,
    passages: list[dict[str, Any]],
    reranking_endpoint: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Re-rank *passages* using OPEA TEI Reranking service.

    Args:
        query:               Free-form query string.
        passages:            List of candidate passage dicts (must have ``'text'`` key).
        reranking_endpoint:  Base URL of the TEI reranking service.
        top_k:               Number of top passages to return.

    Returns:
        Re-ranked (and truncated to *top_k*) passage list.
        Falls back to original order on any failure.
    """
    try:
        texts = [p['text'] for p in passages]
        response = opea_post(
            f'{reranking_endpoint}/rerank',
            json={'query': query, 'texts': texts},
            timeout=10,
        )
        if response is None:
            log.warning('[rag.reranking] OPEA reranking unavailable — keeping cosine order')
            return passages[:top_k]

        response.raise_for_status()
        scores = response.json()

        for i, score_data in enumerate(scores):
            if i < len(passages):
                idx = score_data.get('index', i)
                passages[idx]['rerank_score'] = score_data.get('score', 0)

        passages.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        for p in passages[:top_k]:
            p['retrieval_method'] = p.get('retrieval_method', '') + '+reranking'
        return passages[:top_k]

    except Exception as exc:
        log.warning('[rag.reranking] Reranking failed — keeping cosine order', error=str(exc))
        return passages[:top_k]
