"""
CDIE v5 — RAG Retrieval Module

Handles:
- Loading telecom playbooks from ``DATA_DIR/telecom_playbooks.json``
- Building a TF-IDF sparse index (fallback, always available)
- Building a dense embedding index via OPEA TEI Embedding
- Retrieving top-k analogies for a free-form query
"""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cdie.config import DATA_DIR
from cdie.observability import get_logger
from cdie.utils.circuit_breaker import opea_post

log = get_logger(__name__)


# ── Playbook Loading ──────────────────────────────────────────────────────────

def load_historical_events() -> list[dict[str, Any]]:
    """Load telecom playbooks from the data directory.

    Returns an empty list if the file is missing (degraded mode).
    """
    playbooks_path = DATA_DIR / 'telecom_playbooks.json'
    if playbooks_path.exists():
        with open(playbooks_path, encoding='utf-8') as f:
            return cast(list[dict[str, Any]], json.load(f))
    log.warning('[rag.retrieval] Playbooks file not found — RAG analogies disabled', path=str(playbooks_path))
    return []


# ── Index Builders ────────────────────────────────────────────────────────────

def build_tfidf_index(texts: list[str]) -> tuple[TfidfVectorizer | None, Any]:
    """Build a TF-IDF sparse index over *texts*.

    Returns:
        ``(vectorizer, tfidf_matrix)`` on success, ``(None, None)`` if *texts* is empty.
    """
    if not texts:
        return None, None
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def build_opea_index(texts: list[str], embedding_endpoint: str) -> np.ndarray | None:
    """Build a dense embedding index using OPEA TEI Embedding.

    Args:
        texts:              List of document strings to embed.
        embedding_endpoint: Base URL of the TEI embedding service.

    Returns:
        ``np.ndarray`` of shape ``(n_texts, embedding_dim)`` on success,
        or ``None`` if the service is unreachable.
    """
    response = opea_post(f'{embedding_endpoint}/embed', json={'inputs': texts}, timeout=30)
    if response is None:
        return None
    try:
        response.raise_for_status()
        return np.array(response.json())
    except Exception as exc:
        log.warning('[rag.retrieval] OPEA embedding index build failed', error=str(exc))
        return None


def embed_query_opea(query: str, embedding_endpoint: str) -> np.ndarray | None:
    """Embed a single query string using OPEA TEI Embedding.

    Returns:
        ``np.ndarray`` on success, ``None`` on failure.
    """
    response = opea_post(f'{embedding_endpoint}/embed', json={'inputs': query}, timeout=10)
    if response is None:
        return None
    try:
        response.raise_for_status()
        return np.array(response.json())
    except Exception as exc:
        log.warning('[rag.retrieval] Query embedding failed', error=str(exc))
        return None


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_tfidf(
    query: str,
    events: list[dict[str, Any]],
    vectorizer: TfidfVectorizer | None,
    tfidf_matrix: Any,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve top-k analogies using TF-IDF cosine similarity."""
    if vectorizer is None or tfidf_matrix is None or not events:
        return []
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        sim = float(similarities[idx])
        event = events[idx].copy()
        event['similarity'] = float(np.round(sim, 4))
        event['confidence'] = 'High' if sim > 0.4 else 'Medium' if sim > 0.2 else 'Low'
        event['retrieval_method'] = 'tfidf'
        results.append(event)
    return results


def retrieve_opea(
    query: str,
    events: list[dict[str, Any]],
    embeddings_cache: np.ndarray,
    embedding_endpoint: str,
    top_k: int = 3,
) -> list[dict[str, Any]] | None:
    """Retrieve top-k analogies using OPEA TEI dense embeddings.

    Returns:
        List of analogy dicts on success, ``None`` if OPEA is unreachable
        (caller should fall back to TF-IDF).
    """
    query_embedding = embed_query_opea(query, embedding_endpoint)
    if query_embedding is None:
        return None

    qe = query_embedding.reshape(1, -1) if query_embedding.ndim == 1 else query_embedding
    cache = embeddings_cache.reshape(1, -1) if embeddings_cache.ndim == 1 else embeddings_cache

    similarities = cosine_similarity(qe, cache)[0]
    n_candidates = min(top_k * 2, len(events))
    top_indices = np.argsort(similarities)[::-1][:n_candidates]

    candidates: list[dict[str, Any]] = []
    for idx in top_indices:
        event = events[idx].copy()
        event['similarity'] = float(np.round(similarities[idx], 4))
        event['confidence'] = (
            'High' if similarities[idx] > 0.6 else 'Medium' if similarities[idx] > 0.3 else 'Low'
        )
        event['retrieval_method'] = 'opea_tei_embedding'
        candidates.append(event)
    return candidates
