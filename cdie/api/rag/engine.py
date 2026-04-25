"""
CDIE v5 — RAG Engine

Orchestrates retrieval → (optional reranking) → (optional LLM) → explanation.

This is the single public class imported by the rest of the codebase::

    from cdie.api.rag import ExplanationEngine
    # or equivalently:
    from cdie.api.rag.engine import ExplanationEngine

The Engine is designed to degrade gracefully at every layer:
- No playbooks → empty analogies, template explanation only
- No OPEA embedding → TF-IDF retrieval
- No OPEA reranking → cosine similarity ordering
- No OPEA LLM → template explanation
- No Redis → no caching
"""

from __future__ import annotations

import os
from typing import Any

from cdie.api.rag.cache import build_redis_client, cache_get, cache_set, make_cache_key
from cdie.api.rag.llm import (
    build_template_explanation,
    check_sufficiency,
    generate_llm_explanation,
)
from cdie.api.rag.reranking import rerank_opea
from cdie.api.rag.retrieval import (
    build_opea_index,
    build_tfidf_index,
    load_historical_events,
    retrieve_opea,
    retrieve_tfidf,
)
from cdie.observability import get_logger

log = get_logger(__name__)


class ExplanationEngine:
    """Orchestrates RAG retrieval and LLM explanation generation.

    OPEA Integration:
    - TEI Embedding (BAAI/bge-base-en-v1.5) for semantic vector retrieval
    - TEI Reranking (BAAI/bge-reranker-base) for passage re-ranking
    - LLM TextGen (Intel/neural-chat-7b-v3-3) for Assumption Intelligence
    """

    def __init__(self) -> None:
        self.events = load_historical_events()
        self.event_texts = [e['text'] for e in self.events]

        # TF-IDF index (always built as fallback)
        self.vectorizer, self.tfidf_matrix = build_tfidf_index(self.event_texts)

        # OPEA TEI Embedding
        self.embedding_endpoint = os.environ.get('OPEA_EMBEDDING_ENDPOINT', '').strip() or None
        self.reranking_endpoint = os.environ.get('OPEA_RERANKING_ENDPOINT', '').strip() or None
        self.embeddings_cache: Any = None
        self.embedding_provider = 'tfidf'

        if self.embedding_endpoint:
            embeddings = build_opea_index(self.event_texts, self.embedding_endpoint)
            if embeddings is not None:
                self.embeddings_cache = embeddings
                self.embedding_provider = 'opea_tei'
                log.info('[rag.engine] OPEA TEI Embedding ready', endpoint=self.embedding_endpoint)
            else:
                log.warning('[rag.engine] OPEA TEI Embedding unavailable — using TF-IDF')
        else:
            log.info('[rag.engine] OPEA_EMBEDDING_ENDPOINT not set — using TF-IDF')

        self.reranking_provider = 'opea_tei' if self.reranking_endpoint else 'cosine'
        if self.reranking_endpoint:
            log.info('[rag.engine] OPEA TEI Reranking ready', endpoint=self.reranking_endpoint)
        else:
            log.info('[rag.engine] Using cosine similarity for ranking')

        # OPEA TextGen / OpenAI
        self.opea_endpoint = os.environ.get('OPEA_LLM_ENDPOINT', '').strip() or None
        self.openai_api_key = os.environ.get('OPENAI_API_KEY', '').strip() or None
        self.llm_model = os.environ.get('LLM_MODEL_ID', 'Intel/neural-chat-7b-v3-3')
        self.client: Any = None
        self.llm_provider = 'template'

        # Redis cache
        self.cache = build_redis_client()

        # Initialise LLM client
        if self.opea_endpoint:
            self._init_llm_client(self.opea_endpoint, self.openai_api_key)
        elif self.openai_api_key:
            self._init_llm_client('https://api.openai.com', self.openai_api_key, model='gpt-4o-mini')

    def _init_llm_client(self, base_url: str, api_key: str | None, model: str | None = None) -> None:
        try:
            from openai import OpenAI

            self.client = OpenAI(
                base_url=f'{base_url}/v1',
                api_key=api_key or 'opea-placeholder',
            )
            if model:
                self.llm_model = model
            self.llm_provider = 'openai_compat'
            log.info('[rag.engine] LLM client ready', base_url=base_url, model=self.llm_model)
        except Exception as exc:
            log.warning('[rag.engine] LLM client init failed — using templates', error=str(exc))

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve_analogies(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Retrieve top-k historical telecom analogies for a query.

        Tries OPEA TEI dense embeddings first, falls back to TF-IDF.
        If OPEA reranking is configured, applies a second-stage re-rank.
        """
        candidates: list[dict[str, Any]] | None = None

        if self.embedding_provider == 'opea_tei' and self.embeddings_cache is not None:
            candidates = retrieve_opea(
                query, self.events, self.embeddings_cache, self.embedding_endpoint or '', top_k=top_k * 2
            )
            if candidates is None:
                log.warning('[rag.engine] OPEA retrieval failed — falling back to TF-IDF')

        if candidates is None:
            candidates = retrieve_tfidf(query, self.events, self.vectorizer, self.tfidf_matrix, top_k=top_k)

        if self.reranking_endpoint and candidates:
            candidates = rerank_opea(query, candidates, self.reranking_endpoint, top_k=top_k)

        return candidates[:top_k] if candidates else []

    def generate_explanation(
        self,
        query_type: str,
        source: str,
        target: str,
        effect: dict[str, Any],
        refutation_status: dict[str, Any] | None = None,
        temporal_info: dict[str, Any] | None = None,
        analogies: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a causal explanation combining RAG analogies and (optionally) LLM generation.

        Returns a Markdown-formatted explanation string. Never raises.
        """
        if analogies is None:
            query = f'{query_type} {source} {target} causal intervention telecom fraud'
            analogies = self.retrieve_analogies(query)

        # Cache key
        cache_key = make_cache_key(query_type, source, target, effect, refutation_status)
        cached = cache_get(self.cache, cache_key)
        if cached:
            return cached

        # Template fallback (always computed — may be replaced by LLM below)
        fallback_text = build_template_explanation(
            query_type, source, target, effect, refutation_status, analogies, temporal_info
        )

        # Sufficiency gate: skip LLM when causal evidence is already strong
        if check_sufficiency(effect, refutation_status, analogies):
            cache_set(self.cache, cache_key, fallback_text)
            return fallback_text

        report = fallback_text
        if self.client:
            try:
                llm_text = generate_llm_explanation(
                    self.client,
                    self.llm_model,
                    query_type,
                    source,
                    target,
                    effect,
                    refutation_status,
                    analogies,
                )
                if llm_text:
                    report = llm_text
            except Exception as exc:
                log.warning('[rag.engine] LLM generation failed — using template', error=str(exc))
                report = fallback_text

        cache_set(self.cache, cache_key, report)
        return report
