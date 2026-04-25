"""
CDIE v5 — RAG Package

Backward-compatible entry point. All imports from ``cdie.api.rag`` continue
to work unchanged::

    from cdie.api.rag import ExplanationEngine          # unchanged
    from cdie.api.rag.engine import ExplanationEngine   # explicit sub-module

Sub-modules:
    cache      — Redis connection setup, cache key generation, read/write
    retrieval  — TF-IDF and OPEA TEI dense embedding index + analogy retrieval
    reranking  — OPEA TEI passage re-ranking with cosine fallback
    llm        — LLM explanation generation and sufficiency gate
    engine     — ExplanationEngine orchestrator (imports from all sub-modules)
"""

from cdie.api.rag.engine import ExplanationEngine

__all__ = ['ExplanationEngine']
