# ADR-003 — OPEA GenAIComps for LLM + Embedding Services

**Date:** 2026-04-24  
**Status:** Accepted  
**Deciders:** CDIE v5 Architecture Team

---

## Context

CDIE's RAG Explanation Engine requires three AI capabilities:
1. **Text embeddings** for semantic retrieval over telecom playbooks.
2. **Passage re-ranking** for precision in the top-k results.
3. **LLM text generation** for natural-language causal reports.

The system must run on **Intel hardware** (Xeon Sapphire Rapids / AMX-capable CPUs)
and must remain operational when LLM services are unavailable (graceful degradation).

---

## Decision

Use **OPEA (Open Platform for Enterprise AI) GenAIComps** microservices as the primary
AI backend, with automatic fallback at every layer:

```
Layer 1 — LLM Explanation
  OPEA TextGen (opea/llm-textgen, wraps TGI)
    └── Fallback: OpenAI gpt-4o-mini (if OPENAI_API_KEY set)
          └── Fallback: template-based deterministic explanation

Layer 2 — Embeddings (RAG retrieval)
  OPEA TEI Embedding (BAAI/bge-base-en-v1.5, Intel-optimized)
    └── Fallback: TF-IDF sparse vectors (sklearn, always available)

Layer 3 — Reranking
  OPEA TEI Reranking (BAAI/bge-reranker-base)
    └── Fallback: cosine similarity on TF-IDF vectors
```

All OPEA calls go through `cdie/utils/circuit_breaker.py` (`opea_post`),
which retries up to 3 times with exponential backoff before returning `None`.

---

## Alternatives Considered

| Option | Intel Optimization | Self-hosted | Graceful Degradation | Why Rejected |
|--------|-------------------|-------------|----------------------|--------------|
| **Direct HuggingFace TGI** | Partial | Yes | Manual | OPEA adds OpenAI-compatible /v1/ interface, metrics, and health checks |
| **OpenAI API only** | No | No | No (API key required) | Vendor lock-in; fails in air-gapped/hackathon environments |
| **Ollama** | No | Yes | Partial | No official Intel AMX optimization; different API surface |
| **vLLM** | Partial (ROCm/CUDA) | Yes | No | Not officially supported on Intel CPU path |
| **OPEA (chosen)** | ✅ Full AMX/AVX-512 | Yes | ✅ 3-layer fallback | — |

---

## Consequences

**Positive:**
- Full Intel AMX/AVX-512 acceleration via `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.
- OpenAI-compatible `/v1/chat/completions` allows drop-in OpenAI SDK usage.
- Three independent fallback paths ensure the API never returns 500 for LLM failure.
- Sufficiency gate (`_check_sufficiency` in `rag/llm.py`) skips LLM calls entirely
  when causal evidence is already high-confidence, reducing cost and latency.

**Negative:**
- OPEA TextGen image (`opea/llm-textgen:latest`) requires internet access to pull.
- TGI model download (7B parameters ≈ 14 GB) takes significant time on first run.
- Port mapping (`8888:9000`) differs from the container's internal port, which can
  confuse ops teams (documented in `.env.example` and `TROUBLESHOOTING.md`).

**Mitigation:**
- `DEPLOYMENT_READINESS.md` documents the pre-pull step.
- All OPEA endpoints are optional: the system falls back to template-based explanations
  if none are reachable, producing correct causal outputs without LLM augmentation.

---

## References

- `cdie/api/rag/` — RAG engine
- `cdie/utils/circuit_breaker.py` — retry wrapper
- `docker-compose.yml` — OPEA service definitions
- [OPEA Project](https://opea-project.github.io/)
