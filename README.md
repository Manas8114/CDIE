# CDIE

Causal Decision Intelligence Engine — causal AI + RAG for telecom fraud detection with OPEA GenAI microservices.

## Overview

CDIE discovers *causal mechanisms* behind SIM box fraud (not just correlations), validates them through refutation tests, and generates actionable intelligence reports via RAG.

## Architecture (3 OPEA Components)

```
┌─────────────────────────────────────────────────────────────────┐
│  OPEA GenAIComps Layer (3 Microservices)                        │
│  ┌──────────────┐  ┌───────────────────────────────┐           │
│  │  TGI Backend  │──│  OPEA TextGen Microservice    │           │
│  │  Intel/neural- │  │  /v1/chat/completions         │           │
│  │  chat-7b-v3-3  │  │  (OpenAI compatible)          │           │
│  └──────────────┘  └────────────────┬────────────────┘           │
│  ┌──────────────┐  ┌────────────────┴────────────────┐           │
│  │  TEI Embedding│  │  TEI Reranking                   │           │
│  │  BAAI/bge-    │  │  BAAI/bge-reranker-base         │           │
│  │  base-en-v1.5  │  │  Cross-encoder precision       │           │
│  └──────────────┘  └──────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│  CDIE v5 Engine                                               │
│  OFFLINE PIPELINE                    ONLINE API                │
│  ┌──────────────┐  ┌──────────┐    ┌────────────────────────┐ │
│  │ SCM DataGen  │→ │ CATL     │→   │ FastAPI + Safety Map   │ │
│  │ (12-node DAG)│  │ (5 tests)│    │ Lookup + KS-Staleness  │ │
│  ├──────────────┤  ├──────────┤    ├────────────────────────┤ │
│  │ GFCI + PCMCI+│→ │ DoWhy    │→   │ Intent Parser + RAG    │ │
│  │ Discovery    │  │ Refute   │    │ (TEI Embed→Rerank→      │ │
│  ├──────────────┤  ├──────────┤    │  OPEA LLM Briefing)    │ │
│  │ LinearDML    │→ │ MAPIE    │→   ├────────────────────────┤ │
│  │ (EconML)     │  │ (CI)     │    │ Prescriptive Engine    │ │
│  └──────────────┘  └──────────┘    │ + safe_run_pipeline    │ │
└─────────────────────────────────────────────────────────────────┘
```

## Key Innovations

| Component | Innovation |
|-----------|------------|
| **CATL** | Causal Assumption Transparency Layer — 5 explicit tests before discovery |
| **GFCI** | Handles latent confounders (FCI) + temporal (PCMCI+) |
| **LinearDML** | Doubly-robust ATE with 95% conformal intervals |
| **Safety Map** | Pre-computed SQLite — <200ms online lookup |
| **OPEA RAG** | 3 microservices (TGI + TEI×2) for Intel-optimized inference |
| **KS-Staleness** | Detects data drift between training and serving |

## Quick Start

```bash
# One-click (Docker)
cp .env.example .env
# Edit: HF_TOKEN=your_token
chmod +x setup.sh && ./setup.sh
# Starts 7 containers: TGI→TextGen, TEI Embed, TEI Rerank, Pipeline, FastAPI, Next.js

# Manual
pip install -r requirements.lock
export CDIE_RUNTIME_DIR=/tmp/cdie-runtime
python -m cdie.pipeline.run_pipeline
python -m uvicorn cdie.api.main:app --port 8000
cd frontend && npm install && npm run build && npm run start
```

## Telecom Use Case: SIM Box Fraud

- **Problem**: $3.8B/yr revenue leakage (GSMA)
- **Causal Question**: "Does tightening fraud policy → reduce leakage?"
- **Correlation Trap**: High CDR volume correlates with fraud, but *causes* fraud?
- **CDIE Answer**: "Tightening policy by 20% → 18% leakage reduction (95% CI [14%, 22%])"

## Intel Optimization

- TGI + TEI containers use `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`
- Benchmarks show ~18× throughput on Sapphire Rapids vs baseline

## Limitations

- Synthetic validation (12-node SCM, 2000 rows)
- Real operator data needed for production calibration
- GFCI memory-intensive (~57GB peak on 12-node)
- Static DAG — no online causal updating

## License

MIT