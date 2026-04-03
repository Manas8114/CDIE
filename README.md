# CDIE v4 — Causal Decision Intelligence Engine for Telecom

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OPEA Components](https://img.shields.io/badge/OPEA-3%20Components-blue.svg)](https://opea.dev)
[![Intel Optimized](https://img.shields.io/badge/Intel-AMX%2FAVX--512-0071C5.svg)](https://www.intel.com)
[![ITU AI4Good](https://img.shields.io/badge/ITU-AI4Good%20Challenge-orange.svg)](https://aiforgood.itu.int)

> **Causal AI + RAG for Telecom Fraud Detection** — We don't just detect fraud correlations; we discover causal mechanisms, validate them through refutation tests, and generate actionable intelligence reports using OPEA GenAIComps.

---

## 📚 Documentation

For a deep dive into the system, see the **[Comprehensive Documentation Index](docs/README.md)**.

- **[Introduction](docs/introduction.md)** — Causal AI vs. ML in Telecom.
- **[Architecture](docs/architecture.md)** — Two-phase system design.
- **[Causal Pipeline](docs/causal_pipeline.md)** — GFCI, LinearDML, and DoWhy Refutation.
- **[OPEA Integration](docs/opea_integration.md)** — Intel-optimized GenAI microservices.
- **[API Reference](docs/api_reference.md)** — FastAPI endpoint documentation.

---

## 🎯 Use Case: Telecom SIM Box Fraud Detection

CDIE v4 addresses **SIM box fraud** — a **$3.8B/year** industry problem where fraudsters route international calls through unauthorized SIM gateways, causing massive revenue leakage for mobile network operators (MNOs).

Unlike correlation-based ML (XGBoost/SHAP) that identifies **what is associated**, CDIE discovers **what causes what** — enabling operators to make provably correct interventions with guaranteed impact estimates and 95% confidence intervals.

| Approach | Output | Risk |
|---|---|---|
| **Correlation-Based ML** | "CDR volume correlates with fraud" | Wrong interventions (Simpson's Paradox) |
| **CDIE Causal AI** | "Tightening fraud policy → 18% reduction in revenue leakage (95% CI [14%, 22%])" | Doubly-robust, refutation-validated |

> **Validation Note**: Current causal impact claims are verified against a **12-node synthetic telecom benchmark**. We are awaiting real-world validation against production datasets (e.g., GSMA Fraud Intelligence).

---

## 🏗 System Architecture (3 OPEA Components)

```
┌─────────────────────────────────────────────────────────────────────┐
│  OPEA GenAIComps Layer (3 Microservices)                            │
│  ┌──────────────────┐  ┌───────────────────────────────┐           │
│  │  TGI Backend      │──│  OPEA TextGen Microservice     │          │
│  │  Intel/neural-chat│  │  /v1/chat/completions          │          │
│  │  -7b-v3-3         │  │  (OpenAI compatible)           │          │
│  └──────────────────┘  └──────────────┬────────────────┘           │
│  ┌──────────────────┐  ┌──────────────┴────────────────┐           │
│  │  TEI Embedding    │  │  TEI Reranking                 │          │
│  │  BAAI/bge-base    │  │  BAAI/bge-reranker-base        │          │
│  │  -en-v1.5 (768d)  │  │  Cross-encoder precision       │          │
│  └──────────────────┘  └───────────────────────────────┘           │
└──────────────────────────────────────┬──────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────┐
│  CDIE v4 Engine                                                     │
│                                                                     │
│  OFFLINE PIPELINE                    ONLINE API                     │
│  ┌──────────────┐  ┌──────────┐    ┌──────────────────────────┐   │
│  │ SCM DataGen  │→ │ CATL     │→   │ FastAPI + Safety Map     │   │
│  │ (12-node DAG)│  │ (4 tests)│    │ Lookup + KS-Staleness    │   │
│  ├──────────────┤  ├──────────┤    ├──────────────────────────┤   │
│  │ GFCI + PCMCI+│→ │ DoWhy    │→   │ Intent Parser + RAG      │   │
│  │ Discovery    │  │ Refute   │    │ (TEI Embed → Rerank →    │   │
│  ├──────────────┤  ├──────────┤    │  OPEA LLM Briefing)      │   │
│  │ LinearDML    │→ │ MAPIE    │→   ├──────────────────────────┤   │
│  │ (EconML)     │  │ (CI)     │    │ Prescriptive Engine      │   │
│  └──────────────┘  └──────────┘    │ + HITL Edge Rejection    │   │
│                                     └──────────────────────────┘   │
│  FRONTENDS:  Streamlit (Legacy) │ Next.js + React Flow (Production)│
└─────────────────────────────────────────────────────────────────────┘
```

| Phase | Pipeline | Output |
|-------|----------|--------|
| **Offline** | Data → CATL → GFCI → Granger → DoWhy → LinearDML → MAPIE → Benchmarks | Safety Map (SQLite) |
| **Online** | Query → Intent → TEI Embed → TEI Rerank → Safety Map | Map Lookup: < 200ms |
| **Synthesis** | Safety Map Evidence → RAG Document Db → OPEA LLM TextGen | Full Report: 2-8s (CPU) |

---

## ⚡ One-Click Setup (< 10 minutes)

### Prerequisites

- **Docker** and **Docker Compose** installed
- **HuggingFace token** (for model downloads)
- **Hardware**: 64GB RAM, 4-core CPU (Intel preferred), GPU optional

### Setup

```bash
# 1. Clone repository
git clone https://github.com/Manas8114/Agent.git && cd Agent

# 2. Configure environment
cp .env.example .env
# Edit .env → set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# 3. One-click deploy (Linux/Mac)
chmod +x setup.sh && ./setup.sh

# 3. One-click deploy (Windows)
setup.cmd
```

This starts **7 containers**: TGI → OPEA TextGen → TEI Embedding → TEI Reranking → Offline Pipeline → FastAPI → Streamlit UI.

### Manual Setup (without Docker)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Recommended: keep runtime state on a local writable disk
export CDIE_RUNTIME_DIR=/tmp/cdie-runtime
# Windows PowerShell:
# $env:CDIE_RUNTIME_DIR="C:\cdie-runtime"

# Run the offline causal discovery pipeline
python -m cdie.pipeline.run_pipeline

# Start the API server
python -m uvicorn cdie.api.main:app --host 0.0.0.0 --port 8000

# Start the Next.js frontend (production)
cd frontend && npm install && npm run dev
```

### Runtime Storage Recommendation

- `data/` stores canonical artifacts such as `safety_map.json`
- `CDIE_RUNTIME_DIR` stores runtime SQLite mirrors and drift history
- avoid OneDrive/network-synced folders for `CDIE_RUNTIME_DIR`

See [`docs/DEPLOYMENT_READINESS.md`](docs/DEPLOYMENT_READINESS.md) for the deployment checklist.

---

## 🔌 OPEA Integration Details (3 Components)

| # | OPEA Component | Docker Image | Model | Port | Purpose |
|---|---|---|---|---|---|
| 1 | **LLM TextGen** | `opea/llm-textgen:latest` | Intel/neural-chat-7b-v3-3 | 9000 | OPEA Causal Intelligence Reports |
| 2 | **TEI Embedding** | `ghcr.io/huggingface/text-embeddings-inference:cpu-latest` | BAAI/bge-base-en-v1.5 | 6006 | 768-dim semantic vector retrieval |
| 3 | **TEI Reranking** | `ghcr.io/huggingface/text-embeddings-inference:cpu-latest` | BAAI/bge-reranker-base | 8808 | Cross-encoder passage re-ranking |

**RAG Pipeline Flow:**

1. **Embed** — User query → TEI Embedding → dense 768-dim vector
2. **Retrieve** — Cosine similarity against pre-embedded telecom fraud playbooks
3. **Rerank** — Top candidates → TEI Reranking → cross-encoder precision scoring
4. **Generate** — OPEA LLM TextGen → structured "OPEA Causal Intelligence Report"
5. **Fallback** — OPEA → OpenAI → Template (graceful degradation at every layer)

---

## 📊 Expected Outcomes

When fully deployed, CDIE v4 produces:

| Output | Description |
|---|---|
| **Causal DAG** | Interactive 12-node directed acyclic graph showing telecom fraud causal mechanisms |
| **Effect Estimates** | Doubly-robust ATE with 95% conformal prediction intervals for every causal edge |
| **Refutation Validation** | 3-test suite (placebo, random confounder, data subset) verifying each causal claim |
| **OPEA Intelligence Reports** | Markdown-formatted executive briefings combining causal evidence + RAG playbook advice |
| **Prescriptive Recommendations** | Ranked interventions sorted by estimated causal impact with confidence bounds |
| **Benchmark Results** | SACHS/ALARM precision, F1, SHD scores against academic ground truth |

### Demo Queries (Telecom Domain)

- *"What happens if SIM box fraud attempts increase by 30%?"*
- *"What if we tighten fraud policy strictness by 20%?"*
- *"Why did revenue leakage volume increase?"*
- *"When does a change in SIM box fraud affect revenue leakage?"*

---

## ⚠️ Limitations & Open Challenges

This project bridges causal inference and generative AI, but several open problems remain before production use:

1. **Synthetic Validation**: Empirical results rest on a synthetic telecommunications Data Generating Process (DGP). It requires validation against raw, noisy operator datasets (e.g., GSMA).
2. **Heterogeneous Effects (CATE)**: The system computes robust Average Treatment Effects (ATE), but does not yet localize heterogeneous effects (CATE) to specific subscriber cohorts (e.g., prepaid vs. postpaid).
3. **Static Causal DAG Identifiability**: We assume the 12-node SCM graph is fully identifiable. If real-world operator data introduces unmeasured confounders violating ignorability, the system currently lacks latent-variable identifiability guarantees.
4. **Online Updating**: Fast <200ms queries rely on a pre-computed "Safety Map". The static causal graph cannot yet dynamically re-calculate causal mechanisms online without queuing a full offline batch pipeline run.

---

## 🖥️ Intel Hardware Optimization

All containers are configured with Intel AMX/AVX-512 optimizations:

```yaml
environment:
  - DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  - KMP_AFFINITY=granularity=fine,compact,1,0
  - KMP_BLOCKTIME=1
  - OMP_NUM_THREADS=4
```

Run the Intel benchmark:

```bash
# Automated hardware benchmark
python benchmarks/intel_hardware_benchmark.py

# Or via API endpoint
curl http://localhost:8000/benchmark/hardware
```

---

## 🔬 Key Technologies

| Category | Technology | Purpose |
|---|---|---|
| **OPEA** | opea/llm-textgen, TEI Embedding, TEI Reranking | GenAI microservice layer |
| **Causal Discovery** | GFCI (causal-learn) | Structure learning with latent confounders |
| **Temporal Causality** | Granger (statsmodels) | Time-lagged causal discovery |
| **Causal Inference** | DoWhy | 3-test refutation suite |
| **Effect Estimation** | LinearDML (EconML) | Doubly-robust treatment effects |
| **Uncertainty** | MAPIE | Distribution-free conformal prediction |
| **Benchmarking** | SACHS + ALARM | Academic ground-truth validation |
| **Frontend** | Next.js + React Flow | Interactive causal graph + dashboard |
| **Backend** | FastAPI | Sub-200ms online query API |
| **Hardware** | Intel AMX/AVX-512 | Optimized inference across all services |

---

## 📁 Project Structure

```
├── setup.sh / setup.cmd       # One-click deployment scripts
├── docker-compose.yml         # 7-service orchestration
├── LICENSE                    # MIT License
├── TECHNICAL_REPORT.md        # 2-page architecture report
├── requirements.txt           # Python dependencies
├── data/
│   └── telecom_playbooks.json # RAG knowledge base (10 playbooks)
├── cdie/
│   ├── pipeline/              # Offline Causal Discovery
│   │   ├── run_pipeline.py    # Orchestrator (8 steps)
│   │   ├── data_generator.py  # 12-node SCM simulation
│   │   ├── catl.py            # Causal Assumption Transparency Layer
│   │   ├── gfci_discovery.py  # GFCI + PC fallback
│   │   ├── pcmci_temporal.py  # PCMCI+ + Granger cross-val
│   │   ├── refutation.py      # DoWhy 3-test refutation
│   │   ├── estimation.py      # LinearDML + MAPIE
│   │   ├── benchmarks.py      # SACHS + ALARM evaluation
│   │   └── safety_map.py      # Safety Map SQLite writer
│   ├── api/                   # Online FastAPI
│   │   ├── main.py            # Endpoints + benchmarks
│   │   ├── lookup.py          # Safety Map + KS-staleness
│   │   ├── intent_parser.py   # Query classification
│   │   ├── rag.py             # OPEA TEI + LLM RAG engine
│   │   └── models.py          # Pydantic schemas
│   └── ui/                    # Legacy Streamlit frontend
│       └── app.py
├── frontend/                  # Production Next.js UI
│   └── src/components/
│       ├── Dashboard.tsx       # Main dashboard + Prescriptive Mode
│       └── CausalGraph.tsx     # React Flow interactive graph
└── benchmarks/
    └── intel_hardware_benchmark.py  # Intel AMX/AVX-512 benchmark
```

---

## 🏆 Performance Benchmarks

```bash
# Safety Map lookup latency
curl http://localhost:8000/benchmark/latency

# OPEA TEI Embedding + Reranking benchmark
curl http://localhost:8000/benchmark/embedding

# Intel hardware detection (AMX/AVX-512)
curl http://localhost:8000/benchmark/hardware

# Full system info (all 3 OPEA components)
curl http://localhost:8000/info
```

---

## 🤝 Open-Source Contribution

See [OPEA_CONTRIBUTION.md](OPEA_CONTRIBUTION.md) for a ready-to-file GitHub issue proposing a **Causal AI GenAIComp** for the OPEA ecosystem.

---

## 📝 Third-Party Dependencies

All dependencies are MIT or Apache 2.0 compatible. Key libraries:

| Library | License | Purpose |
|---|---|---|
| causal-learn | MIT | GFCI causal discovery |
| dowhy | MIT | Causal inference & refutation |
| econml | MIT | LinearDML effect estimation |
| statsmodels | BSD-3 | Granger temporal discovery |
| mapie | BSD-3 | Conformal prediction |
| fastapi | MIT | Backend API |
| openai | MIT | OPEA TextGen client |
| react-markdown | MIT | Frontend markdown rendering |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

**Copyright (c) 2025 CDIE v4 — Causal Decision Intelligence Engine**
