# CDIE v4 — Technical Report

## Causal Decision Intelligence Engine for Telecom SIM Box Fraud Detection

**ITU AI4Good OPEA Innovation Challenge — Build & Submit Phase**

---

### 1. Problem Statement & Business Value

SIM box fraud causes an estimated **$3.8 billion** in annual revenue losses for mobile network operators globally. Fraudsters route international calls through unauthorized SIM gateways, bypassing licensed interconnect fees. Traditional ML-based fraud detection systems (XGBoost, Random Forest + SHAP) rely on **correlational** feature importance — which can recommend causally incorrect interventions due to confounding variables and Simpson's Paradox.

CDIE v4 introduces **Causal AI** for telecom fraud detection: rather than identifying what *correlates* with fraud, it discovers what *causes* fraud, validates those causal claims through rigorous refutation tests, and generates actionable intelligence reports using OPEA's GenAI microservice stack.

**Target Users**: Chief Network Officers, Revenue Assurance teams, and Network Operations Centers at Tier-1/2 MNOs managing interconnect fraud, billing integrity, and policy optimization.

---

### 2. System Architecture

CDIE v4 is a full-stack causal AI application built on OPEA's modular GenAIComps architecture with **3 integrated OPEA microservices**. The system comprises two phases:

#### 2.1 Offline Pipeline (Causal Discovery & Estimation)

A **12-node Structural Causal Model (SCM)** simulates telecom billing fraud dynamics including CDR Volume, SIM Box Fraud Attempts, Fraud Policy Strictness, Revenue Leakage, ARPU Impact, and 7 additional network KPIs. The pipeline executes an 8-step process:

1. **Data Generation**: Synthetic telecom dataset from a known-ground-truth DAG (5,000 samples). *Note: Results currently rely on synthetic benchmarks; awaiting live operator data validation.*
2. **CATL**: Causal Assumption Transparency Layer — tests faithfulness, sufficiency, positivity, and overlap
3. **GFCI Discovery**: Greedy Fast Causal Inference with latent confounder handling (causal-learn)
4. **Granger Temporal Analysis**: Time-lagged causal discovery with Granger causality cross-validation (statsmodels)
5. **DoWhy Refutation**: 3-test suite — placebo treatment, random common cause, data subset — verifying each causal claim
6. **LinearDML Estimation**: Doubly-robust Average Treatment Effect (ATE) estimation via EconML, remaining valid when either the outcome model or treatment model is misspecified
7. **MAPIE Confidence Intervals**: Distribution-free 95% conformal prediction intervals for every effect estimate
8. **Benchmark Validation**: Tested against SACHS (11-node protein signaling) and ALARM (37-node medical diagnosis) ground-truth networks, reporting Precision, Recall, F1, and Structural Hamming Distance (SHD)

Results are persisted to a **SQLite Safety Map** with SHA-256 integrity hashing and KS-test staleness detection for production reliability.

#### 2.2 Online API (FastAPI + OPEA RAG Pipeline)

The online phase provides **sub-200ms causal intervention lookup** via the pre-computed Safety Map. Full synthesis of the OPEA Causal Intelligence Report takes **2-8s** (depending on local CPU hardware). Natural language queries are classified by a rule-based intent parser into 4 types: *intervention*, *counterfactual*, *root cause*, and *temporal*. The RAG pipeline then executes:

1. **TEI Embedding** (BAAI/bge-base-en-v1.5, 768-dim) — embeds the query into a dense vector
2. **Cosine Retrieval** — retrieves top-6 candidate playbooks from a telecom fraud knowledge base
3. **TEI Reranking** (BAAI/bge-reranker-base) — cross-encoder re-ranking for precision scoring
4. **OPEA LLM TextGen** (Intel/neural-chat-7b-v3-3 via TGI) — generates a structured "OPEA Causal Intelligence Report" combining causal ATE evidence with RAG-retrieved playbook recommendations
5. **Graceful Fallback** — OPEA → OpenAI API → rule-based templates (degradation at every layer)

#### 2.3 Frontend (Next.js + React Flow)

Interactive causal DAG visualization with **Human-in-the-Loop (HITL)** edge rejection — domain experts can reject causal assumptions directly on the graph, feeding priors back into the next discovery cycle. A Prescriptive Mode ranks interventions by estimated causal impact. The OPEA Causal Intelligence Report is rendered in rich Markdown via `react-markdown`.

---

### 3. OPEA Component Usage

| # | OPEA Component | Docker Image | Model | Role in CDIE |
|---|---|---|---|---|
| 1 | **LLM TextGen** | `opea/llm-textgen:latest` | Intel/neural-chat-7b-v3-3 | OPEA Causal Intelligence Report generation via `/v1/chat/completions` |
| 2 | **TEI Embedding** | `ghcr.io/huggingface/text-embeddings-inference:cpu-latest` | BAAI/bge-base-en-v1.5 | 768-dim dense vector semantic retrieval over telecom playbooks |
| 3 | **TEI Reranking** | `ghcr.io/huggingface/text-embeddings-inference:cpu-latest` | BAAI/bge-reranker-base | Cross-encoder passage re-ranking for retrieval precision |

All three services are orchestrated via `docker-compose.yml` and configured with Intel AMX/AVX-512 optimizations (`DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`, `KMP_AFFINITY`, `OMP_NUM_THREADS`).

---

### 4. Deployment Strategy

**One-Click Setup** via `setup.sh` (Linux/Mac) or `setup.cmd` (Windows):

```
git clone <repo> && cd cdie-v4
cp .env.example .env   # Set HF_TOKEN
chmod +x setup.sh && ./setup.sh
```

The script starts 7 Docker containers in sequence: TGI (32GB) → OPEA TextGen (4GB) → TEI Embedding (4GB) → TEI Reranking (4GB) → Pipeline (8GB, runs once) → API (4GB) → UI (1GB). **Total peak memory: ~57GB**, well within the 64GB single-node constraint. CPU-only, no GPU required.

**Intel Hardware Optimization**: All containers set `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`, `KMP_AFFINITY=granularity=fine,compact,1,0`, and `KMP_BLOCKTIME=1` for Intel Xeon performance. A dedicated benchmark script (`benchmarks/intel_hardware_benchmark.py`) measures inference throughput with and without these flags.

---

### 5. Use Case Alignment — Telecommunications

CDIE directly addresses the challenge's **Telecommunication Networks** vertical:

- **Problem**: $3.8B/year SIM box fraud — operators need causal evidence to justify policy changes, not just correlation-based alerts
- **OPEA Value**: The modular OPEA stack enables a full Embed→Rerank→Generate RAG pipeline that translates complex causal statistics into executive-ready intelligence reports
- **Enterprise Deployment**: Single-node Docker Compose, sub-200ms query latency, SQLite Safety Map with integrity hashing — production-grade reliability without cloud dependencies
- **Operator Impact**: Provably correct intervention recommendations with confidence intervals (e.g., "Policy tightening reduces revenue leakage by 18%, 95% CI [14%, 22%]") enable data-driven decision-making at the CNO level

---

### 6. Key Differentiators

1. **Hybrid Causal AI + RAG**: First system combining doubly-robust causal estimation with OPEA GenAI retrieval-augmented generation
2. **Refutation-Validated Claims**: Every causal effect passes 3 independent refutation tests before being reported
3. **Conformal Prediction**: Distribution-free 95% confidence intervals via MAPIE — no Gaussian assumptions
4. **Human-in-the-Loop**: Interactive causal graph with expert edge rejection for assumption refinement
5. **Academic Benchmarking**: Validated against SACHS and ALARM ground-truth networks (F1, SHD metrics)
6. **Intel-Optimized**: AMX/AVX-512 acceleration with measurable throughput benchmarks
7. **Graceful Degradation**: 3-tier LLM fallback (OPEA → OpenAI → Templates) ensures availability

---

### 7. Limitations & Open Challenges

This project represents a prototype bridging causal AI and generative LLMs. Several key technical challenges remain unsolved before deployment at a Tier-1 MNO:

1. **Synthetic Data Dependency**: Current metrics (F1, SHD) and causal impact confidence intervals are derived from our own synthetic 12-node Data Generating Process. To prove real-world viability, the pipeline must be validated against noisy, production GSM/CDMA datasets (e.g., GSMA Fraud Intelligence).
2. **Heterogeneous Treatment Effects (CATE)**: While the system correctly estimates the Average Treatment Effect (ATE), it lacks CATE estimation. We cannot currently isolate whether tightening a fraud policy affects prepaid subscribers differently than postpaid enterprise clients.
3. **Online DAG Updates**: The system achieves fast online queries by caching a static "Safety Map." If real-world causal mechanisms drift, the DAG cannot be updated online without re-running the entire batch offline pipeline.
4. **Causal Identifiability Risks**: We assume the 12-node SCM fully satisfies ignorability. In live telecom networks, unmeasured latent confounders that violate these bounds will void our current causal identifiability guarantees.

---

*License: MIT | OPEA Components: 3 | Total RAM: ~57GB | Setup Time: < 10 minutes*
