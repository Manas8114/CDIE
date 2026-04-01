# 🤖 OPEA Integration: Intel-Optimized GenAI

CDIE v4 (Causal Decision Intelligence Engine) is built on the **Open Platform for Enterprise AI (OPEA)** modular microservice architecture. By using OPEA's GenAIComps, the system translates complex causal statistics into human-readable executive intelligence reports with sub-200ms retrieval latency.

---

## 🏗️ 3-Component OPEA Stack

The system orchestrates three OPEA-compliant microservices in a unified **RAG (Retrieval-Augmented Generation)** pipeline.

| # | OPEA Component | Docker Image | Model | Role in CDIE |
|---|---|---|---|---|
| 1 | **LLM TextGen** | `opea/llm-textgen:latest` | Intel/neural-chat-7b-v3-3 | Generates structured Causal Intelligence Reports. |
| 2 | **TEI Embedding** | `ghcr.io/huggingface/text-embeddings-inference:cpu-latest` | BAAI/bge-base-en-v1.5 | 768-dim dense vector embedding of telecom playbooks. |
| 3 | **TEI Reranking** | `ghcr.io/huggingface/text-embeddings-inference:cpu-latest` | BAAI/bge-reranker-base | Cross-encoder reranking for maximum retrieval precision. |

---

## ⚡ Intel Hardware Optimizations
CDIE v4 is optimized for **Intel Xeon** performance using the following acceleration flags in the OPEA containers:
- **AMX/AVX-512**: Enabled via `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.
- **Parallelism**: Configured with `OMP_NUM_THREADS` and `KMP_AFFINITY=granularity=fine,compact,1,0` for efficient core pinning.
- **Latency**: Sub-200ms end-to-end query time (Lookup + Retrieval + Generation).

---

## 🛠️ The 5-Step RAG Pipeline

### 1. Intent Parsing
Natural language queries ("What happens if SIM box attempts increase by 30%?") are parsed into structured causal variables (`source`, `target`, `magnitude`).

### 2. Safety Map Retrieval
The system performs a high-speed lookup in the **Safety Map** (SQLite/JSON) to find the pre-validated causal effect estimate (ATE), confidence intervals (CI), and refutation status.

### 3. TEI Semantic Embedding
The OPEA **TEI Embedding** microservice converts the user's query into a dense vector (768-dim) and performs a cosine-similarity search over a knowledge base of **Telecom Fraud Playbooks**.

### 4. TEI Cross-Encoder Reranking
To ensure the most relevant context is provided to the LLM, the **TEI Reranker** cross-scores the top-6 candidate playbooks, prioritizing actions that specifically mitigate the identified causal risk. The necessity of this reranking stage is grounded in the embedding precision ceiling (Weller et al., 2025): single-vector bi-encoders lose token-level interaction signals — negations, causal qualifiers, domain-specific phrasing — that cross-encoders recover through joint query–document attention.

### 5. GenAI Report Generation
Finally, the OPEA **LLM TextGen** microservice (running **Intel/neural-chat-7b-v3-3** via TGI) synthesizes the causal evidence and the retrieved playbooks into a professional Markdown report for the operator.

---

## 🛡️ Robustness & Fallbacks
CDIE v4 ensures 100% availability through a **3-tier graceful degradation** strategy:
1.  **Tier 1 (OPEA)**: Primary path using local Intel-optimized microservices.
2.  **Tier 2 (OpenAI)**: Fallback to the OpenAI API if local OPEA services are offline.
3.  **Tier 3 (Template)**: Fallback to rule-based Markdown templates if all LLM providers are unavailable.
