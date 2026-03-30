# OPEA Open-Source Contribution: Causal AI GenAIComp Proposal

## GitHub Issue Draft for `opea-project/GenAIComps`

> **Title:** [Feature Request] Add Causal AI / Causal Inference GenAI Component

### Motivation

Current GenAIComps focus on text generation, embedding, retrieval, and reranking —
core building blocks for RAG and chatbot applications. However, enterprise decision-making
increasingly requires **causal inference** capabilities beyond correlation-based analytics.

We have built **CDIE v4 (Causal Decision Intelligence Engine)** as part of the
ITU AI4Good OPEA Innovation Challenge. It integrates 3 OPEA components:

1. `opea/llm-textgen` — LLM Text Generation (Intel/neural-chat-7b-v3-3 via TGI)
2. TEI Embedding — BAAI/bge-base-en-v1.5 for semantic retrieval
3. TEI Reranking — BAAI/bge-reranker-base for passage re-ranking

### Proposed Component: `opea/causal-inference`

A new OPEA-compatible microservice that wraps causal discovery and estimation:

**Input API:**
```json
POST /v1/causal/discover
{
  "data": "base64-encoded CSV or reference to data source",
  "method": "GFCI",  // or "PC", "FCI", "PCMCI+"
  "domain_priors": [{"from": "A", "to": "B", "forbidden": false}],
  "significance_level": 0.05
}
```

**Output:**
```json
{
  "edges": [{"from": "A", "to": "B", "weight": 0.85}],
  "adjacency_matrix": [[0, 1], [0, 0]],
  "refutation_results": {"placebo": "PASS", "random_common_cause": "PASS"}
}
```

### Benefits to OPEA Ecosystem

1. **Enterprise Decision Support** — Goes beyond "what correlates" to "what causes"
2. **Telecom Vertical** — Native support for network KPI causal analysis
3. **Composable** — Plugs into existing OPEA pipelines as a post-processing step
4. **Intel-Optimized** — Uses Intel Extension for PyTorch (IPEX) for NumPy/scipy acceleration

### Reference Implementation

Our CDIE v4 project demonstrates this pattern:
- Repository: [link to your GitHub repo]
- Architecture: TEI Embedding → Causal Discovery → DoWhy Refutation → OPEA LLM Briefing

### How to File This

1. Go to https://github.com/opea-project/GenAIComps/issues/new
2. Title: `[Feature Request] Add Causal AI / Causal Inference GenAI Component`
3. Copy the content above (from "### Motivation" onward)
4. Label: `enhancement`, `feature-request`
5. Submit → This counts as an Open-Source Contribution (+5 bonus pts)

---

**Note:** Even filing a documentation improvement or bug report counts for bonus points.
Some easier alternatives:

- Fix a typo in any OPEA README
- Report an unclear error message you encountered
- Suggest adding a "one-click setup" example for a new vertical (telecom)
