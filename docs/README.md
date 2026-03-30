# CDIE v4 Documentation Index

Welcome to the comprehensive documentation for the **Causal Decision Intelligence Engine (CDIE) v4**. This system is an Intel-optimized, OPEA-integrated platform for causal inference and decision support in the telecommunications industry.

---

## 🗺️ Documentation Map

- **[Introduction](introduction.md)** — Project background, Causal AI vs. ML, and the SIM Box Fraud use case.
- **[Architecture](architecture.md)** — High-level system design, encompassing the Offline Causal Pipeline and Online RAG API.
- **[Installation & Setup](installation.md)** — Prerequisites, environment configuration, and Docker deployment.
- **[Causal Pipeline](causal_pipeline.md)** — Deep dive into Discovery (GFCI), Estimation (LinearDML), and Validation (DoWhy).
- **[OPEA Integration](opea_integration.md)** — How we use Intel's Open Platform for Enterprise AI (LLM, Embeddings, Reranking).
- **[API Reference](api_reference.md)** — Detailed documentation for FastAPI endpoints and integration patterns.
- **[Development Guide](development.md)** — Folder structure, testing strategies, and contribution guidelines.

---

## 🚀 Quick Start

If you are already configured with Docker and an Intel Xeon CPU:

```bash
git clone <repo-url> && cd cdie-v4
cp .env.example .env   # Add your HF_TOKEN
./setup.sh             # Linux/Mac
# OR
setup.cmd              # Windows
```

---

## 🏗️ Core Technologies

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Causal Discovery** | `causal-learn` (GFCI/PC) | Structure learning from observational data |
| **Estimation** | `EconML` (LinearDML) | Doubly-robust treatment effect estimation |
| **Validation** | `DoWhy` | Causal refutation and robustness testing |
| **Confidence** | `MAPIE` | 95% Distribution-free conformal prediction |
| **GenAI Stack** | **OPEA** (LLM, TEI) | RAG-based intelligence report generation |
| **Frontend** | `Next.js` + `React Flow` | Interactive DAG visualization |
