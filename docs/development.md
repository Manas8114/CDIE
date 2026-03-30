# 🛠️ Development & Contribution Guide

This guide is for developers and researchers who want to contribute to the **CDIE v4** (Causal Decision Intelligence Engine) platform. 

---

## 📂 Project Structure

```text
cdie-v4/
├── cdie/
│   ├── api/            # FastAPI backend (Fast Inference, RAG)
│   ├── pipeline/       # Offline Causal Pipeline (GFCI, EconML, DoWhy)
│   ├── ui/             # Next.js Frontend (React Flow, Charts)
│   └── tests/          # Pytest suite for causal and API logic
├── data/               # Persistent storage (Safety Map, Scenarios, PII-free data)
├── docs/               # Comprehensive documentation
├── benchmarks/         # Performance profiling (Intel Hardware benchmarks)
├── docker-compose.yml  # Multi-container orchestration
└── setup.sh            # One-click environment boostrapper
```

---

## 🧪 Testing Strategies

CDIE follows a strict **Causal Accuracy** testing protocol:

### 1. Pytest (Unit & Integration)
Run standard unit tests for data generation and API endpoints:
```bash
pytest tests/
```

### 2. Causal Ground-Truth Validation
The pipeline is regularly benchmarked against Academic structural models (SACHS protein signaling and ALARM medical diagnosis networks) to ensure the Structural Hamming Distance (SHD) and F1-score for edge discovery remain within acceptable bounds.

### 3. OPEA Mocking
During development, if you do not have access to an Intel Xeon node, you can mock the OPEA microservices in the `.env` file to use the **OpenAI fallback** or **Rule-based templates**.

---

## 🪵 Logging & Debugging

- **API Logs**: `docker logs cdie-api -f`
- **Pipeline Logs**: Check `data/pipeline_run.log` or the console output during `docker-compose up pipeline`.
- **RAG Debugging**: Set `LOG_LEVEL=DEBUG` in your `.env` to see the retrieval scores from OPEA TEI.

---

## 🗺️ Roadmap: CDIE v5
- **Federated Causal Learning**: Allowing multiple operators to share causal insights without moving raw data.
- **Auto-Priors**: Using Large Language Models (LLMs) to automatically generate `DOMAIN_PRIORS` from telecom PDFs during discovery.
- **Dynamic SCM**: Supporting streaming data for real-time causal graph updates.

---

## 📜 Feedback & Contributions
We welcome contributions through Pull Requests. Please ensure all new causal logic is accompanied by a **DoWhy Refutation** test case.
