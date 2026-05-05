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

## 🧩 How to Add a New Causal Method

Adding a new causal discovery or estimation method to CDIE is modular and straightforward. Follow these steps:

### 1. Implement the Algorithm Wrapper
Create a new file in `cdie/pipeline/` (e.g., `fci_discovery.py` or `dml_estimation.py`).
Encapsulate your logic in a function that accepts a Pandas DataFrame and optional prior knowledge, returning a structured output.
For a discovery method, return a list of directed edges:
```python
import pandas as pd

def run_custom_discovery(df: pd.DataFrame, prior_knowledge=None) -> list[tuple[str, str]]:
    # 1. Initialize your algorithm
    # 2. Apply background knowledge
    # 3. Execute causal discovery
    # 4. Return list of directed edges
    return [('feature_A', 'feature_B'), ('feature_C', 'feature_D')]
```

### 2. Register in the Pipeline
Open `cdie/pipeline/run_pipeline.py`. Import your new method and integrate it into the `run_full_pipeline()` execution flow. You can expose it as an alternative algorithm flag or as an automated fallback.

### 3. Update the API Endpoints (If Applicable)
If your method exposes new hyperparameters (like alpha or sparsity penalties) that users should tweak, update the `CausalRequest` Pydantic model in `cdie/api/main.py` and pass those parameters into the pipeline function call.

### 4. Write a Refutation Test
We enforce test-driven causal validation. Add a test in `tests/test_causal_methods.py` (or similar) to verify your method behaves correctly against a known Ground Truth (like SACHS or ALARM). Ensure you include a **DoWhy Refutation** (e.g., random common cause, placebo treatment) to validate the robustness of your estimates.

---

## 🗺️ Roadmap: CDIE v5
- **Federated Causal Learning**: Allowing multiple operators to share causal insights without moving raw data.
- **Auto-Priors**: Using Large Language Models (LLMs) to automatically generate `DOMAIN_PRIORS` from telecom PDFs during discovery.
- **Dynamic SCM**: Supporting streaming data for real-time causal graph updates.

---

## 📜 Feedback & Contributions
We welcome contributions through Pull Requests. Please ensure all new causal logic is accompanied by a **DoWhy Refutation** test case.
