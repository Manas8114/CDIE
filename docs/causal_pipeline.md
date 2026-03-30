# 🧬 Causal Pipeline: Deep Dive

The **Offline Causal Pipeline** in CDIE v4 is the core engine for causal discovery, estimation, and validation. This document provides a step-by-step breakdown of how the pipeline processes raw telecom data to generate a **Safety Map**.

---

## 🔁 8-Step Pipeline Workflow

### 1. Data Generation & Ingestion
- **Synthetic Mode**: Generates 5,000 samples from a 12-node telecom SCM with known ground-truth edges.
- **Import Mode**: (Optional) Ingests real-world CSV/SQL data from network operators.

### 2. CATL (Causal Assumption Transparency Layer)
Causal AI relies on strong assumptions. CATL performs statistical tests to verify:
- **Faithfulness**: Ensure observed independencies map to the causal graph.
- **Sufficiency**: Check if there's evidence of unobserved confounders (Latent Variables).
- **Positivity & Overlap**: Ensure every sample has a non-zero probability of treatment.

### 3. Causal Discovery (GFCI)
CDIE uses **Greedy Fast Causal Inference (GFCI)** from the `causal-learn` library.
- **PAG (Partial Ancestral Graph)**: Discovery handles latent confounders by representing them as bi-directed edges.
- **Prior Injection**: Domain expertise (e.g., `FraudAttempts` → `RevenueLeakage`) is injected as hard priors to resolve PAG ambiguities into a **Directed Acyclic Graph (DAG)**.

### 4. Granger Temporal Discovery
Time-lagged causality is crucial for telecom (e.g., "Policy Change at T" → "Revenue Change at T+2").
- **statsmodels**: Uses Granger Causality cross-validation to detect time-lagged causal links.
- **Temporal Consistency**: Confirms findings across multiple lagged horizons to ensure stability.

### 5. DoWhy Refutation Suite
Every discovered causal edge must pass 3 rigorous tests to be considered "Validated":
1.  **Placebo Treatment**: Replace the treatment with a random variable; the effect should drop to zero.
2.  **Random Common Cause**: Add a random confounder; the effect estimate should remain stable.
3.  **Data Subset**: Re-estimate the effect on a subset of data; the impact should be consistent.

### 6. Causal Estimation (LinearDML)
CDIE uses **Doubly Robust Machine Learning** (DML) via `EconML`.
- **DML Formula**: Estimates the conditional average treatment effect (CATE).
- **LinearDML**: Remaining valid even if either the outcome model (Y|X) or treatment model (T|X) is slightly misspecified.

### 7. Uncertainty Quantification (MAPIE)
We provide **95% Confidence Intervals** for every causal estimate.
- **Conformal Prediction**: Uses MAPIE for distribution-free CI generation.
- **Reliability**: Ensures operators know the "Worst Case" and "Best Case" impact of a policy change.

### 8. Safety Map Storage
The results of the validated causal effects are hashed (SHA-256) and stored in a versioned **Safety Map**.
- **Staleness Check**: Uses the **Kolmogorov-Smirnov (KS) test** to detect data drift between training and inference time.
