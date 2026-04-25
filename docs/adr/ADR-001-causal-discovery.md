# ADR-001 — Causal Discovery: GFCI + PC Fallback

**Date:** 2026-04-24  
**Status:** Accepted  
**Deciders:** CDIE v5 Architecture Team

---

## Context

CDIE requires an offline causal discovery algorithm to learn a Directed Acyclic Graph (DAG)
from telecom fraud telemetry data (12 variables, N ≈ 5 000 synthetic samples).
The discovered graph drives the Safety Map and all downstream causal effect estimates.

### Requirements

1. Handle **both directed and undirected/latent edges** (real-world data has unmeasured confounders).
2. Scale to **12 variables** in < 60 seconds on a laptop-class CPU.
3. Produce a **PAG (Partial Ancestral Graph)** that can be resolved into a DAG using domain priors.
4. Degrade gracefully: if the primary algorithm fails or times out, continue with a simpler fallback.

---

## Decision

Use **GFCI (Greedy Fast Causal Inference)** as the primary algorithm, with a **PC fallback** triggered by timeout or exception, and a final **domain-priors-only** mode when both fail.

```
GFCI (60 s timeout)
   ├── Success → extract PAG edges → resolve to MAP-DAG
   ├── Timeout → PC algorithm
   │              ├── Success → extract edges → resolve to MAP-DAG
   │              └── Failure → DOMAIN_PRIORS_ONLY mode
   └── Exception → PC algorithm (same as timeout path)
```

Domain priors (DOMAIN_PRIORS in `gfci_discovery.py`) are merged with optional OPEA-extracted priors
from `PriorExtractor` using a confidence threshold of 0.70.

---

## Alternatives Considered

| Algorithm | Pros | Cons | Why Rejected |
|-----------|------|------|--------------|
| **PC only** | Simple, well-understood | Assumes no latent confounders (causal sufficiency) | Violated in real-world telecom data |
| **GES** | Fast, score-based | No latent handling; Greedy search can miss global optima | Same limitation as PC |
| **LiNGAM** | Exact in linear non-Gaussian regime | Requires strict distributional assumptions | SCM data is mixed Gaussian |
| **DoWhy auto-discovery** | Integrated with estimation layer | Wraps PC; same latent limitation | Same limitation as PC |
| **GFCI only (no fallback)** | Consistent | Hangs indefinitely on large matrices | Timeout would break pipeline |

---

## Consequences

**Positive:**
- PAG representation handles latent confounders without requiring causal sufficiency.
- 60 s timeout prevents pipeline hangs on edge cases.
- DOMAIN_PRIORS_ONLY mode ensures the pipeline always produces a usable graph.

**Negative:**
- Two-algorithm chain adds complexity.
- GFCI result quality depends on Fisher-Z independence test, which assumes Gaussian noise.
- PAG → DAG resolution via priors is heuristic (acyclicity is enforced, not provably correct).

**Mitigation:** All graph edges are validated by 3 refutation tests (placebo, random confounder,
data subset). Edges failing refutation are quarantined before entering the Safety Map.

---

## References

- Ogarrio et al., "A Hybrid Causal Search Algorithm for Latent Variable Models" (2016)
- `cdie/pipeline/gfci_discovery.py`
- `cdie/pipeline/refutation.py`
