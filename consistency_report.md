# CDIE v5 Consistency Audit Report

**Date**: 2026-04-09
**Status**: Resolved / Pending Final Verification

---

## 1. Identified Inconsistencies & Errors

### [FIXED] Frontend State Persistence Bug

* **Description**: The `hteReport` state in `scenarioStore.ts` was not cleared during a full dashboard reset or when a new query/prescription was submitted.
* **Impact**: Users would see causal nuances (heatmaps/outliers) from the previous scenario while a new one was loading.
* **Resolution**: Updated `scenarioStore.ts` to explicitly set `hteReport: null` in the `reset`, `submitQuery`, and `submitPrescription` functions.

### [NOTE] OPEA Environment Configuration

* **Description**: Initial audit suggested OPEA variables were missing from `.env.example`.
* **Correction**: Upon close inspection, `OPEA_LLM_ENDPOINT`, `OPEA_EMBEDDING_ENDPOINT`, and `OPEA_RERANKING_ENDPOINT` are already present.
* **Action**: Updated header to reflect v5 versioning.

### [OBSERVATION] HTE Report Truncation

* **Description**: `discover_heterogeneity` in `estimation.py` returns a maximum of 50 individual effects.
* **Rationale**: This protects UI performance and prevents massive JSON payloads.
* **Constraint**: If future UI requires full population distribution, this limit must be increased.

### [OBSERVATION] Variable Whitelist Scope

* **Description**: HTE context variables (`SubscriberTenureMonths`, etc.) are excluded from the main `VARIABLE_NAMES` whitelist in `data_generator.py`.
* **Rationale**: This keeps the primary GFCI graph focused on high-level network nodes. Interactions are handled as localized effects in the HTE module.

---

## 2. Technical Debt & Stabilization

| Component | Status | Debt Item |
|-----------|--------|-----------|
| Backend   | Green  | `ForestDRLearner` is non-linear but computationally expensive. |
| API       | Green  | `/hte/heatmap` returns static images; interactive SVG preferred for v6. |
| Global    | Yellow | Synthetic data dependency remains high. |

---

## 3. Next Steps for Stabilization

- [ ] Run `pytest tests/test_hte_api.py` to verify API integrity.
* [ ] Verify Dashboard UI reset behavior manually.

CDIE v5 Test Coverage Audit & Strategy
This document identifies all critical functions within the CDIE v5 codebase that require systematic testing. It categorizes them by risk level and provides a baseline of current coverage.

User Review Required
IMPORTANT

This audit focuses on causal integrity and system robustness. Some functions (like LLM-based intent parsing) require non-deterministic evaluation strategies (e.g., using a "Golden Dataset" of queries).

WARNING

High-risk logic like discover_heterogeneity and validate_schema currently have limited edge-case coverage. We should prioritize these to prevent false causal claims.

1. TIER 1: Core Causal Pipeline (High Impact)
These functions are responsible for the "intelligence" of CDIE. Any bug here results in incorrect causal insights.

estimation.py
Function Complexity Current Coverage Status
discover_heterogeneity High (CausalForest) Basic (Mock data) ✅ Partially Tested
compute_ate_dml High (Double ML) Minimal ⚠️ Refinement Needed
compute_cate Medium Minimal ⚠️ Needs Testing
run_estimation High (Orchestrator) Basic ✅ Partially Tested
add_mapie_intervals Medium (Uncertainty) None 🔴 Untested
gfci_discovery.py
Function Complexity Current Coverage Status
run_discovery High (Graph Search) Basic ✅ Partially Tested
build_map_dag Medium Basic ✅ Partially Tested
_run_pc_fallback Medium (Robustness) None 🔴 Untested
catl.py
Function Complexity Current Coverage Status
run_catl Medium Integration tests ✅ Partially Tested
test_faithfulness Medium Logic unit tests ✅ Partially Tested
test_positivity High (Propensity) None 🔴 Untested
2. TIER 2: API & Intelligence Layer (User-Facing)
These functions handle incoming user intent and deliver the results via the web dashboard.

intent_parser.py
Function Complexity Current Coverage Status
classify_query Medium (LLM/Regex) Basic ✅ Partially Tested
extract_variables Medium Basic ✅ Partially Tested
extract_magnitude Medium Basic ✅ Partially Tested
extract_entities_llm High (Non-det) Mock-only ⚠️ Needs Benchmarking
lookup.py
Function Complexity Current Coverage Status
SafetyMapLookup.find_best_scenario Medium Integration ✅ Partially Tested
check_staleness High (KS-Test) Integration ✅ Partially Tested
find_prescriptions Medium Basic ✅ Partially Tested
3. TIER 3: Data Safety & Contracts (Infrastructure)
These shield the system from bad data or adversarial injection.

schema_contract.py
Function Complexity Current Coverage Status
validate_schema High (Security) Basic ✅ Partially Tested
_detect_adversarial_injection High (Security) Limited ⚠️ Critical Guardrail
_check_timestamp_granularity Medium Integration ✅ Partially Tested
data_ingestion.py
Function Complexity Status
DataStoreManager.merge_ingested_data High (Stateful) ⚠️ Needs Stress Test
identify_data_format Medium ✅ Partially Tested
Proposed Verification Plan
Automated Tests
Unit Tests: Add tests for test_positivity, compute_ate_dml, and add_mapie_intervals.
Adversarial Tests: Create a suite of "bad data" for _detect_adversarial_injection in schema_contract.py.
Stochastic Benchmarking: Run test_interaction_recovery multiple times to ensure the seed-based recovery is stable across different noise distributions.
Manual Verification
Verify the API response latency using /benchmark/performance after implementing heavy tests to ensure no regression.
