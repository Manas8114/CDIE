# CDIE v5 Migration Guide (v4 → v5)

CDIE v5 introduces significant architectural improvements, primarily transitioning from a runtime causal engine to an **Offline Pipeline + Online Safety Map** architecture.

## 1. Structural Changes
- **Pipeline Architecture**: Causal discovery and estimation now happen offline. The results are stored in a `Safety Map` (SQLite/JSON) for sub-millisecond lookups.
- **Data Directory**: Raw datasets should now be placed in `data/`, and the pipeline will generate `safety_map.db`.

## 2. New Causal Primitives
- **GFCI (Greedy Fast Causal Inference)**: Replaces simple DAG discovery with a more robust algorithm that accounts for unobserved confounders and selection bias.
- **CATL (Causal Audit & Test Library)**: A new suite of automated refutation tests (Placebo, Confounder, Subset) is applied to every edge.

## 3. API Changes
- **Endpoint Evolution**: `/api/query` now returns a `QueryResponse` containing `drift_detected` and `kl_divergence` metrics.
- **Batch Processing**: New `/api/query/batch` endpoint allows for bulk causal inference on large datasets.

## 4. Observability & Monitoring
- **Structured Logging**: Replaced all stdout `print` calls with `structlog`-backed JSON logging.
- **Drift Detection**: Integrated KS-test and KL-divergence to monitor for staleness in the Safety Map.

## 5. UI Enhancements
- **Glassmorphism Design**: Completely redesigned the dashboard using a premium dark-mode aesthetic.
- **Audit Ribbon**: Added a detailed audit trail for every query, including Safety Hash and structural reliability scores.

## Breaking Changes
- `cdie.pipeline.causal_engine` has been refactored into `cdie.pipeline.gfci_discovery` and `cdie.pipeline.estimation`.
- The `REDIS_URL` environment variable is now required for production-grade rate limiting.
