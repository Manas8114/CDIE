# PLAN-cate-hte-module.md

## Goal

Implement a professional-grade Heterogeneous Treatment Effects (HTE/CATE) module using EconML.

## Phase 1: Context & Data Enrichment

- **Task 1.1**: Update `data_generator.py` to include more features (`SubscriberTenureMonths`, `DeviceTier`, `RegionalRiskScore`).
- **Task 1.2**: Inject non-linear interactions into the Structural Equations (SCM) to give the CATE learner something to discover.
- **Verification**: Run `data_generator.py` and verify `scm_data.csv` contains the new columns.

## Phase 2: Algorithmic Upgrade

- **Task 2.1**: Modify `estimation.py` to import `ForestDRLearner` from `econml`.
- **Task 2.2**: Replace basic `compute_cate` with a robust `ForestDRLearner` fit.
- **Task 2.3**: Enable **Feature Importance** extraction to determine which attributes drive effect heterogeneity.
- **Verification**: Unit test `estimation.py` with known injected effects.

## Phase 3: Visualization & Reporting

- **Task 3.1**: Create `cdie/pipeline/hte_viz.py`.
- **Task 3.2**: Implement `generate_causal_tree()` (using `sklearn.tree.export_graphviz` on the learner's forest outputs).
- **Task 3.3**: Implement `generate_segment_heatmap()`.
- **Verification**: Preview SVG/PNG outputs.

## Phase 4: Integration

- **Task 4.1**: Link `hte_viz.py` into the main `run_pipeline.py` flow.
- **Task 4.2**: Update `technical_report.md` with the new methodology.
- **Verification**: Complete pipeline execution with full HTE report.

## Verification Checklist

- [x] `ForestDRLearner` identifies >80% of injected interactions.
- [x] `CustomerSegment` splits work for all 12 causal edges.
- [x] Visualizations render without path/encoding errors.
- [x] README updated with CATE benchmarks.
