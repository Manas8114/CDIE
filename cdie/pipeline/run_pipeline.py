"""
CDIE v4 — Offline Pipeline Orchestrator
Runs all pipeline components in sequence to produce the Safety Map.
"""

import contextlib
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, cast

import pandas as pd

from cdie.config import DATA_DIR, GROUND_TRUTH_EDGES, VARIABLE_NAMES
from cdie.observability import get_logger
from cdie.pipeline.benchmarks import run_benchmarks
from cdie.pipeline.catl import run_catl
from cdie.pipeline.data_generator import (
    generate_ground_truth_dag,
    generate_scm_data,
    preprocess_data,
)
from cdie.pipeline.datastore import DataStoreManager
from cdie.pipeline.estimation import run_estimation
from cdie.pipeline.gfci_discovery import run_discovery
from cdie.pipeline.hte_viz import run_hte_visualization
from cdie.pipeline.pcmci_temporal import run_temporal_discovery
from cdie.pipeline.refutation import run_refutation
from cdie.pipeline.safety_map import build_safety_map, save_safety_map

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

for stream_name in ('stdout', 'stderr'):
    stream = getattr(sys, stream_name, None)
    if stream is not None and hasattr(stream, 'reconfigure'):
        try:
            # sys.stdout/stderr are TextIOWrapper which has reconfigure
            cast(Any, stream).reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

log = get_logger(__name__)


def run_pipeline(
    df: pd.DataFrame | None = None, 
    output_dir: Path | None = None, 
    dag: Any = None
) -> dict[str, Any]:
    """Execute the complete offline pipeline."""
    start_time = time.time()
    out = output_dir or DATA_DIR

    log.info('CDIE v5 — Offline Pipeline starting')

    # Step 1: Data Generation / Ingestion
    log.info('STEP 1/8: Data Generation / Ingestion')
    store_manager = DataStoreManager()
    if df is None:
        dag = generate_ground_truth_dag()
        df = generate_scm_data()
        df, _preprocess_report = preprocess_data(df)
        store_manager.save_pipeline_data(df, dag)
        log.info('[Pipeline] Using synthesized data generator')
    else:
        df, _preprocess_report = preprocess_data(df)
        if dag is None:
            dag = generate_ground_truth_dag()  # Fallback for now
        store_manager.save_pipeline_data(df, dag)
        log.info('[Pipeline] Using ingested custom dataset')

    # Step 2: CATL — Assumption Testing
    log.info('STEP 2/8: Causal Assumption Transparency Layer (CATL)')
    catl_results = run_catl(df, VARIABLE_NAMES)

    # Step 3: GFCI Causal Discovery
    log.info('STEP 3/8: GFCI Causal Discovery')

    # Load dynamic priors if extracted via OPEA
    dynamic_priors = None
    priors_file = out / 'extracted_priors.json'
    if priors_file.exists():
        try:
            with open(priors_file) as f:
                dynamic_priors = json.load(f)
            log.info('[Pipeline] Loaded dynamic OPEA priors', n_priors=len(dynamic_priors))
        except Exception as e:
            log.warning('[Pipeline] Failed to load dynamic priors', error=str(e))

    discovery_results = run_discovery(df, VARIABLE_NAMES, dynamic_priors=dynamic_priors)

    # Step 4: Granger Temporal Discovery
    log.info('STEP 4/8: Granger Temporal Discovery')
    temporal_results = run_temporal_discovery(df, VARIABLE_NAMES)

    # Step 5: DoWhy Refutation
    log.info('STEP 5/8: DoWhy Refutation Tests')
    refutation_results = run_refutation(df, discovery_results['map_dag'], VARIABLE_NAMES)

    # Check for all-quarantined edge case
    if len(refutation_results['validated_edges']) == 0:
        log.warning('[CRITICAL] All edges quarantined — no validated causal links found')
        log.warning('[CRITICAL] Safety Map will be empty; consider re-running with different parameters')
        refutation_results['validated_edges'] = []  # Keep empty, don't fall back to ground truth
        with open(DATA_DIR / 'all_edges_quarantined.json', 'w') as f:
            json.dump(
                {
                    'quarantined_edges': refutation_results['quarantined_edges'],
                    'tests': refutation_results['edge_results'],
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                },
                f,
                indent=2,
            )
        log.info('[INFO] Saved quarantined edges analysis', path='data/all_edges_quarantined.json')

    # Step 6: Effect Estimation (LinearDML + MAPIE + HTE Discovery)
    log.info('STEP 6/8: LinearDML + MAPIE + ForestDRLearner HTE')
    estimation_results = run_estimation(df, refutation_results['validated_edges'], VARIABLE_NAMES)

    # HTE Visualization (runs after estimation, non-blocking)
    log.info('[Pipeline] Generating HTE visualizations')
    hte_viz_results = {}
    try:
        hte_viz_results = run_hte_visualization(estimation_results, out)
        n_viz = sum(1 for v in hte_viz_results.values() if v)
        log.info('[Pipeline] HTE visualizations complete', n_outputs=n_viz)
    except Exception as e:
        log.warning('[Pipeline] HTE visualization skipped', error=str(e))

    # Step 7: Benchmark Evaluation
    log.info('STEP 7/8: SACHS + ALARM Benchmarks')
    discovered_edge_tuples = list(discovery_results['map_dag'].edges())
    benchmark_results = run_benchmarks(discovered_edge_tuples, GROUND_TRUTH_EDGES, VARIABLE_NAMES)

    # Step 8: Safety Map Assembly
    log.info('STEP 8/8: Safety Map Generation')
    safety_map = build_safety_map(
        data=df,
        estimation_results=estimation_results,
        refutation_results=refutation_results,
        catl_results=catl_results,
        temporal_results=temporal_results,
        benchmark_results=benchmark_results,
        discovery_results=discovery_results,
    )

    # Final cleanup
    del df
    gc.collect()

    filepath, sha256 = save_safety_map(safety_map, out)

    # Summary
    elapsed = time.time() - start_time
    log.info(
        'PIPELINE COMPLETE',
        elapsed_seconds=round(elapsed, 1),
        safety_map=str(filepath),
        sha256_prefix=sha256[:16],
        n_scenarios=len(safety_map['scenarios']),
        validated_edges=len(refutation_results['validated_edges']),
        quarantined_edges=len(refutation_results['quarantined_edges']),
    )
    if hte_viz_results.get('hte_report'):
        log.info('[Pipeline] HTE Report saved', path=str(hte_viz_results['hte_report']))

    return safety_map


if __name__ == '__main__':
    try:
        run_pipeline()
    except Exception as e:
        log.error('[FATAL] Pipeline failed', error=str(e))
        traceback.print_exc()
        sys.exit(1)
