"""
CDIE v4 — Offline Pipeline Orchestrator
Runs all pipeline components in sequence to produce the Safety Map.
"""

import sys
import time
import traceback
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from cdie.pipeline.data_generator import (
    generate_scm_data,
    preprocess_data,
    save_data,
    generate_ground_truth_dag,
    VARIABLE_NAMES,
    GROUND_TRUTH_EDGES,
    DATA_DIR,
)
from cdie.pipeline.catl import run_catl
from cdie.pipeline.gfci_discovery import run_discovery
from cdie.pipeline.pcmci_temporal import run_temporal_discovery
from cdie.pipeline.refutation import run_refutation
from cdie.pipeline.estimation import run_estimation
from cdie.pipeline.benchmarks import run_benchmarks
from cdie.pipeline.safety_map import build_safety_map, save_safety_map


def run_pipeline(df=None, output_dir: Path = None, dag=None):
    """Execute the complete offline pipeline."""
    start_time = time.time()
    out = output_dir or DATA_DIR

    print("=" * 70)
    print("  CDIE v4/v5 — Offline Pipeline")
    print("  Causal Decision Intelligence Engine")
    print("=" * 70)

    # Step 1: Data Generation / Ingestion
    print("\n" + "─" * 50)
    print("STEP 1/8: Data Generation / Ingestion")
    print("─" * 50)
    if df is None:
        dag = generate_ground_truth_dag()
        df = generate_scm_data()
        df, preprocess_report = preprocess_data(df)
        save_data(df, dag, out)
        print("[Pipeline] Using synthesized data generator.")
    else:
        df, preprocess_report = preprocess_data(df)
        if dag is None:
            dag = generate_ground_truth_dag()  # Fallback for now
        save_data(df, dag, out)
        print("[Pipeline] Using ingested custom dataset.")

    # Step 2: CATL — Assumption Testing
    print("\n" + "─" * 50)
    print("STEP 2/8: Causal Assumption Transparency Layer (CATL)")
    print("─" * 50)
    catl_results = run_catl(df, VARIABLE_NAMES)

    # Step 3: GFCI Causal Discovery
    print("\n" + "─" * 50)
    print("STEP 3/8: GFCI Causal Discovery")
    print("─" * 50)

    # Load dynamic priors if extracted via OPEA
    dynamic_priors = None
    priors_file = out / "extracted_priors.json"
    if priors_file.exists():
        try:
            import json

            with open(priors_file, "r") as f:
                dynamic_priors = json.load(f)
            print(f"[Pipeline] Loaded {len(dynamic_priors)} dynamic OPEA priors.")
        except Exception as e:
            print(f"[Pipeline] WARNING: Failed to load dynamic priors: {e}")

    discovery_results = run_discovery(df, VARIABLE_NAMES, dynamic_priors=dynamic_priors)

    # Step 4: Granger Temporal Discovery
    print("\n" + "─" * 50)
    print("STEP 4/8: Granger Temporal Discovery")
    print("─" * 50)
    temporal_results = run_temporal_discovery(df, VARIABLE_NAMES)

    # Step 5: DoWhy Refutation
    print("\n" + "─" * 50)
    print("STEP 5/8: DoWhy Refutation Tests")
    print("─" * 50)
    refutation_results = run_refutation(
        df, discovery_results["map_dag"], VARIABLE_NAMES
    )

    # Check for all-quarantined edge case
    if len(refutation_results["validated_edges"]) == 0:
        print("[CRITICAL] All edges quarantined — no validated causal links found")
        print("[CRITICAL] Safety Map will be empty; consider re-running with different parameters")
        refutation_results["validated_edges"] = []  # Keep empty, don't fall back to ground truth
        # Optionally: save the empty result for analysis
        with open(DATA_DIR / "all_edges_quarantined.json", "w") as f:
            json.dump({
                "quarantined_edges": refutation_results["quarantined_edges"],
                "tests": refutation_results["edge_results"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }, f, indent=2)
        print("[INFO] Saved analysis of quarantined edges to data/all_edges_quarantined.json")

    # Step 6: Effect Estimation (LinearDML + MAPIE)
    print("\n" + "─" * 50)
    print("STEP 6/8: LinearDML + MAPIE Estimation")
    print("─" * 50)
    estimation_results = run_estimation(
        df, refutation_results["validated_edges"], VARIABLE_NAMES
    )

    # Step 7: Benchmark Evaluation
    print("\n" + "─" * 50)
    print("STEP 7/8: SACHS + ALARM Benchmarks")
    print("─" * 50)
    discovered_edge_tuples = list(discovery_results["map_dag"].edges())
    benchmark_results = run_benchmarks(
        discovered_edge_tuples, GROUND_TRUTH_EDGES, VARIABLE_NAMES
    )

    # Step 8: Safety Map Assembly
    print("\n" + "─" * 50)
    print("STEP 8/8: Safety Map Generation")
    print("─" * 50)
    safety_map = build_safety_map(
        data=df,
        estimation_results=estimation_results,
        refutation_results=refutation_results,
        catl_results=catl_results,
        temporal_results=temporal_results,
        benchmark_results=benchmark_results,
        discovery_results=discovery_results,
    )

    filepath, sha256 = save_safety_map(safety_map, out)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Safety Map: {filepath}")
    print(f"  SHA-256: {sha256[:32]}...")
    print(f"  Scenarios: {len(safety_map['scenarios'])}")
    print(f"  Validated Edges: {len(refutation_results['validated_edges'])}")
    print(f"  Quarantined Edges: {len(refutation_results['quarantined_edges'])}")
    print("=" * 70)

    return safety_map


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"\n[FATAL] Pipeline failed: {e}")
        traceback.print_exc()
        sys.exit(1)
