"""
CDIE v5 — Batch Benchmark Runner

Runs all benchmark evaluations (SACHS, ALARM, own SCM ground truth)
and writes a timestamped JSON report to benchmarks/latest.json.

Usage:
    python scripts/batch_benchmark.py
    python scripts/batch_benchmark.py --output benchmarks/run_2026-04-24.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_all_benchmarks() -> dict:
    """Execute all benchmark suites and return a consolidated report."""
    from cdie.pipeline.benchmarks import run_benchmarks
    from cdie.config import GROUND_TRUTH_EDGES, VARIABLE_NAMES

    print('=' * 60)
    print('CDIE v5 — Batch Benchmark Runner')
    print('=' * 60)

    start = time.time()

    # Try to load the Safety Map's discovered edges for own-SCM evaluation
    discovered_edges = []
    try:
        from cdie.config import DATA_DIR
        import json as _json

        json_path = DATA_DIR / 'safety_map.json'
        if json_path.exists():
            with open(json_path, encoding='utf-8') as f:
                sm = _json.load(f)
            graph_edges = sm.get('graph', {}).get('edges', [])
            discovered_edges = [(e['from'], e['to']) for e in graph_edges]
            print(f'[batch] Loaded {len(discovered_edges)} edges from Safety Map for own-SCM evaluation.')
        else:
            print('[batch] No Safety Map found — own-SCM benchmark skipped.')
    except Exception as exc:
        print(f'[batch] Could not load Safety Map edges: {exc}')

    report = run_benchmarks(
        discovered_edges=discovered_edges or None,
        ground_truth_edges=GROUND_TRUTH_EDGES if discovered_edges else None,
        variable_names=VARIABLE_NAMES if discovered_edges else None,
    )

    elapsed = round(time.time() - start, 2)

    summary = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'elapsed_seconds': elapsed,
        'benchmarks': report,
        'summary': {
            name: {
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0),
                'shd': metrics.get('shd', 0),
                'status': metrics.get('status', 'UNKNOWN'),
            }
            for name, metrics in report.items()
        },
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Run CDIE benchmark suite.')
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'benchmarks' / 'latest.json',
        help='Output path for the benchmark report JSON.',
    )
    args = parser.parse_args()

    report = run_all_benchmarks()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)

    print()
    print(f'✅ Benchmark report written to: {args.output}')
    print()
    print('Summary:')
    for name, metrics in report['summary'].items():
        print(
            f'  {name:15s}  P={metrics["precision"]:.3f}  '
            f'R={metrics["recall"]:.3f}  '
            f'F1={metrics["f1"]:.3f}  '
            f'SHD={metrics["shd"]:3d}  '
            f'[{metrics["status"]}]'
        )
    print(f'\nTotal time: {report["elapsed_seconds"]}s')


if __name__ == '__main__':
    main()
