"""
CDIE v5 — HTE Visualization Module
Generates Causal Tree and Segment Heatmap outputs from ForestDRLearner results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_segment_heatmap(
    hte_results: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    """
    Generate a 2-D heatmap PNG showing CATE effect magnitudes
    for each (edge, segment) combination.
    Returns the path to the saved PNG or None on failure.
    """
    try:
        import matplotlib
        import numpy as np

        matplotlib.use('Agg')
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        edges = list(hte_results.keys())
        if not edges:
            return None

        # Collect segment names across all edges
        all_segments: list[str] = []
        for data in hte_results.values():
            for seg in data.get('cate_by_segment', {}):
                if seg not in all_segments:
                    all_segments.append(seg)
        if not all_segments:
            return None

        matrix = np.full((len(edges), len(all_segments)), np.nan)
        for i, edge in enumerate(edges):
            cbs = hte_results[edge].get('cate_by_segment', {})
            for j, seg in enumerate(all_segments):
                entry = cbs.get(seg, {})
                if isinstance(entry, dict) and entry.get('ate') is not None:
                    matrix[i, j] = entry['ate']

        fig, ax = plt.subplots(figsize=(max(6, len(all_segments) * 1.4), max(4, len(edges) * 0.5 + 1)))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')

        norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(matrix), vcenter=0, vmax=np.nanmax(matrix))
        im = ax.imshow(matrix, cmap='RdYlGn', norm=norm, aspect='auto')

        ax.set_xticks(range(len(all_segments)))
        ax.set_xticklabels(all_segments, rotation=30, ha='right', color='white', fontsize=8)
        ax.set_yticks(range(len(edges)))
        edge_labels = [e.replace('->', ' → ') for e in edges]
        ax.set_yticklabels(edge_labels, color='white', fontsize=7)

        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.set_label('CATE (ATE)', color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

        ax.set_title('Heterogeneous Treatment Effects by Subscriber Segment', color='white', fontsize=11, pad=12)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

        plt.tight_layout()
        out_path = output_dir / 'hte_segment_heatmap.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'[HTEViz] Segment heatmap saved → {out_path}')
        return out_path
    except Exception as e:
        print(f'[HTEViz] Heatmap generation skipped: {e}')
        return None


def generate_effect_distribution(
    hte_results: dict[str, Any],
    output_dir: Path,
    top_n: int = 5,
) -> Path | None:
    """
    Generate a bar chart comparing ATE vs segment-level CATE for the top-N edges.
    Highlights which subscriber groups respond most/least to interventions.
    """
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Pick the top-N edges by absolute ATE
        ranked = sorted(
            [(k, v) for k, v in hte_results.items() if isinstance(v.get('ate'), dict)],
            key=lambda x: abs(x[1]['ate'].get('ate', 0)),
            reverse=True,
        )[:top_n]
        if not ranked:
            return None

        fig, axes = plt.subplots(1, len(ranked), figsize=(4 * len(ranked), 5), sharey=False)
        fig.patch.set_facecolor('#0d1117')
        if len(ranked) == 1:
            axes = [axes]

        palette = ['#00e5a0', '#f97316', '#38bdf8', '#a78bfa', '#fb7185']

        for ax, (edge_key, data) in zip(axes, ranked, strict=False):
            ax.set_facecolor('#161b22')
            ate_val = data['ate'].get('ate', 0)
            segments = data.get('cate_by_segment', {})
            names = ['Population\nAverage'] + [
                str(s) for s in segments if isinstance(segments[s], dict) and segments[s].get('ate') is not None
            ]
            values = [ate_val] + [
                segments[s]['ate']
                for s in segments
                if isinstance(segments[s], dict) and segments[s].get('ate') is not None
            ]

            colors = [palette[i % len(palette)] for i in range(len(names))]
            bars = ax.bar(names, values, color=colors, edgecolor='#0d1117', linewidth=0.8, width=0.6)
            ax.axhline(0, color='#555', linewidth=0.8, linestyle='--')
            ax.set_title(edge_key.replace('->', ' →'), color='white', fontsize=8, pad=8, wrap=True)
            ax.tick_params(colors='white', labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')
            ax.set_facecolor('#161b22')
            for bar, val in zip(bars, values, strict=False):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + (0.01 if val >= 0 else -0.03),
                    f'{val:.3f}',
                    ha='center',
                    va='bottom' if val >= 0 else 'top',
                    color='white',
                    fontsize=6,
                )

        fig.suptitle('CATE vs Population ATE by Subscriber Segment', color='white', fontsize=11, y=1.01)
        plt.tight_layout()
        out_path = output_dir / 'hte_effect_distribution.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'[HTEViz] Effect distribution chart saved → {out_path}')
        return out_path
    except Exception as e:
        print(f'[HTEViz] Effect distribution chart skipped: {e}')
        return None


def save_hte_report(hte_results: dict[str, Any], output_dir: Path) -> Path:
    """Save the full HTE results as a machine-readable JSON report."""
    import numpy as np

    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(i) for i in obj]
        return obj

    report_path = output_dir / 'hte_report.json'
    report_path.write_text(
        json.dumps(_convert(hte_results), indent=2, default=str),
        encoding='utf-8',
    )
    print(f'[HTEViz] HTE report saved → {report_path}')
    return report_path


def run_hte_visualization(
    hte_results: dict[str, Any],
    output_dir: Path,
) -> dict[str, str | None]:
    """Main entry point: generate all HTE visualizations + JSON report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap = generate_segment_heatmap(hte_results, output_dir)
    distribution = generate_effect_distribution(hte_results, output_dir)
    report = save_hte_report(hte_results, output_dir)

    return {
        'hte_segment_heatmap': str(heatmap) if heatmap else None,
        'hte_effect_distribution': str(distribution) if distribution else None,
        'hte_report': str(report),
    }
