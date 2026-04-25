"""
Presentation helpers for the Streamlit command center.

These helpers keep UI logic testable and make the displayed evidence respond to
the active query instead of relying on mostly static copy.
"""

from __future__ import annotations

from collections import deque
from typing import Any


def compute_validation_summary(query_result: dict[str, Any]) -> dict[str, Any]:
    ref = query_result.get('refutation_status') or {}
    if not isinstance(ref, dict):
        ref = {}

    items = [
        ('Placebo Treatment', ref.get('placebo', 'NOT_TESTED')),
        ('Random Confounder', ref.get('confounder', 'NOT_TESTED')),
        ('Data Subset Stability', ref.get('subset', 'NOT_TESTED')),
    ]
    score = int(sum(1 for _, status in items if status == 'PASS') * 100 // len(items))
    return {'score': score, 'items': items}


def compute_assumption_rows(catl_data: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key, payload in catl_data.items():
        if key.startswith('_') or not isinstance(payload, dict):
            continue
        rows.append(
            {
                'label': key.replace('_', ' ').title(),
                'status': payload.get('status', 'UNKNOWN'),
                'tooltip': str(payload.get('tooltip', '')),
            }
        )
    return rows


def derive_causal_path(graph_data: dict[str, Any], source: str, target: str) -> dict[str, set[str]]:
    """Return highlighted nodes/edges for the shortest visible path."""
    edges = graph_data.get('edges', [])
    if not isinstance(edges, list) or not source or not target:
        return {'nodes': set(), 'edges': set()}

    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = str(edge.get('from', edge.get('source', '')))
        tgt = str(edge.get('to', edge.get('target', '')))
        if src and tgt:
            adjacency.setdefault(src, []).append(tgt)

    queue: deque[tuple[str, list[str]]] = deque([(source, [source])])
    visited = {source}
    while queue:
        node, path = queue.popleft()
        if node == target:
            path_edges = {f'{path[i]}->{path[i + 1]}' for i in range(len(path) - 1)}
            return {'nodes': set(path), 'edges': path_edges}
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, [*path, neighbor]))

    direct = f'{source}->{target}'
    return {'nodes': {source, target}, 'edges': {direct}}


def compute_structural_reliability(query_result: dict[str, Any], benchmark_data: dict[str, Any]) -> dict[str, Any]:
    effect = query_result.get('effect') or {}
    point = abs(float(effect.get('point_estimate', 0) or 0))
    lower = float(effect.get('ci_lower', 0) or 0)
    upper = float(effect.get('ci_upper', 0) or 0)
    ci_width = abs(upper - lower)
    ci_ratio = ci_width / point if point else 99.0

    validation = compute_validation_summary(query_result)
    score = float(validation['score']) * 0.45

    if query_result.get('confidence_label') in ('VALIDATED', 'HIGH'):
        score += 20
    elif query_result.get('confidence_label') == 'UNPROVEN':
        score += 4
    else:
        score += 10

    if query_result.get('match_type') == 'exact':
        score += 15
    elif query_result.get('match_type') == 'nearest':
        score += 8
    else:
        score += 3

    if ci_ratio <= 0.6:
        score += 10
    elif ci_ratio <= 1.0:
        score += 5

    ks_stat = float(query_result.get('ks_statistic', 0) or 0)
    score += 10 if ks_stat < 0.2 else 3

    bench = benchmark_data.get('own_scm', {}) if isinstance(benchmark_data, dict) else {}
    if isinstance(bench, dict):
        score += max(0.0, min(bench.get('f1', 0.0) * 5, 5))

    score = max(0, min(int(round(score)), 100))

    if score >= 80:
        headline = 'High structural reliability'
    elif score >= 55:
        headline = 'Moderate structural reliability'
    else:
        headline = 'Low structural reliability'

    details = (
        f'{query_result.get("source", "Source")} -> {query_result.get("target", "Target")} | '
        f'match={query_result.get("match_type", "unknown")} | '
        f'CI ratio={ci_ratio:.2f} | KS={ks_stat:.3f}'
    )
    return {'score': score, 'headline': headline, 'details': details}


def build_correlation_story(query_result: dict[str, Any]) -> dict[str, str]:
    source = str(query_result.get('source', 'the intervention'))
    target = str(query_result.get('target', 'the outcome'))
    point = float((query_result.get('effect') or {}).get('point_estimate', 0) or 0)
    direction = 'increases' if point >= 0 else 'reduces'
    return {
        'wrong_ai': (
            'Correlation-only systems chase whichever metric moves nearby, even when '
            'that variable is just a passenger on the same trend.'
        ),
        'right_ai': (
            f'CDIE isolates {source} as the actionable driver and estimates how it '
            f'{direction} {target} by {abs(point):.2f} units.'
        ),
        'insight': (
            f'The difference is not just prediction quality. It is intervention quality: '
            f'changing {source} is modeled as a do-operator, not as a coincidental pattern.'
        ),
    }


def format_cate_rows(cate_segments: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seg in cate_segments:
        if not isinstance(seg, dict):
            continue
        risk = str(seg.get('risk_level', 'Low'))
        risk_badge = '🔴 Critical' if risk == 'Critical' else '🟠 High' if risk == 'High' else '🟢 Low'
        rows.append(
            {
                'Segment': str(seg.get('segment', '?')),
                'Impact': f'{float(seg.get("ate", 0) or 0):.3f}',
                'CI': f'[{float(seg.get("ci_lower", 0) or 0):.3f}, {float(seg.get("ci_upper", 0) or 0):.3f}]',
                'Risk': risk_badge,
            }
        )
    return rows
