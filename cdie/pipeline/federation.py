"""
CDIE v5 — Federated Causal Learning (Foundation)
Enables sharing causal structure (PAG edges + confidence) between operators
without exposing raw CDR data. Supports: export, import, weighted aggregation.
"""

import time
from typing import Any

from cdie.pipeline.data_generator import VARIABLE_NAMES


class PAGSerializer:
    """Exports/imports the discovered PAG as a portable, data-free JSON."""

    @staticmethod
    def export_pag(
        edges: list[tuple[str, str]],
        ate_map: dict[str, float],
        operator_id: str = 'default',
        algorithm: str = 'GFCI',
    ) -> dict[str, Any]:
        """
        Export the current operator's PAG (no raw data included).
        This is safe to share with other operators or a central aggregator.
        """
        return {
            'format': 'CDIE_PAG_v1',
            'operator_id': operator_id,
            'exported_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'algorithm': algorithm,
            'variable_schema': VARIABLE_NAMES,
            'edges': [
                {
                    'source': src,
                    'target': tgt,
                    'ate': ate_map.get(f'{src}->{tgt}'),
                    'confidence': min(1.0, abs(ate_map.get(f'{src}->{tgt}', 0)) * 2 + 0.5),
                }
                for src, tgt in edges
            ],
            'n_edges': len(edges),
            'data_shared': False,
        }

    @staticmethod
    def validate_pag(pag: dict[str, Any]) -> tuple[bool, str]:
        """Validate an imported PAG has the correct structure."""
        if pag.get('format') != 'CDIE_PAG_v1':
            return False, f'Unknown format: {pag.get("format")}'
        if 'edges' not in pag or not isinstance(pag['edges'], list):
            return False, "Missing or invalid 'edges' field"
        for edge in pag['edges']:
            if edge.get('source') not in VARIABLE_NAMES:
                return False, f'Unknown source variable: {edge.get("source")}'
            if edge.get('target') not in VARIABLE_NAMES:
                return False, f'Unknown target variable: {edge.get("target")}'
        return True, 'Valid'


class FederatedAggregator:
    """Aggregates PAGs from multiple operators via weighted edge voting."""

    @staticmethod
    def aggregate_pags(
        pags: list[dict[str, Any]],
        vote_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Weighted vote aggregation of multiple operator PAGs.
        An edge is included if it appears in >= vote_threshold fraction of PAGs.
        ATE is the weighted average across operators that have the edge.
        """
        n_operators = len(pags)
        if n_operators == 0:
            return {'error': 'No PAGs provided'}

        edge_votes: dict[str, dict[str, Any]] = {}

        for pag in pags:
            operator = pag.get('operator_id', 'unknown')
            for edge in pag.get('edges', []):
                key = f'{edge["source"]}->{edge["target"]}'
                if key not in edge_votes:
                    edge_votes[key] = {
                        'source': edge['source'],
                        'target': edge['target'],
                        'operators': [],
                        'ates': [],
                        'confidences': [],
                    }
                edge_votes[key]['operators'].append(operator)
                if edge.get('ate') is not None:
                    edge_votes[key]['ates'].append(edge['ate'])
                edge_votes[key]['confidences'].append(edge.get('confidence', 0.5))

        # Apply voting threshold
        consensus_edges = []
        operator_specific = []

        for _key, info in edge_votes.items():
            vote_fraction = len(info['operators']) / n_operators
            avg_ate = sum(info['ates']) / len(info['ates']) if info['ates'] else None
            avg_conf = sum(info['confidences']) / len(info['confidences'])

            edge_result = {
                'source': info['source'],
                'target': info['target'],
                'vote_fraction': round(vote_fraction, 2),
                'n_votes': len(info['operators']),
                'operators': info['operators'],
                'avg_ate': round(avg_ate, 4) if avg_ate is not None else None,
                'avg_confidence': round(avg_conf, 3),
            }

            if vote_fraction >= vote_threshold:
                edge_result['classification'] = 'consensus'
                consensus_edges.append(edge_result)
            else:
                edge_result['classification'] = 'operator_specific'
                operator_specific.append(edge_result)

        consensus_edges.sort(key=lambda x: x['avg_confidence'], reverse=True)
        operator_specific.sort(key=lambda x: x['avg_confidence'], reverse=True)

        return {
            'format': 'CDIE_FEDERATED_v1',
            'aggregated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'n_operators': n_operators,
            'vote_threshold': vote_threshold,
            'consensus_edges': consensus_edges,
            'operator_specific_edges': operator_specific,
            'summary': {
                'total_unique_edges': len(edge_votes),
                'consensus_count': len(consensus_edges),
                'operator_specific_count': len(operator_specific),
            },
        }
