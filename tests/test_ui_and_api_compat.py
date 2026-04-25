import sys
from pathlib import Path

from fastapi.testclient import TestClient  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))

import cdie.api.main as api_main  # type: ignore
from cdie.ui.presentation import (  # type: ignore
    build_correlation_story,
    compute_structural_reliability,
    compute_validation_summary,
    derive_causal_path,
    format_cate_rows,
)

client = TestClient(api_main.app, raise_server_exceptions=False)


def test_post_drift_compare_is_supported(monkeypatch):
    called = {}

    def fake_compare(id_from: int, id_to: int):
        called['pair'] = (id_from, id_to)
        return {
            'from': {'id': id_from, 'timestamp': '2026-04-01', 'n_edges': 1},
            'to': {'id': id_to, 'timestamp': '2026-04-02', 'n_edges': 2},
            'new_edges': [],
            'removed_edges': [],
            'stable_edges': 1,
            'ate_changes': [],
            'summary': {
                'added': 0,
                'removed': 0,
                'stable': 1,
                'strengthened': 0,
                'weakened': 0,
            },
        }

    monkeypatch.setattr(api_main.drift_analyzer, 'compare_snapshots', fake_compare)

    response = client.post('/api/drift/compare', json={'id_from': 1, 'id_to': 2})
    assert response.status_code == 200
    assert called['pair'] == (1, 2)


def test_api_ingest_alias_routes_to_shared_handler(monkeypatch):
    async def fake_ingest(background_tasks, file):
        return {'status': 'accepted', 'filename': file.filename}

    monkeypatch.setattr(api_main, '_ingest_uploaded_file', fake_ingest)

    response = client.post(
        '/api/ingest',
        files={'file': ('demo.csv', b'col1,col2\n1,2\n', 'text/csv')},
    )
    assert response.status_code == 200
    assert response.json()['status'] == 'accepted'


def test_backtest_accepts_alias_payloads():
    response = client.post(
        '/api/backtest',
        json={
            'intervention': 'sim box fraud',
            'outcome': 'revenue leakage',
            'magnitude': 20,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['source'] == 'SIMBoxFraudAttempts'
    assert payload['target'] == 'RevenueLeakageVolume'


def test_prescribe_resolves_human_target_names(monkeypatch):
    monkeypatch.setattr(api_main.safety_map_lookup, 'is_loaded', lambda: True)
    monkeypatch.setattr(
        api_main.safety_map_lookup,
        'find_prescriptions',
        lambda target, limit, maximize: [{'target': target, 'limit': limit, 'maximize': maximize}],
    )

    response = client.post(
        '/prescribe',
        json={'target': 'revenue leakage', 'maximize': False, 'limit': 3},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['target'] == 'RevenueLeakageVolume'


def test_validation_summary_is_query_specific():
    summary = compute_validation_summary(
        {
            'refutation_status': {
                'placebo': 'PASS',
                'confounder': 'WARN',
                'subset': 'FAIL',
            }
        }
    )
    assert summary['score'] == 33
    assert summary['items'][0][0] == 'Placebo Treatment'


def test_derive_causal_path_finds_shortest_visible_route():
    graph = {
        'edges': [
            {'from': 'A', 'to': 'B'},
            {'from': 'B', 'to': 'C'},
            {'from': 'A', 'to': 'D'},
        ]
    }
    highlight = derive_causal_path(graph, 'A', 'C')
    assert highlight['nodes'] == {'A', 'B', 'C'}
    assert highlight['edges'] == {'A->B', 'B->C'}


def test_structural_reliability_changes_with_evidence():
    strong = compute_structural_reliability(
        {
            'source': 'A',
            'target': 'B',
            'match_type': 'exact',
            'confidence_label': 'VALIDATED',
            'ks_statistic': 0.01,
            'effect': {'point_estimate': 1.0, 'ci_lower': 0.8, 'ci_upper': 1.1},
            'refutation_status': {
                'placebo': 'PASS',
                'confounder': 'PASS',
                'subset': 'PASS',
            },
        },
        {'own_scm': {'f1': 0.9}},
    )
    weak = compute_structural_reliability(
        {
            'source': 'A',
            'target': 'B',
            'match_type': 'fallback',
            'confidence_label': 'UNPROVEN',
            'ks_statistic': 0.4,
            'effect': {'point_estimate': 1.0, 'ci_lower': -1.0, 'ci_upper': 2.5},
            'refutation_status': {
                'placebo': 'FAIL',
                'confounder': 'WARN',
                'subset': 'FAIL',
            },
        },
        {'own_scm': {'f1': 0.9}},
    )
    assert strong['score'] > weak['score']
    assert strong['headline'] != weak['headline']


def test_correlation_story_and_cate_rows_are_dynamic():
    story = build_correlation_story(
        {
            'source': 'FraudPolicyStrictness',
            'target': 'RevenueLeakageVolume',
            'effect': {'point_estimate': -0.42},
        }
    )
    rows = format_cate_rows(
        [
            {
                'segment': 'Enterprise',
                'ate': 0.12,
                'ci_lower': 0.05,
                'ci_upper': 0.2,
                'risk_level': 'High',
            }
        ]
    )

    assert 'FraudPolicyStrictness' in story['right_ai']
    assert 'RevenueLeakageVolume' in story['right_ai']
    assert rows[0]['Segment'] == 'Enterprise'
    assert rows[0]['Risk'] == '🟠 High'
