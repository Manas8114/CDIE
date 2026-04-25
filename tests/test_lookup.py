"""
CDIE v4 - Unit Tests for SafetyMapLookup
Tests the JSON-backed Safety Map lookup, staleness checking, and prescriptions.
"""

import json
import sys
from pathlib import Path

import pytest  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))

from cdie.api.lookup import SafetyMapLookup  # type: ignore


@pytest.fixture
def sample_map(tmp_path):
    """Create a temporary JSON Safety Map for testing."""
    json_path = tmp_path / 'safety_map.json'
    safety_map = {
        'sha256_hash': 'test_hash_abc123',
        'version': '4.0.0',
        'created_at': '2026-03-25T00:00:00Z',
        'n_variables': 3,
        'graph': {
            'nodes': [{'id': 'A', 'label': 'A'}, {'id': 'B', 'label': 'B'}],
            'edges': [{'from': 'A', 'to': 'B', 'tests': [{'status': 'PASS'}, {'status': 'PASS'}, {'status': 'PASS'}]}],
        },
        'training_distributions': {'SIMBoxFraudAttempts': {'mean': 50, 'sample_values': list(range(100))}},
        'scenarios': {
            'A__B__increase_10': {
                'id': 'A__B__increase_10',
                'source': 'A',
                'target': 'B',
                'magnitude_key': 'increase_10',
                'magnitude_value': 0.1,
                'effect': {
                    'point_estimate': 0.35,
                    'ci_lower': 0.1,
                    'ci_upper': 0.6,
                    'confidence_level': 0.95,
                    'ate_used': 0.35,
                    'intervention_amount': 10,
                },
                'causal_path': 'A -> B',
                'refutation_status': 'VALIDATED',
                'cate_by_segment': {},
            },
            'A__B__increase_20': {
                'id': 'A__B__increase_20',
                'source': 'A',
                'target': 'B',
                'magnitude_key': 'increase_20',
                'magnitude_value': 0.2,
                'effect': {
                    'point_estimate': 0.7,
                    'ci_lower': 0.3,
                    'ci_upper': 1.1,
                    'confidence_level': 0.95,
                    'ate_used': 0.7,
                    'intervention_amount': 20,
                },
                'causal_path': 'A -> B',
                'refutation_status': 'VALIDATED',
                'cate_by_segment': {},
            },
            'C__B__increase_30': {
                'id': 'C__B__increase_30',
                'source': 'C',
                'target': 'B',
                'magnitude_key': 'increase_30',
                'magnitude_value': 0.3,
                'effect': {
                    'point_estimate': -0.5,
                    'ci_lower': -0.9,
                    'ci_upper': -0.1,
                    'confidence_level': 0.95,
                    'ate_used': -0.5,
                    'intervention_amount': 30,
                },
                'causal_path': 'C -> B',
                'refutation_status': 'UNPROVEN',
                'cate_by_segment': {},
            },
        },
    }
    json_path.write_text(json.dumps(safety_map), encoding='utf-8')
    return str(json_path)


class TestSafetyMapLookupInit:
    def test_init_no_path(self):
        lookup = SafetyMapLookup()
        assert lookup.db_path is None
        assert not lookup.is_loaded()

    def test_init_with_valid_path(self, sample_map):
        lookup = SafetyMapLookup(sample_map)
        assert lookup.is_loaded()
        assert lookup.sha256_hash == 'test_hash_abc123'

    def test_init_with_invalid_path(self, tmp_path):
        lookup = SafetyMapLookup(str(tmp_path / 'nonexistent.json'))
        assert not lookup.is_loaded()


class TestSafetyMapLookupLoad:
    def test_load_json(self, sample_map):
        lookup = SafetyMapLookup()
        result = lookup.load(sample_map)
        assert result is True
        assert lookup.is_loaded()
        assert lookup.get_storage_backend() == 'json'

    def test_load_nonexistent(self, tmp_path):
        lookup = SafetyMapLookup()
        result = lookup.load(str(tmp_path / 'missing.json'))
        assert result is False


class TestFindScenario:
    def test_find_existing_scenario(self, sample_map):
        lookup = SafetyMapLookup(sample_map)
        result = lookup.find_scenario('A', 'B', 'increase_10')
        assert result is not None
        assert result['id'] == 'A__B__increase_10'
        assert result['effect']['point_estimate'] == 0.35

    def test_find_scenario_no_magnitude(self, sample_map):
        lookup = SafetyMapLookup(sample_map)
        result = lookup.find_scenario('A', 'B')
        assert result is not None

    def test_find_nonexistent_scenario(self, sample_map):
        lookup = SafetyMapLookup(sample_map)
        result = lookup.find_scenario('X', 'Y', 'increase_10')
        assert result is None

    def test_find_scenario_not_loaded(self):
        lookup = SafetyMapLookup()
        result = lookup.find_scenario('A', 'B')
        assert result is None


class TestFindBestScenario:
    def test_exact_match(self, sample_map):
        lookup = SafetyMapLookup(sample_map)
        scenario, is_exact = lookup.find_best_scenario('A', 'B', 10)
        assert scenario is not None
        assert is_exact is True
        assert scenario['id'] == 'A__B__increase_10'

    def test_interpolation(self, sample_map):
        lookup = SafetyMapLookup(sample_map)
        scenario, is_exact = lookup.find_best_scenario('A', 'B', 15)
        assert scenario is not None
        assert is_exact is False

    def test_no_match(self, sample_map):
        lookup = SafetyMapLookup(sample_map)
        scenario, is_exact = lookup.find_best_scenario('X', 'Y', 10)
        assert scenario is None
        assert is_exact is False

    def test_not_loaded(self):
        lookup = SafetyMapLookup()
        scenario, is_exact = lookup.find_best_scenario('A', 'B', 10)
        assert scenario is None
        assert is_exact is False


class TestFindPrescriptions:
    def test_find_prescriptions_maximize(self, sample_map):
        lookup = SafetyMapLookup(sample_map)
        results = lookup.find_prescriptions('B', limit=3, maximize=True)
        assert isinstance(results, list)
        for result in results:
            assert result.get('refutation_status') == 'VALIDATED'

    def test_find_prescriptions_not_loaded(self):
        lookup = SafetyMapLookup()
        results = lookup.find_prescriptions('B')
        assert results == []


class TestCheckStaleness:
    def test_staleness_not_loaded(self):
        lookup = SafetyMapLookup()
        result = lookup.check_staleness('A', 50.0)
        assert result['warning'] is False

    def test_staleness_insufficient_data(self, sample_map):
        lookup = SafetyMapLookup(sample_map)
        result = lookup.check_staleness('SIMBoxFraudAttempts', 50.0)
        assert result['warning'] is False
        assert result.get('reason') == 'insufficient_data'


class TestGetters:
    def test_get_graph(self, sample_map):
        lookup = SafetyMapLookup(sample_map)
        graph = lookup.get_graph()
        assert 'nodes' in graph
        assert 'edges' in graph
        assert len(graph['nodes']) == 2

    def test_get_metadata(self, sample_map):
        lookup = SafetyMapLookup(sample_map)
        meta = lookup.get_metadata()
        assert meta['version'] == '4.0.0'
        assert meta['n_scenarios'] == 3
        assert meta['sha256_hash'] == 'test_hash_abc123'
        assert meta['storage_backend'] == 'json'

    def test_get_graph_not_loaded(self):
        lookup = SafetyMapLookup()
        graph = lookup.get_graph()
        assert graph == {'nodes': [], 'edges': []}

    def test_get_metadata_not_loaded(self):
        lookup = SafetyMapLookup()
        meta = lookup.get_metadata()
        assert meta['n_scenarios'] == 0
