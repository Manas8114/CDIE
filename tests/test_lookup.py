"""
CDIE v4 — Unit Tests for SafetyMapLookup
Tests the SQLite-backed Safety Map lookup, staleness checking, and prescriptions.
"""

import json
import sqlite3
import tempfile
import os
import pytest  # type: ignore
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdie.api.lookup import SafetyMapLookup  # type: ignore


@pytest.fixture
def sample_db(tmp_path):
    """Create a temporary SQLite Safety Map DB for testing."""
    db_path = tmp_path / "safety_map.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create tables matching the production schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS store (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scenarios (
            id TEXT PRIMARY KEY,
            source TEXT,
            target TEXT,
            magnitude_value REAL,
            data_payload TEXT
        )
    """)

    # Insert metadata
    cursor.execute("INSERT INTO store (key, value) VALUES (?, ?)",
                   ("sha256_hash", json.dumps("test_hash_abc123")))
    cursor.execute("INSERT INTO store (key, value) VALUES (?, ?)",
                   ("version", json.dumps("4.0.0")))
    cursor.execute("INSERT INTO store (key, value) VALUES (?, ?)",
                   ("created_at", json.dumps("2026-03-25T00:00:00Z")))
    cursor.execute("INSERT INTO store (key, value) VALUES (?, ?)",
                   ("n_variables", json.dumps(3)))
    cursor.execute("INSERT INTO store (key, value) VALUES (?, ?)",
                   ("graph", json.dumps({
                       "nodes": [{"id": "A", "label": "A"}, {"id": "B", "label": "B"}],
                       "edges": [{"from": "A", "to": "B", "tests": [
                           {"status": "PASS"}, {"status": "PASS"}, {"status": "PASS"}
                       ]}]
                   })))
    cursor.execute("INSERT INTO store (key, value) VALUES (?, ?)",
                   ("training_distributions", json.dumps({
                       "SIMBoxFraudAttempts": {"mean": 50, "sample_values": list(range(100))}
                   })))

    # Insert test scenarios
    scenarios = [
        ("A__B__increase_10", "A", "B", 0.1, {
            "id": "A__B__increase_10",
            "effect": {"point_estimate": 0.35, "ci_lower": 0.1, "ci_upper": 0.6,
                       "confidence_level": 0.95, "ate_used": 0.35, "intervention_amount": 10},
            "causal_path": "A → B",
            "refutation_status": "VALIDATED",
            "cate_by_segment": {},
        }),
        ("A__B__increase_20", "A", "B", 0.2, {
            "id": "A__B__increase_20",
            "effect": {"point_estimate": 0.7, "ci_lower": 0.3, "ci_upper": 1.1,
                       "confidence_level": 0.95, "ate_used": 0.7, "intervention_amount": 20},
            "causal_path": "A → B",
            "refutation_status": "VALIDATED",
            "cate_by_segment": {},
        }),
        ("C__B__increase_30", "C", "B", 0.3, {
            "id": "C__B__increase_30",
            "effect": {"point_estimate": -0.5, "ci_lower": -0.9, "ci_upper": -0.1,
                       "confidence_level": 0.95, "ate_used": -0.5, "intervention_amount": 30},
            "causal_path": "C → B",
            "refutation_status": "UNPROVEN",
            "cate_by_segment": {},
        }),
    ]
    for sid, src, tgt, mag, payload in scenarios:
        cursor.execute(
            "INSERT INTO scenarios (id, source, target, magnitude_value, data_payload) VALUES (?, ?, ?, ?, ?)",
            (sid, src, tgt, mag, json.dumps(payload))
        )

    conn.commit()
    conn.close()
    return str(db_path)


# ─────────────────────── Unit Tests: SafetyMapLookup ───────────────────────


class TestSafetyMapLookupInit:
    def test_init_no_path(self):
        lookup = SafetyMapLookup()
        assert lookup.db_path is None
        assert not lookup.is_loaded()

    def test_init_with_valid_path(self, sample_db):
        lookup = SafetyMapLookup(sample_db)
        assert lookup.is_loaded()
        assert lookup.sha256_hash == "test_hash_abc123"

    def test_init_with_invalid_path(self, tmp_path):
        lookup = SafetyMapLookup(str(tmp_path / "nonexistent.db"))
        assert not lookup.is_loaded()


class TestSafetyMapLookupLoad:
    def test_load_db(self, sample_db):
        lookup = SafetyMapLookup()
        result = lookup.load(sample_db)
        assert result is True
        assert lookup.is_loaded()

    def test_load_json_suffix_converts_to_db(self, sample_db):
        json_path = sample_db.replace(".db", ".json")
        lookup = SafetyMapLookup()
        lookup.load(json_path)
        assert lookup.db_path == sample_db

    def test_load_nonexistent(self, tmp_path):
        lookup = SafetyMapLookup()
        result = lookup.load(str(tmp_path / "missing.db"))
        assert result is False


class TestFindScenario:
    def test_find_existing_scenario(self, sample_db):
        lookup = SafetyMapLookup(sample_db)
        result = lookup.find_scenario("A", "B", "increase_10")
        assert result is not None
        assert result["id"] == "A__B__increase_10"
        assert result["effect"]["point_estimate"] == 0.35

    def test_find_scenario_no_magnitude(self, sample_db):
        lookup = SafetyMapLookup(sample_db)
        result = lookup.find_scenario("A", "B")
        assert result is not None

    def test_find_nonexistent_scenario(self, sample_db):
        lookup = SafetyMapLookup(sample_db)
        result = lookup.find_scenario("X", "Y", "increase_10")
        assert result is None

    def test_find_scenario_not_loaded(self):
        lookup = SafetyMapLookup()
        result = lookup.find_scenario("A", "B")
        assert result is None


class TestFindBestScenario:
    def test_exact_match(self, sample_db):
        lookup = SafetyMapLookup(sample_db)
        scenario, is_exact = lookup.find_best_scenario("A", "B", 10)
        assert scenario is not None
        assert is_exact is True
        assert scenario["id"] == "A__B__increase_10"

    def test_interpolation(self, sample_db):
        lookup = SafetyMapLookup(sample_db)
        scenario, is_exact = lookup.find_best_scenario("A", "B", 15)
        assert scenario is not None
        assert is_exact is False

    def test_no_match(self, sample_db):
        lookup = SafetyMapLookup(sample_db)
        scenario, is_exact = lookup.find_best_scenario("X", "Y", 10)
        assert scenario is None
        assert is_exact is False

    def test_not_loaded(self):
        lookup = SafetyMapLookup()
        scenario, is_exact = lookup.find_best_scenario("A", "B", 10)
        assert scenario is None
        assert is_exact is False


class TestFindPrescriptions:
    def test_find_prescriptions_maximize(self, sample_db):
        lookup = SafetyMapLookup(sample_db)
        results = lookup.find_prescriptions("B", limit=3, maximize=True)
        assert isinstance(results, list)
        # C->B scenario is UNPROVEN, should be excluded
        for r in results:
            assert r.get("refutation_status") != "UNPROVEN"

    def test_find_prescriptions_not_loaded(self):
        lookup = SafetyMapLookup()
        results = lookup.find_prescriptions("B")
        assert results == []


class TestCheckStaleness:
    def test_staleness_not_loaded(self):
        lookup = SafetyMapLookup()
        result = lookup.check_staleness("A", 50.0)
        assert result["warning"] is False

    def test_staleness_insufficient_data(self, sample_db):
        lookup = SafetyMapLookup(sample_db)
        result = lookup.check_staleness("SIMBoxFraudAttempts", 50.0)
        assert result["warning"] is False
        assert result.get("reason") == "insufficient_data"


class TestGetters:
    def test_get_graph(self, sample_db):
        lookup = SafetyMapLookup(sample_db)
        graph = lookup.get_graph()
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) == 2

    def test_get_metadata(self, sample_db):
        lookup = SafetyMapLookup(sample_db)
        meta = lookup.get_metadata()
        assert meta["version"] == "4.0.0"
        assert meta["n_scenarios"] == 3
        assert meta["sha256_hash"] == "test_hash_abc123"

    def test_get_graph_not_loaded(self):
        lookup = SafetyMapLookup()
        graph = lookup.get_graph()
        assert graph == {"nodes": [], "edges": []}

    def test_get_metadata_not_loaded(self):
        lookup = SafetyMapLookup()
        meta = lookup.get_metadata()
        assert meta["n_scenarios"] == 0
