"""
CDIE v4 — Integration Tests for FastAPI Endpoints
Tests all API endpoints using FastAPI TestClient to ensure no 500 errors.
"""

from fastapi.testclient import TestClient  # type: ignore
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cdie.api.main import app  # type: ignore

client = TestClient(app, raise_server_exceptions=False)


# ─────────────────────── Health & Info Endpoints ───────────────────────


class TestHealthEndpoints:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]
        assert "memory_mb" in data

    def test_metadata_returns_200(self):
        response = client.get("/metadata")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "n_scenarios" in data

    def test_info_returns_200(self):
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "engine" in data
        assert "opea_components" in data
        assert "intel_optimization" in data

    def test_demo_queries_returns_200(self):
        response = client.get("/demo-queries")
        assert response.status_code == 200
        assert isinstance(response.json(), (list, dict))


# ─────────────────────── Query Endpoint ───────────────────────


class TestQueryEndpoint:
    def test_query_empty_body(self):
        response = client.post("/query", json={"query": ""})
        assert response.status_code in [400, 422]

    def test_query_missing_field(self):
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_query_no_500_on_valid_input(self):
        """Critical regression test: ensures no AttributeError on safety_map access."""
        response = client.post("/query", json={"query": "What happens if SIM fraud increases?"})
        # Should be 200 (if Safety Map loaded) or 503 (if not loaded in test env)
        # Must NEVER be 500
        assert response.status_code != 500, f"Got 500 Internal Server Error: {response.text}"

    def test_query_unrecognized_variable(self):
        response = client.post("/query", json={"query": "the weather is nice today"})
        # Should be 422 (unrecognized) or 503 (safety map not loaded)
        # Must NEVER be 500
        assert response.status_code != 500


# ─────────────────────── Graph & Benchmark Endpoints ───────────────────────


class TestDataEndpoints:
    def test_graph_endpoint(self):
        response = client.get("/graph")
        # 200 if loaded, 503 if not
        assert response.status_code in [200, 503]
        assert response.status_code != 500

    def test_benchmark_endpoint(self):
        response = client.get("/benchmark")
        assert response.status_code in [200, 503]
        assert response.status_code != 500

    def test_catl_endpoint(self):
        response = client.get("/catl")
        assert response.status_code in [200, 503]
        assert response.status_code != 500

    def test_xgboost_endpoint(self):
        response = client.get("/xgboost")
        assert response.status_code in [200, 503]
        assert response.status_code != 500

    def test_temporal_endpoint(self):
        response = client.get("/temporal")
        assert response.status_code in [200, 503]
        assert response.status_code != 500


# ─────────────────────── Benchmark Performance Endpoints ───────────────────────


class TestBenchmarkEndpoints:
    def test_benchmark_hardware(self):
        response = client.get("/benchmark/hardware")
        assert response.status_code == 200
        data = response.json()
        assert "cpu" in data
        assert "intel_features" in data

    def test_benchmark_latency(self):
        response = client.get("/benchmark/latency")
        assert response.status_code in [200, 503]
        assert response.status_code != 500

    def test_benchmark_embedding(self):
        response = client.get("/benchmark/embedding")
        assert response.status_code == 200
        data = response.json()
        assert "embedding_provider" in data

    def test_benchmark_performance(self):
        response = client.get("/benchmark/performance")
        assert response.status_code == 200
        data = response.json()
        # Check for either naming convention
        has_safety = "safety_map_lookup_ms" in data or "safety_map_lookup" in data
        has_latency = "end_to_end_latency_ms" in data or "end_to_end_latency" in data
        assert has_safety or has_latency or "memory_mb" in data


# ─────────────────────── Prescribe Endpoint ───────────────────────


class TestPrescribeEndpoint:
    def test_prescribe_not_loaded(self):
        response = client.post("/prescribe", json={
            "target": "NetworkLoad",
            "maximize": False,
            "limit": 3
        })
        # 200 if loaded, 503 if not; never 500
        assert response.status_code != 500

    def test_prescribe_missing_target(self):
        response = client.post("/prescribe", json={})
        assert response.status_code == 422


# ─────────────────────── Expert Correction Endpoint ───────────────────────


class TestExpertCorrectionEndpoint:
    def test_expert_correct_valid(self):
        response = client.post("/expert/correct", json={
            "from_node": "A",
            "to_node": "B",
            "action": "add"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_expert_correct_invalid_action(self):
        response = client.post("/expert/correct", json={
            "from_node": "A",
            "to_node": "B",
            "action": "invalid_action"
        })
        assert response.status_code == 422
