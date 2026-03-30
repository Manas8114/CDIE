import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdie.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_query_endpoint_missing_variable():
    response = client.post("/query", json={"query": "this is a random sentence with no variables"})
    # 422 if safety map is loaded (variable not recognized), 503 if safety map not loaded
    assert response.status_code in [422, 503]

def test_query_endpoint_valid():
    # Attempting to map to a real query test since we fixed the 500
    response = client.post("/query", json={"query": "What happens to network load if SIM box fraud increases?"})
    
    # It might return a 503 if the map isn't loaded offline in testing, 
    # but we just want to ensure it doesn't give a 500 internal server error due to code bugs.
    assert response.status_code in [200, 503]
