import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdie.api.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_hte_report_schema():
    """Verify that the HTE report endpoint returns valid JSON structure or 404/503 if missing."""
    response = client.get('/hte/report')

    # If the pipeline was run, it should be 200. If not, 404 based on our implementation.
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)
        # Check if at least one edge has the expected keys
        if len(data) > 0:
            first_edge = list(data.values())[0]
            assert 'ate' in first_edge or 'cate_by_segment' in first_edge
    else:
        assert response.status_code in [404, 503]


def test_hte_heatmap_mime():
    """Verify that the heatmap endpoint returns a PNG image."""
    response = client.get('/hte/heatmap')

    if response.status_code == 200:
        assert response.headers['content-type'] == 'image/png'
        assert len(response.content) > 0
    else:
        assert response.status_code in [404, 503]


def test_hte_invalid_endpoint():
    """Verify standard 404 for non-existent HTE routes."""
    response = client.get('/hte/invalid')
    assert response.status_code == 404


if __name__ == '__main__':
    pytest.main([__file__])
