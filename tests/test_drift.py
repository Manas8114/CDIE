from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from cdie.api.drift import DriftAnalyzer  # type: ignore


def test_drift_json_snapshot_roundtrip(tmp_path):
    analyzer = DriftAnalyzer(
        db_path=tmp_path / "missing.db",
        history_dir=tmp_path / "drift_history",
    )

    summary = analyzer.save_snapshot(
        edges=[("A", "B"), ("B", "C")],
        ate_map={"A->B": 0.5, "B->C": -0.25},
        metadata={"snapshot_status": "json-fallback"},
    )

    assert summary["id"] == 1
    assert summary["storage_backend"].startswith("json")

    timeline = analyzer.get_timeline()
    assert len(timeline) == 1
    assert timeline[0]["status"] == "json-fallback"

    snapshot = analyzer.get_snapshot(1)
    assert snapshot is not None
    assert snapshot["n_edges"] == 2

    history = analyzer.get_edge_history("A", "B")
    assert len(history) == 1
    assert history[0]["present"] is True
    assert history[0]["ate"] == 0.5
