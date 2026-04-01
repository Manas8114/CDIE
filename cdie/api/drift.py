"""
CDIE v5 — Causal Drift Analyzer
Compares DAG snapshots across pipeline runs to detect structural drift.
Supports: timeline queries, pairwise comparison, edge-level ATE history.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from cdie.pipeline.data_generator import DATA_DIR


class DriftAnalyzer:
    """Analyzes causal drift across pipeline runs using versioned DAG snapshots."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or (DATA_DIR / "safety_map.db"))

    def _ensure_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dag_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    edges_json TEXT NOT NULL,
                    ate_map_json TEXT NOT NULL,
                    n_edges INTEGER,
                    algorithm TEXT,
                    metadata_json TEXT
                )
            """)

    def save_snapshot(
        self,
        edges: List[tuple],
        ate_map: Dict[str, float],
        algorithm: str = "GFCI",
        metadata: Optional[Dict] = None,
    ):
        """Save a DAG snapshot after a pipeline run."""
        self._ensure_table()
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        edges_json = json.dumps([{"source": s, "target": t} for s, t in edges])
        ate_json = json.dumps(ate_map)
        meta_json = json.dumps(metadata or {})

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO dag_history 
                (timestamp, edges_json, ate_map_json, n_edges, algorithm, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (now, edges_json, ate_json, len(edges), algorithm, meta_json),
            )
            # Keep only last 50 snapshots
            conn.execute("""
                DELETE FROM dag_history WHERE id NOT IN (
                    SELECT id FROM dag_history ORDER BY id DESC LIMIT 50
                )
            """)

    def get_timeline(self) -> List[Dict[str, Any]]:
        """Return list of all DAG snapshots with timestamps."""
        self._ensure_table()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, timestamp, n_edges, algorithm FROM dag_history ORDER BY id DESC"
            ).fetchall()
        return [
            {"id": r[0], "timestamp": r[1], "n_edges": r[2], "algorithm": r[3]}
            for r in rows
        ]

    def get_snapshot(self, snapshot_id: int) -> Optional[Dict[str, Any]]:
        """Return a full snapshot by ID."""
        self._ensure_table()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, timestamp, edges_json, ate_map_json, n_edges, algorithm, metadata_json FROM dag_history WHERE id=?",
                (snapshot_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "timestamp": row[1],
            "edges": json.loads(row[2]),
            "ate_map": json.loads(row[3]),
            "n_edges": row[4],
            "algorithm": row[5],
            "metadata": json.loads(row[6]),
        }

    def compare_snapshots(self, id_from: int, id_to: int) -> Dict[str, Any]:
        """Compare two DAG snapshots and produce a diff."""
        snap_from = self.get_snapshot(id_from)
        snap_to = self.get_snapshot(id_to)

        if not snap_from or not snap_to:
            return {"error": "One or both snapshots not found."}

        edges_from = {(e["source"], e["target"]) for e in snap_from["edges"]}
        edges_to = {(e["source"], e["target"]) for e in snap_to["edges"]}

        new_edges = edges_to - edges_from
        removed_edges = edges_from - edges_to
        stable_edges = edges_from & edges_to

        ate_from = snap_from["ate_map"]
        ate_to = snap_to["ate_map"]

        ate_changes = []
        for s, t in stable_edges:
            key = f"{s}->{t}"
            old_ate = ate_from.get(key, 0)
            new_ate = ate_to.get(key, 0)
            if old_ate != 0:
                pct_change = ((new_ate - old_ate) / abs(old_ate)) * 100
            else:
                pct_change = 0
            ate_changes.append(
                {
                    "source": s,
                    "target": t,
                    "ate_before": old_ate,
                    "ate_after": new_ate,
                    "change_pct": round(pct_change, 1),
                    "status": "strengthened"
                    if new_ate > old_ate
                    else "weakened"
                    if new_ate < old_ate
                    else "stable",
                }
            )

        ate_changes.sort(key=lambda x: abs(x["change_pct"]), reverse=True)

        return {
            "from": {
                "id": id_from,
                "timestamp": snap_from["timestamp"],
                "n_edges": snap_from["n_edges"],
            },
            "to": {
                "id": id_to,
                "timestamp": snap_to["timestamp"],
                "n_edges": snap_to["n_edges"],
            },
            "new_edges": [{"source": s, "target": t} for s, t in new_edges],
            "removed_edges": [{"source": s, "target": t} for s, t in removed_edges],
            "stable_edges": len(stable_edges),
            "ate_changes": ate_changes,
            "summary": {
                "added": len(new_edges),
                "removed": len(removed_edges),
                "stable": len(stable_edges),
                "strengthened": sum(
                    1 for c in ate_changes if c["status"] == "strengthened"
                ),
                "weakened": sum(1 for c in ate_changes if c["status"] == "weakened"),
            },
        }

    def get_edge_history(self, source: str, target: str) -> List[Dict[str, Any]]:
        """Return ATE history for a specific edge across all snapshots."""
        self._ensure_table()
        history = []

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, timestamp, edges_json, ate_map_json FROM dag_history ORDER BY id ASC"
            ).fetchall()

        key = f"{source}->{target}"
        for row in rows:
            edges = json.loads(row[2])
            ate_map = json.loads(row[3])
            edge_present = any(
                e["source"] == source and e["target"] == target for e in edges
            )
            history.append(
                {
                    "snapshot_id": row[0],
                    "timestamp": row[1],
                    "present": edge_present,
                    "ate": ate_map.get(key, None),
                }
            )

        return history
