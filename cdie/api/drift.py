"""
CDIE v5 - Causal Drift Analyzer
Compares DAG snapshots across pipeline runs to detect structural drift.
Supports timeline queries, pairwise comparison, and edge-level ATE history.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cdie.pipeline.data_generator import DATA_DIR
from cdie.runtime import get_runtime_paths


class DriftAnalyzer:
    """Analyzes causal drift across pipeline runs using versioned DAG snapshots."""

    def __init__(self, db_path: Optional[Path] = None, history_dir: Optional[Path] = None):
        runtime_paths = get_runtime_paths(DATA_DIR)
        self.db_path = str(db_path or runtime_paths["runtime_db"])
        self.history_dir = Path(history_dir or runtime_paths["drift_dir"])
        self.index_path = self.history_dir / "index.json"
        self.snapshots_dir = self.history_dir / "snapshots"

    def _candidate_paths(self) -> list[Path]:
        primary = Path(self.db_path)
        project_db = DATA_DIR / "safety_map.db"
        candidates = [
            primary,
            primary.with_suffix(".db.bak"),
            project_db,
            project_db.with_suffix(".db.bak"),
        ]
        deduped: list[Path] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _connect(self) -> sqlite3.Connection:
        last_error: Exception | None = None
        for candidate in self._candidate_paths():
            if not candidate.exists():
                continue
            conn: sqlite3.Connection | None = None
            try:
                conn = sqlite3.connect(str(candidate))
                conn.execute("PRAGMA schema_version").fetchone()
                self.db_path = str(candidate)
                return conn
            except Exception as e:
                last_error = e
                if conn is not None:
                    conn.close()
        if last_error is not None:
            raise last_error
        raise FileNotFoundError(f"No readable Safety Map database found for {self.db_path}")

    def _ensure_table(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dag_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    edges_json TEXT NOT NULL,
                    ate_map_json TEXT NOT NULL,
                    n_edges INTEGER,
                    algorithm TEXT,
                    metadata_json TEXT
                )
                """
            )

    def _load_json_index(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _save_json_index(self, snapshots: list[dict[str, Any]]) -> None:
        self.history_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(snapshots, f, indent=2)

    def _snapshot_path(self, snapshot_id: int) -> Path:
        return self.snapshots_dir / f"{snapshot_id}.json"

    def _save_json_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        with open(self._snapshot_path(snapshot["id"]), "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)

    def _load_json_snapshot(self, snapshot_id: int) -> Optional[dict[str, Any]]:
        snapshot_path = self._snapshot_path(snapshot_id)
        if not snapshot_path.exists():
            return None
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def _timeline_from_json(self) -> list[dict[str, Any]]:
        snapshots = self._load_json_index()
        if not snapshots:
            return []
        return sorted(snapshots, key=lambda item: item["id"], reverse=True)

    def _sqlite_timeline(self) -> list[dict[str, Any]]:
        try:
            self._ensure_table()
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT id, timestamp, n_edges, algorithm, metadata_json FROM dag_history ORDER BY id DESC"
                ).fetchall()
        except Exception:
            return []
        timeline = []
        for row in rows:
            metadata = json.loads(row[4]) if row[4] else {}
            timeline.append(
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "n_edges": row[2],
                    "algorithm": row[3],
                    "status": metadata.get("snapshot_status", "validated"),
                    "storage_backend": metadata.get("storage_backend", "sqlite"),
                }
            )
        return timeline

    def save_snapshot(
        self,
        edges: List[tuple],
        ate_map: Dict[str, float],
        algorithm: str = "GFCI",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Save a DAG snapshot after a pipeline run, preferring JSON history."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        index = self._load_json_index()
        next_id = max((item["id"] for item in index), default=0) + 1

        metadata_copy = dict(metadata or {})
        sqlite_saved = False
        sqlite_error = ""

        edges_json = json.dumps([{"source": s, "target": t} for s, t in edges])
        ate_json = json.dumps(ate_map)

        try:
            self._ensure_table()
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO dag_history
                    (timestamp, edges_json, ate_map_json, n_edges, algorithm, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (now, edges_json, ate_json, len(edges), algorithm, json.dumps(metadata_copy)),
                )
                conn.execute(
                    """
                    DELETE FROM dag_history WHERE id NOT IN (
                        SELECT id FROM dag_history ORDER BY id DESC LIMIT 50
                    )
                    """
                )
            sqlite_saved = True
        except Exception as e:
            sqlite_error = str(e)

        metadata_copy["storage_backend"] = "json+sqlite" if sqlite_saved else "json"
        metadata_copy["snapshot_status"] = metadata_copy.get(
            "snapshot_status", "validated" if sqlite_saved else "json-fallback"
        )
        if sqlite_error:
            metadata_copy["sqlite_error"] = sqlite_error

        snapshot = {
            "id": next_id,
            "timestamp": now,
            "edges": [{"source": s, "target": t} for s, t in edges],
            "ate_map": ate_map,
            "n_edges": len(edges),
            "algorithm": algorithm,
            "metadata": metadata_copy,
        }
        self._save_json_snapshot(snapshot)

        summary = {
            "id": next_id,
            "timestamp": now,
            "n_edges": len(edges),
            "algorithm": algorithm,
            "status": metadata_copy["snapshot_status"],
            "storage_backend": metadata_copy["storage_backend"],
        }
        index.append(summary)
        index = sorted(index, key=lambda item: item["id"], reverse=True)[:50]
        keep_ids = {item["id"] for item in index}
        for snapshot_path in self.snapshots_dir.glob("*.json"):
            try:
                snapshot_id = int(snapshot_path.stem)
            except ValueError:
                continue
            if snapshot_id not in keep_ids:
                try:
                    snapshot_path.unlink()
                except Exception:
                    pass
        self._save_json_index(sorted(index, key=lambda item: item["id"]))
        return summary

    def get_timeline(self) -> List[Dict[str, Any]]:
        """Return list of all DAG snapshots with timestamps."""
        json_timeline = self._timeline_from_json()
        if json_timeline:
            return json_timeline
        return self._sqlite_timeline()

    def get_snapshot(self, snapshot_id: int) -> Optional[Dict[str, Any]]:
        """Return a full snapshot by ID."""
        json_snapshot = self._load_json_snapshot(snapshot_id)
        if json_snapshot is not None:
            return json_snapshot

        try:
            self._ensure_table()
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT id, timestamp, edges_json, ate_map_json, n_edges, algorithm, metadata_json FROM dag_history WHERE id=?",
                    (snapshot_id,),
                ).fetchone()
        except Exception:
            return None
        if not row:
            return None
        return {
            "id": row[0],
            "timestamp": row[1],
            "edges": json.loads(row[2]),
            "ate_map": json.loads(row[3]),
            "n_edges": row[4],
            "algorithm": row[5],
            "metadata": json.loads(row[6]) if row[6] else {},
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
        for source, target in stable_edges:
            key = f"{source}->{target}"
            old_ate = ate_from.get(key, 0)
            new_ate = ate_to.get(key, 0)
            pct_change = ((new_ate - old_ate) / abs(old_ate)) * 100 if old_ate else 0
            ate_changes.append(
                {
                    "source": source,
                    "target": target,
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

        ate_changes.sort(key=lambda item: abs(item["change_pct"]), reverse=True)

        return {
            "from": {
                "id": id_from,
                "timestamp": snap_from["timestamp"],
                "n_edges": snap_from["n_edges"],
                "status": snap_from.get("metadata", {}).get("snapshot_status", "validated"),
            },
            "to": {
                "id": id_to,
                "timestamp": snap_to["timestamp"],
                "n_edges": snap_to["n_edges"],
                "status": snap_to.get("metadata", {}).get("snapshot_status", "validated"),
            },
            "new_edges": [{"source": source, "target": target} for source, target in new_edges],
            "removed_edges": [{"source": source, "target": target} for source, target in removed_edges],
            "stable_edges": len(stable_edges),
            "ate_changes": ate_changes,
            "summary": {
                "added": len(new_edges),
                "removed": len(removed_edges),
                "stable": len(stable_edges),
                "strengthened": sum(1 for item in ate_changes if item["status"] == "strengthened"),
                "weakened": sum(1 for item in ate_changes if item["status"] == "weakened"),
            },
        }

    def get_edge_history(self, source: str, target: str) -> List[Dict[str, Any]]:
        """Return ATE history for a specific edge across all snapshots."""
        timeline = list(reversed(self.get_timeline()))
        history = []
        edge_key = f"{source}->{target}"

        for item in timeline:
            snapshot = self.get_snapshot(item["id"])
            if snapshot is None:
                continue
            edges = snapshot["edges"]
            edge_present = any(
                edge["source"] == source and edge["target"] == target for edge in edges
            )
            history.append(
                {
                    "snapshot_id": snapshot["id"],
                    "timestamp": snapshot["timestamp"],
                    "present": edge_present,
                    "ate": snapshot["ate_map"].get(edge_key),
                    "status": snapshot.get("metadata", {}).get("snapshot_status", "validated"),
                }
            )

        return history
