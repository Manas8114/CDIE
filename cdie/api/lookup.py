"""
CDIE v4 — Safety Map Lookup + KS-Test Staleness Monitor
Loads pre-computed Safety Map SQLite Database and handles query matching.
"""

import json
import sqlite3
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, Any

from scipy import stats  # type: ignore

class SafetyMapLookup:
    """Loads and queries the pre-computed Safety Map DB."""

    def __init__(self, safety_map_path: str = None):  # type: ignore
        self.db_path: str | None = None
        self.sha256_hash = ""
        self.query_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.max_history = 100

        if safety_map_path:
            self.load(safety_map_path)

    def load(self, path: str):
        """Load Safety Map DB and verify integrity."""
        base_path = Path(path)
        if base_path.suffix == '.json':
            self.db_path = str(base_path.with_suffix('.db'))
        else:
            self.db_path = str(base_path)

        if not Path(self.db_path).exists():  # type: ignore
            print(f"[Lookup] Safety Map DB not found at {self.db_path}")
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:  # type: ignore
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM store WHERE key='sha256_hash'")
                row = cursor.fetchone()
                if row:
                    self.sha256_hash = json.loads(row[0])

                cursor.execute("SELECT COUNT(*) FROM scenarios")
                n_scen = cursor.fetchone()[0]
                
            print(f"[Lookup] Connected to SQLite Safety Map: {n_scen} scenarios")
            return True
        except Exception as e:
            print(f"[Lookup] Error connecting to DB: {e}")
            return False

    def is_loaded(self) -> bool:
        return self.db_path is not None and Path(self.db_path).exists()  # type: ignore

    def _get_store_val(self, key: str, default):
        """Helper to get a value from the key-value store table."""
        if not self.is_loaded():
            return default
        try:
            with sqlite3.connect(self.db_path) as conn:  # type: ignore
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM store WHERE key=?", (key,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception:
            pass
        return default

    def find_scenario(self, source: str, target: str, magnitude_key: str = None) -> dict | None:  # type: ignore
        """Find a matching scenario directly from SQL index."""
        if not self.is_loaded():
            return None

        try:
            with sqlite3.connect(self.db_path) as conn:  # type: ignore
                cursor = conn.cursor()
                if magnitude_key:
                    scenario_id = f"{source}__{target}__{magnitude_key}"
                    cursor.execute("SELECT data_payload FROM scenarios WHERE id=?", (scenario_id,))
                else:
                    cursor.execute(
                        "SELECT data_payload FROM scenarios WHERE source=? AND target=? ORDER BY ABS(magnitude_value) ASC LIMIT 1",
                        (source, target)
                    )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            print(f"Error executing find_scenario query: {e}")
            
        return None

    def find_best_scenario(self, source: str, target: str, magnitude_pct: float) -> tuple[dict | None, bool]:
        """Find the closest pre-computed scenario for a given magnitude via SQL."""
        if not self.is_loaded():
            return None, False

        mag_key_map = {
            10: "increase_10", 20: "increase_20",
            30: "increase_30", 50: "increase_50",
            -10: "decrease_10", -20: "decrease_20",
        }

        exact_key = mag_key_map.get(int(magnitude_pct))

        try:
            with sqlite3.connect(self.db_path) as conn:  # type: ignore
                cursor = conn.cursor()
                if exact_key:
                    scenario_id = f"{source}__{target}__{exact_key}"
                    cursor.execute("SELECT data_payload FROM scenarios WHERE id=?", (scenario_id,))
                    row = cursor.fetchone()
                    if row:
                        return json.loads(row[0]), True

                cursor.execute("SELECT magnitude_value, data_payload FROM scenarios WHERE source=? AND target=?", (source, target))
                rows = cursor.fetchall()

                best_scenario = None
                best_dist = float("inf")
                for mag_val, payload_str in rows:
                    dist = abs(mag_val * 100 - magnitude_pct)
                    if dist < best_dist:
                        best_dist = dist
                        best_scenario = json.loads(payload_str)
                        if dist == 0:
                            return best_scenario, True

                return best_scenario, False
        except sqlite3.Error as e:
            print(f"[Lookup] Error querying scenario directly: {e}")
            return None, False

    def find_prescriptions(self, target: str, limit: int = 3, maximize: bool = True) -> list:
        """Find the top interventions to maximize or minimize a target."""
        if not self.is_loaded():
            return []
        
        try:
            with sqlite3.connect(self.db_path) as conn:  # type: ignore
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT data_payload FROM scenarios
                    WHERE target = ?
                ''', (target,))
                
                rows = cursor.fetchall()
                prescriptions = []
                
                for row in rows:
                    scenario = json.loads(row[0])
                    if scenario.get("refutation_status") != "UNPROVEN":
                        prescriptions.append(scenario)
                        
                prescriptions.sort(key=lambda x: x.get("effect", {}).get("point_estimate", 0), reverse=maximize)
                
                return list(prescriptions[:limit])  # type: ignore
        except sqlite3.Error as e:
            print(f"[Lookup] Error finding prescriptions: {e}")
            return []

    def check_staleness(self, source: str, current_mean: float) -> Dict[str, Any]:
        """KS-test staleness check against pre-computed DB distributions."""
        if not self.is_loaded():
            return {"warning": False, "ks_statistic": 0.0, "p_value": 1.0}

        self.query_history[source].append(current_mean)

        training_dists = self._get_store_val("training_distributions", {})
        training_dist = training_dists.get(source, {})
        sample_values = training_dist.get("sample_values", [])

        if len(self.query_history[source]) < 10 or len(sample_values) < 10:
            return {"warning": False, "ks_statistic": 0.0, "p_value": 1.0, "reason": "insufficient_data"}

        recent_values = list(self.query_history[source])
        ks_stat, p_value = stats.ks_2samp(recent_values, sample_values)

        warning = ks_stat > 0.2 or p_value < 0.05

        return {
            "warning": warning,
            "ks_statistic": float(f"{float(ks_stat):.4f}"),
            "p_value": float(f"{float(p_value):.4f}"),
        }

    def get_graph(self) -> dict:
        return self._get_store_val("graph", {"nodes": [], "edges": []})

    def get_catl(self) -> dict:
        return self._get_store_val("catl", {})

    def get_benchmarks(self) -> dict:
        return self._get_store_val("benchmarks", {})

    def get_xgboost_comparison(self) -> dict:
        return self._get_store_val("xgboost_comparison", {})

    def get_temporal(self) -> dict:
        return self._get_store_val("temporal", {})

    def get_metadata(self) -> dict:
        n_scen = 0
        if self.is_loaded():
            try:
                with sqlite3.connect(self.db_path) as conn:  # type: ignore
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM scenarios")
                    n_scen = cursor.fetchone()[0]
            except Exception:
                pass

        return {
            "version": self._get_store_val("version", "4.0.0"),
            "created_at": self._get_store_val("created_at", ""),
            "n_variables": self._get_store_val("n_variables", 0),
            "sha256_hash": self.sha256_hash,
            "n_scenarios": n_scen,
            "refutation_summary": self._get_store_val("refutation_summary", {}),
        }
