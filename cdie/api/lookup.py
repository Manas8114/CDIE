"""
CDIE v4 — Safety Map Lookup + KS-Test Staleness Monitor
Loads pre-computed Safety Map SQLite Database and handles query matching.
"""

import json
import sqlite3
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, Any, Optional

from scipy import stats  # type: ignore


class SafetyMapLookup:
    """Loads and queries the pre-computed Safety Map DB."""

    def __init__(self, safety_map_path: str = None):  # type: ignore
        self.db_path: str | None = None
        self.loaded = False
        self.sha256_hash = ""
        self.json_store: dict[str, Any] | None = None
        self.query_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.max_history = 100

        if safety_map_path:
            self.load(safety_map_path)

    def _candidate_paths(self, path: Path) -> list[Path]:
        candidates: list[Path] = []
        if path.suffix == ".json":
            db_candidate = path.with_suffix(".db")
            candidates.extend([db_candidate, path, db_candidate.with_suffix(".db.bak")])
        else:
            candidates.append(path)
            candidates.append(path.with_suffix(".json"))
            candidates.append(path.with_suffix(".db.bak"))
        deduped: list[Path] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _connect_sqlite(self, candidate: Path) -> bool:
        try:
            with sqlite3.connect(str(candidate)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM store WHERE key='sha256_hash'")
                row = cursor.fetchone()
                self.sha256_hash = json.loads(row[0]) if row else ""

                cursor.execute("SELECT COUNT(*) FROM scenarios")
                n_scen = cursor.fetchone()[0]

            self.db_path = str(candidate)
            self.loaded = True
            print(f"[Lookup] Connected to SQLite Safety Map: {n_scen} scenarios")
            return True
        except Exception as e:
            print(f"[Lookup] Error connecting to DB at {candidate}: {e}")
            return False

    def _materialize_json_db(self, json_path: Path) -> Optional[Path]:
        if not json_path.exists():
            return None

        try:
            from cdie.pipeline.safety_map import save_safety_map

            with open(json_path, "r", encoding="utf-8") as f:
                safety_map = json.load(f)

            db_path, _ = save_safety_map(safety_map, json_path.parent)
            return db_path
        except Exception as e:
            print(f"[Lookup] Error converting JSON Safety Map from {json_path}: {e}")
            return None

    def _load_json(self, json_path: Path) -> bool:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                self.json_store = json.load(f)

            self.db_path = str(json_path)
            self.loaded = True
            self.sha256_hash = str(self.json_store.get("sha256_hash", ""))
            n_scen = len(self.json_store.get("scenarios", {}))
            print(f"[Lookup] Loaded JSON Safety Map: {n_scen} scenarios")
            return True
        except Exception as e:
            print(f"[Lookup] Error loading JSON Safety Map at {json_path}: {e}")
            return False

    def load(self, path: str):
        """Load Safety Map DB and verify integrity."""
        base_path = Path(path)
        self.db_path = None
        self.loaded = False
        self.sha256_hash = ""
        self.json_store = None

        for candidate in self._candidate_paths(base_path):
            if not candidate.exists():
                continue

            if candidate.suffix == ".json":
                if self._load_json(candidate):
                    return True
                candidate = self._materialize_json_db(candidate) or candidate
                if candidate.suffix == ".json" or not candidate.exists():
                    continue

            if self._connect_sqlite(candidate):
                return True

        print(f"[Lookup] Safety Map DB not found or unreadable for base path {base_path}")
        return False

    def is_loaded(self) -> bool:
        if self.json_store is not None:
            return bool(self.loaded)
        return bool(self.loaded and self.db_path and Path(self.db_path).exists())

    def get_storage_backend(self) -> str:
        if self.json_store is not None:
            return "json"
        if self.is_loaded():
            return "sqlite"
        return "unloaded"

    def get_loaded_path(self) -> str:
        return self.db_path or ""

    def _get_store_val(self, key: str, default):
        """Helper to get a value from the key-value store table."""
        if not self.is_loaded():
            return default
        if self.json_store is not None:
            return self.json_store.get(key, default)
        try:
            with sqlite3.connect(self.db_path) as conn:  # type: ignore
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM store WHERE key=?", (key,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            print(f"[Lookup] Error fetching store value for {key}: {e}")
        return default

    def find_scenario(
        self, source: str, target: str, magnitude_key: Optional[str] = None
    ) -> dict | None:
        """Find a matching scenario directly from SQL index."""
        if not self.is_loaded():
            return None
        if self.json_store is not None:
            scenarios = self.json_store.get("scenarios", {})
            if magnitude_key:
                return scenarios.get(f"{source}__{target}__{magnitude_key}")

            candidates = [
                scenario
                for scenario in scenarios.values()
                if scenario.get("source") == source and scenario.get("target") == target
            ]
            if not candidates:
                return None
            return min(candidates, key=lambda s: abs(float(s.get("magnitude_value", 0))))

        try:
            with sqlite3.connect(self.db_path) as conn:  # type: ignore
                cursor = conn.cursor()
                if magnitude_key:
                    scenario_id = f"{source}__{target}__{magnitude_key}"
                    cursor.execute(
                        "SELECT data_payload FROM scenarios WHERE id=?", (scenario_id,)
                    )
                else:
                    cursor.execute(
                        "SELECT data_payload FROM scenarios WHERE source=? AND target=? ORDER BY ABS(magnitude_value) ASC LIMIT 1",
                        (source, target),
                    )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            print(f"Error executing find_scenario query: {e}")

        return None

    def find_best_scenario(
        self, source: str, target: str, magnitude_pct: float
    ) -> tuple[dict | None, bool]:
        """Find the closest pre-computed scenario for a given magnitude via SQL."""
        if not self.is_loaded():
            return None, False

        mag_key_map = {
            10: "increase_10",
            20: "increase_20",
            30: "increase_30",
            50: "increase_50",
            -10: "decrease_10",
            -20: "decrease_20",
        }

        exact_key = mag_key_map.get(int(magnitude_pct))

        if self.json_store is not None:
            scenarios = self.json_store.get("scenarios", {})
            if exact_key:
                exact = scenarios.get(f"{source}__{target}__{exact_key}")
                if exact:
                    return exact, True

            best_scenario = None
            best_dist = float("inf")
            for scenario in scenarios.values():
                if scenario.get("source") != source or scenario.get("target") != target:
                    continue
                dist = abs(float(scenario.get("magnitude_value", 0)) * 100 - magnitude_pct)
                if dist < best_dist:
                    best_dist = dist
                    best_scenario = scenario
                    if dist == 0:
                        return best_scenario, True

            return best_scenario, False

        try:
            with sqlite3.connect(self.db_path) as conn:  # type: ignore
                cursor = conn.cursor()
                if exact_key:
                    scenario_id = f"{source}__{target}__{exact_key}"
                    cursor.execute(
                        "SELECT data_payload FROM scenarios WHERE id=?", (scenario_id,)
                    )
                    row = cursor.fetchone()
                    if row:
                        return json.loads(row[0]), True

                cursor.execute(
                    "SELECT magnitude_value, data_payload FROM scenarios WHERE source=? AND target=?",
                    (source, target),
                )
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

    def find_prescriptions(
        self, target: str, limit: int = 3, maximize: bool = True
    ) -> list:
        """Find the top interventions to maximize or minimize a target."""
        if not self.is_loaded():
            return []
        if self.json_store is not None:
            prescriptions = [
                scenario
                for scenario in self.json_store.get("scenarios", {}).values()
                if scenario.get("target") == target
                and scenario.get("refutation_status") == "VALIDATED"
            ]
            prescriptions.sort(
                key=lambda x: x.get("effect", {}).get("point_estimate", 0),
                reverse=maximize,
            )
            return list(prescriptions[:limit])

        try:
            with sqlite3.connect(self.db_path) as conn:  # type: ignore
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT data_payload FROM scenarios
                    WHERE target = ?
                """,
                    (target,),
                )

                rows = cursor.fetchall()
                prescriptions = []

                for row in rows:
                    scenario = json.loads(row[0])
                    if scenario.get("refutation_status") == "VALIDATED":
                        prescriptions.append(scenario)

                prescriptions.sort(
                    key=lambda x: x.get("effect", {}).get("point_estimate", 0),
                    reverse=maximize,
                )

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
            return {
                "warning": False,
                "ks_statistic": 0.0,
                "p_value": 1.0,
                "reason": "insufficient_data",
            }

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
            if self.json_store is not None:
                n_scen = len(self.json_store.get("scenarios", {}))
            else:
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
            "storage_backend": self.get_storage_backend(),
            "loaded_path": self.get_loaded_path(),
        }
