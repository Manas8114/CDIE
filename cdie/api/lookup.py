"""
CDIE v5 — Safety Map Lookup + KS-Test Staleness Monitor
Loads pre-computed Safety Map SQLite Database and handles query matching.
"""

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, cast, List

import numpy as np
from scipy import stats

from cdie.config import MAGNITUDE_LOOKUP
from cdie.observability import get_logger
from cdie.pipeline.datastore import DataStoreManager

log = get_logger(__name__)


class SafetyMapLookup:
    """Loads and queries the pre-computed Safety Map DB."""

    def __init__(self, safety_map_path: str | None = None) -> None:
        self.db_path: str | None = None
        self.loaded = False
        self.sha256_hash = ''
        self.json_store: dict[str, Any] | None = None
        self.query_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=100))
        self.max_history = 100

        if safety_map_path:
            self.load(safety_map_path)

    def _candidate_paths(self, path: Path) -> list[Path]:
        candidates: list[Path] = []
        if path.suffix == '.json':
            db_candidate = path.with_suffix('.db')
            candidates.extend([db_candidate, path, db_candidate.with_suffix('.db.bak')])
        else:
            candidates.append(path)
            candidates.append(path.with_suffix('.json'))
            candidates.append(path.with_suffix('.db.bak'))
        deduped: list[Path] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _connect_sqlite(self, candidate: Path) -> bool:
        try:
            store = DataStoreManager.get_sqlite_store(candidate)
            self.sha256_hash = store.get_kv('store', 'sha256_hash', '')

            n_scen_row = store.fetch_one('SELECT COUNT(*) as count FROM scenarios')
            n_scen = n_scen_row['count'] if n_scen_row else 0

            self.db_path = str(candidate)
            self.loaded = True
            log.info('[Lookup] Connected to SQLite Safety Map', n_scenarios=n_scen)
            return True
        except Exception as e:
            log.warning('[Lookup] Error connecting to DB', candidate=str(candidate), error=str(e))
            return False

    def _materialize_json_db(self, json_path: Path) -> Path | None:
        if not json_path.exists():
            return None

        try:
            from cdie.pipeline.safety_map import save_safety_map

            with open(json_path, encoding='utf-8') as f:
                safety_map = json.load(f)

            db_path, _ = save_safety_map(safety_map, json_path.parent)
            return Path(db_path)
        except Exception as e:
            log.warning('[Lookup] Error converting JSON Safety Map', json_path=str(json_path), error=str(e))
            return None

    def _load_json(self, json_path: Path) -> bool:
        try:
            with open(json_path, encoding='utf-8') as f:
                self.json_store = json.load(f)

            self.db_path = str(json_path)
            self.loaded = True
            self.sha256_hash = str(self.json_store.get('sha256_hash', ''))
            n_scen = len(self.json_store.get('scenarios', {}))
            log.info('[Lookup] Loaded JSON Safety Map', n_scenarios=n_scen)
            return True
        except Exception as e:
            log.warning('[Lookup] Error loading JSON Safety Map', json_path=str(json_path), error=str(e))
            return False

    def load(self, path: str) -> bool:
        """Load Safety Map DB and verify integrity."""
        base_path = Path(path)
        self.db_path = None
        self.loaded = False
        self.sha256_hash = ''

        for candidate in self._candidate_paths(base_path):
            if not candidate.exists():
                continue

            if candidate.suffix == '.db' or candidate.suffix == '.bak':
                if self._connect_sqlite(candidate):
                    return True
            elif candidate.suffix == '.json':
                db_path = self._materialize_json_db(candidate)
                if db_path and self._connect_sqlite(db_path):
                    return True
                if self._load_json(candidate):
                    return True
        return False

    def is_loaded(self) -> bool:
        return self.loaded

    def get_loaded_path(self) -> str:
        return self.db_path or ''

    def _get_store_val(self, key: str, default: Any = None) -> Any:
        """Helper to get a value from the key-value store table."""
        if not self.is_loaded():
            return default

        if self.json_store is not None:
            return self.json_store.get(key, default)

        store = DataStoreManager.get_sqlite_store(self.db_path)  # type: ignore
        return store.get_kv('store', key, default)

    def find_scenario(self, source: str, target: str, magnitude_key: str | None = None) -> dict[str, Any] | None:
        """Find a matching scenario directly from SQL index."""
        if not self.is_loaded():
            return None

        if self.json_store is not None:
            scenarios = self.json_store.get('scenarios', {})
            if magnitude_key:
                return cast(dict[str, Any], scenarios.get(f'{source}__{target}__{magnitude_key}'))

            candidates = [
                scenario
                for scenario in scenarios.values()
                if scenario.get('source') == source and scenario.get('target') == target
            ]
            if not candidates:
                return None
            return cast(dict[str, Any], min(candidates, key=lambda s: abs(float(s.get('magnitude_value', 0)))))

        store = DataStoreManager.get_sqlite_store(self.db_path)  # type: ignore
        if magnitude_key:
            scenario_id = f'{source}__{target}__{magnitude_key}'
            row = store.fetch_one('SELECT data_payload FROM scenarios WHERE id=?', (scenario_id,))
        else:
            row = store.fetch_one(
                'SELECT data_payload FROM scenarios WHERE source=? AND target=? '
                'ORDER BY ABS(magnitude_value) ASC LIMIT 1',
                (source, target),
            )

        if row:
            return cast(dict[str, Any], json.loads(row['data_payload']))
        return None

    def find_best_scenario(self, source: str, target: str, magnitude_pct: float) -> tuple[dict[str, Any] | None, bool]:
        """Find the closest pre-computed scenario for a given magnitude."""
        if not self.is_loaded():
            return None, False

        exact_key = MAGNITUDE_LOOKUP.get(int(magnitude_pct))

        if self.json_store is not None:
            scenarios = self.json_store.get('scenarios', {})
            if exact_key:
                exact = scenarios.get(f'{source}__{target}__{exact_key}')
                if exact:
                    return cast(dict[str, Any], exact), True

            best_scenario = None
            best_dist = float('inf')
            for scenario in scenarios.values():
                if scenario.get('source') != source or scenario.get('target') != target:
                    continue
                dist = abs(float(scenario.get('magnitude_value', 0)) * 100 - magnitude_pct)
                if dist < best_dist:
                    best_dist = dist
                    best_scenario = scenario
                    if dist == 0:
                        return best_scenario, True

            return best_scenario, False

        store = DataStoreManager.get_sqlite_store(self.db_path)  # type: ignore
        if exact_key:
            scenario_id = f'{source}__{target}__{exact_key}'
            row = store.fetch_one('SELECT data_payload FROM scenarios WHERE id=?', (scenario_id,))
            if row:
                return cast(dict[str, Any], json.loads(row['data_payload'])), True

        rows = store.fetch_all(
            'SELECT magnitude_value, data_payload FROM scenarios WHERE source=? AND target=?',
            (source, target),
        )

        best_scenario = None
        best_dist = float('inf')
        for row in rows:
            mag_val = row['magnitude_value']
            dist = abs(mag_val * 100 - magnitude_pct)
            if dist < best_dist:
                best_dist = dist
                best_scenario = cast(dict[str, Any], json.loads(row['data_payload']))
                if dist == 0:
                    return best_scenario, True

        return best_scenario, False

    def find_prescriptions(self, target: str, limit: int = 3, maximize: bool = True) -> list[dict[str, Any]]:
        """Find the top interventions to maximize or minimize a target."""
        if not self.is_loaded():
            return []

        prescriptions: list[dict[str, Any]] = []

        if self.json_store is not None:
            prescriptions = [
                scenario
                for scenario in self.json_store.get('scenarios', {}).values()
                if scenario.get('target') == target and scenario.get('refutation_status') == 'VALIDATED'
            ]
        else:
            store = DataStoreManager.get_sqlite_store(self.db_path)  # type: ignore
            rows = store.fetch_all('SELECT data_payload FROM scenarios WHERE target = ?', (target,))
            for row in rows:
                scenario = json.loads(row['data_payload'])
                if scenario.get('refutation_status') == 'VALIDATED':
                    prescriptions.append(scenario)

        prescriptions.sort(
            key=lambda x: x.get('effect', {}).get('point_estimate', 0),
            reverse=maximize,
        )

        return prescriptions[:limit]

    def check_staleness(self, source: str, current_mean: float) -> dict[str, Any]:
        """KS-test and KL-divergence staleness check against pre-computed DB distributions."""
        if not self.is_loaded():
            return {
                'warning': False,
                'ks_statistic': 0.0,
                'p_value': 1.0,
                'kl_divergence': 0.0,
                'drift_detected': False,
            }

        self.query_history[source].append(current_mean)

        training_dists = self._get_store_val('training_distributions', {})
        training_dist = training_dists.get(source, {})
        sample_values = training_dist.get('sample_values', [])

        if len(self.query_history[source]) < 10 or len(sample_values) < 10:
            return {
                'warning': False,
                'ks_statistic': 0.0,
                'p_value': 1.0,
                'kl_divergence': 0.0,
                'drift_detected': False,
                'reason': 'insufficient_data',
            }

        recent_values = list(self.query_history[source])
        ks_stat, p_value = stats.ks_2samp(recent_values, sample_values)

        # KL Divergence Calculation
        # Use common bins for both distributions to ensure they are defined over the same space
        all_vals = np.concatenate([recent_values, sample_values])
        bins = np.linspace(np.min(all_vals), np.max(all_vals), 20)

        hist_recent, _ = np.histogram(recent_values, bins=bins, density=True)
        hist_training, _ = np.histogram(sample_values, bins=bins, density=True)

        # Add small epsilon to avoid log(0)
        hist_recent = hist_recent + 1e-6
        hist_training = hist_training + 1e-6

        kl_div = float(stats.entropy(hist_recent, hist_training))

        # Warning threshold: KS stat > 0.2 OR low p-value OR high KL divergence
        warning = ks_stat > 0.2 or p_value < 0.05 or kl_div > 0.5

        return {
            'warning': warning,
            'ks_statistic': float(f'{float(ks_stat):.4f}'),
            'p_value': float(f'{float(p_value):.4f}'),
            'kl_divergence': float(f'{kl_div:.4f}'),
            'drift_detected': warning,
        }

    def get_graph(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._get_store_val('graph', {'nodes': [], 'edges': []}))

    def get_catl(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._get_store_val('catl', {}))

    def get_benchmarks(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._get_store_val('benchmarks', {}))

    def get_xgboost_comparison(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._get_store_val('xgboost_comparison', {}))

    def get_temporal(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._get_store_val('temporal', {}))

    def get_metadata(self) -> dict[str, Any]:
        n_scen = 0
        if self.is_loaded():
            if self.json_store is not None:
                n_scen = len(self.json_store.get('scenarios', {}))
            else:
                store = DataStoreManager.get_sqlite_store(self.db_path)  # type: ignore
                row = store.fetch_one('SELECT COUNT(*) as count FROM scenarios')
                n_scen = cast(int, row['count']) if row else 0

        return {
            'status': 'healthy' if self.loaded else 'degraded',
            'sha256_hash': self.sha256_hash,
            'db_path': self.db_path,
            'n_scenarios': n_scen,
            'storage_backend': 'json' if self.json_store else 'sqlite' if self.loaded else 'unloaded',
        }
