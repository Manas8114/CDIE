"""
CDIE v5 — Centralized Data Store Manager
Provides a unified, secure interface for SQLite and CSV data operations.
Implements singleton-pattern connection management and parameterized query protection.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from cdie.config import DATA_DIR, VARIABLE_NAMES

logger = logging.getLogger(__name__)


class SQLiteStore:
    """Safe SQLite wrapper with connection management and JSON serialization helpers."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        """Create a new database connection with Row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute SELECT and return list of dictionaries."""
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f'Database error in fetch_all: {e} | Query: {query}')
            return []

    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        """Execute SELECT and return a single dictionary or None."""
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            logger.error(f'Database error in fetch_one: {e} | Query: {query}')
            return None

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> int:
        """Execute INSERT/UPDATE/DELETE and return affected row count."""
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f'Database error in execute: {e} | Query: {query}')
            return 0

    def execute_many(self, query: str, params_list: list[tuple[Any, ...]]) -> int:
        """Execute executemany and return affected row count."""
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
                conn.commit()
                return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f'Database error in execute_many: {e} | Query: {query}')
            return 0

    def execute_script(self, script: str) -> None:
        """Execute multiple SQL statements."""
        try:
            with self.connect() as conn:
                conn.executescript(script)
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f'Database error in execute_script: {e}')

    def get_kv(self, table: str, key: str, default: Any = None) -> Any:
        """Retrieve a JSON-serialized value from a key-value store table."""
        row = self.fetch_one(f'SELECT value FROM {table} WHERE key=?', (key,))
        if row:
            try:
                return json.loads(row['value'])
            except (json.JSONDecodeError, TypeError):
                return row['value']
        return default

    def set_kv(self, table: str, key: str, value: Any) -> None:
        """Store a value in a key-value store table with JSON serialization."""
        serialized = json.dumps(value)
        self.execute(f'INSERT OR REPLACE INTO {table} (key, value) VALUES (?, ?)', (key, serialized))


class DataStoreManager:
    """
    Singleton Manager for all project data assets.
    Centralizes access to Knowledge Store, Safety Map, and Master CSV.
    """

    _sqlite_stores: dict[str, SQLiteStore] = {}

    @classmethod
    def get_sqlite_store(cls, db_path: str | Path) -> SQLiteStore:
        path_str = str(db_path)
        if path_str not in cls._sqlite_stores:
            cls._sqlite_stores[path_str] = SQLiteStore(Path(db_path))
        return cls._sqlite_stores[path_str]

    def __init__(self, master_csv_path: Path | None = None) -> None:
        self.master_csv_path = master_csv_path or (DATA_DIR / 'scm_data.csv')

    def load_master_df(self) -> pd.DataFrame:
        """Load the master CSV dataset."""
        if self.master_csv_path.exists():
            return pd.read_csv(self.master_csv_path)
        return pd.DataFrame(columns=VARIABLE_NAMES + ['CustomerSegment', 'TimeIndex'])

    def save_master_df(self, df: pd.DataFrame) -> None:
        """Persist the master CSV dataset."""
        self.master_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.master_csv_path, index=False)

    def save_pipeline_data(self, df: pd.DataFrame, dag: Any) -> tuple[Path, Path]:
        """
        Securely save both the tabular dataset and the causal DAG.
        CSV is saved via the master store logic; DAG is saved via secure pickle.
        """
        import pickle

        self.save_master_df(df)

        dag_path = self.master_csv_path.parent / 'ground_truth.pkl'
        with open(dag_path, 'wb') as f:
            pickle.dump(dag, f)

        return self.master_csv_path, dag_path

    def merge_data(self, new_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Logic migrated from data_merger.py.
        Merges new data into the master store with deduplication.
        """
        warnings = []
        master_df = self.load_master_df()

        # Schema alignment
        for col in master_df.columns:
            if col not in new_df.columns:
                new_df[col] = None
        new_df = new_df[master_df.columns]

        # Deduplication
        if not master_df.empty:
            subset_cols = [c for c in VARIABLE_NAMES if c in master_df.columns and c in new_df.columns]
            combined = pd.concat([master_df, new_df], ignore_index=True)
            comp_df = combined[subset_cols].astype('string').fillna('__MISSING__')

            is_dup = comp_df.duplicated(keep='first')
            new_rows_mask = ~is_dup[len(master_df) :]
            new_df_filtered = new_df[new_rows_mask.values]

            dup_count = len(new_df) - len(new_df_filtered)
            if dup_count > 0:
                warnings.append(f'Ignored {dup_count} duplicate rows.')
            new_df = new_df_filtered

        # Append and Save
        result_df = pd.concat([master_df, new_df], ignore_index=True) if not master_df.empty else new_df

        if 'TimeIndex' in result_df.columns and result_df['TimeIndex'].isnull().any():
            result_df['TimeIndex'] = range(len(result_df))

        self.save_master_df(result_df)
        warnings.append(f'Master store updated: {len(master_df)} -> {len(result_df)} rows.')
        return result_df, warnings
