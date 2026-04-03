"""
Runtime storage helpers for CDIE.

Separates durable project artifacts (for example, canonical JSON exports)
from operational runtime state (for example, temp SQLite mirrors and drift
history files) so the app can continue working even when the project folder is
hosted on a restrictive filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_runtime_dir(data_dir: Path | None = None) -> Path:
    """Return a writable runtime directory, preferring a local machine path."""
    env_dir = os.environ.get("CDIE_RUNTIME_DIR")
    if env_dir:
        return Path(env_dir)

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "CDIE"

    if data_dir is not None:
        return data_dir / ".runtime"

    return Path.cwd() / ".cdie_runtime"


def ensure_runtime_dir(data_dir: Path | None = None) -> Path:
    runtime_dir = get_runtime_dir(data_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def get_runtime_paths(
    data_dir: Path | None = None, create: bool = False
) -> dict[str, Path]:
    runtime_dir = ensure_runtime_dir(data_dir) if create else get_runtime_dir(data_dir)
    drift_dir = runtime_dir / "drift_history"
    if create:
        drift_dir.mkdir(parents=True, exist_ok=True)
    return {
        "runtime_dir": runtime_dir,
        "runtime_db": runtime_dir / "safety_map.runtime.db",
        "runtime_temp_db": runtime_dir / "safety_map.runtime.db.tmp",
        "runtime_db_backup": runtime_dir / "safety_map.runtime.db.bak",
        "drift_dir": drift_dir,
        "drift_index": drift_dir / "index.json",
        "drift_snapshots_dir": drift_dir / "snapshots",
    }
