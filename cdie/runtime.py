"""
Runtime storage helpers for CDIE.

Separates durable project artifacts (for example, canonical JSON exports)
from operational runtime state (for example, temp SQLite mirrors and drift
history files) so the app can continue working even when the project folder is
hosted on a restrictive filesystem.

Windows-safe: all paths are resolved with Path.resolve() to normalize
mixed separators and relative segments before use.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_logger = logging.getLogger(__name__)

# Cloud-sync folder keywords — emit a warning when CDIE_RUNTIME_DIR falls inside one
_SYNC_KEYWORDS = ('onedrive', 'dropbox', 'icloud', 'google drive', 'googledrive', 'box sync')


def _warn_if_sync_path(path: Path) -> None:
    """Emit a warning when *path* appears to live inside a cloud-sync folder."""
    path_lower = str(path).lower()
    for kw in _SYNC_KEYWORDS:
        if kw in path_lower:
            _logger.warning(
                '[runtime] CDIE_RUNTIME_DIR=%s looks like a cloud-sync path (%s). '
                'SQLite files in synced folders can become corrupted. '
                'Set CDIE_RUNTIME_DIR to a local path, e.g. C:\\Temp\\cdie or /tmp/cdie.',
                path,
                kw,
            )
            return


def get_runtime_dir(data_dir: Path | None = None) -> Path:
    """Return a writable runtime directory, preferring a local machine path.

    Resolution order:
    1. ``CDIE_RUNTIME_DIR`` env var (must be an absolute or relative path)
    2. ``%LOCALAPPDATA%\\CDIE`` on Windows
    3. ``<data_dir>/.runtime`` when *data_dir* is provided
    4. ``<cwd>/.cdie_runtime`` as a last resort

    All returned paths are resolved to absolute form.
    """
    env_dir = os.environ.get('CDIE_RUNTIME_DIR', '').strip()
    if env_dir:
        resolved = Path(env_dir).resolve()
        _warn_if_sync_path(resolved)
        return resolved

    local_app_data = os.environ.get('LOCALAPPDATA', '').strip()
    if local_app_data:
        return (Path(local_app_data) / 'CDIE').resolve()

    if data_dir is not None:
        return (data_dir / '.runtime').resolve()

    return (Path.cwd() / '.cdie_runtime').resolve()


def ensure_runtime_dir(data_dir: Path | None = None) -> Path:
    """Return the runtime directory, creating it if necessary."""
    runtime_dir = get_runtime_dir(data_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def get_runtime_paths(data_dir: Path | None = None, create: bool = False) -> dict[str, Path]:
    """Return a dict of all well-known runtime sub-paths.

    Args:
        data_dir: Optional base for the runtime directory fallback.
        create:   When True, create the runtime and drift directories.

    Returns:
        Dictionary with keys:
        ``runtime_dir``, ``runtime_db``, ``runtime_temp_db``,
        ``runtime_db_backup``, ``drift_dir``, ``drift_index``,
        ``drift_snapshots_dir``.
    """
    runtime_dir = ensure_runtime_dir(data_dir) if create else get_runtime_dir(data_dir)
    drift_dir = runtime_dir / 'drift_history'
    if create:
        drift_dir.mkdir(parents=True, exist_ok=True)
    return {
        'runtime_dir': runtime_dir,
        'runtime_db': runtime_dir / 'safety_map.runtime.db',
        'runtime_temp_db': runtime_dir / 'safety_map.runtime.db.tmp',
        'runtime_db_backup': runtime_dir / 'safety_map.runtime.db.bak',
        'drift_dir': drift_dir,
        'drift_index': drift_dir / 'index.json',
        'drift_snapshots_dir': drift_dir / 'snapshots',
    }
