from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path(request) -> Path:
    """
    Workspace-local replacement for pytest's tmp_path fixture.

    The default temp directory is not writable on this machine, so tests use
    a sandbox inside the repository instead.
    """
    base_dir = Path(__file__).parent / '.tmp_pytest'
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f'{request.node.name}_{uuid.uuid4().hex[:8]}'
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
