"""
CDIE v5 — Centralized Configuration
Centralizes environment variables, directory paths, and production security settings.
Includes startup validation with actionable warnings for missing/misconfigured variables.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import yaml

# ── Base Paths ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
BASE_DIR = PROJECT_ROOT

# ── Data Storage ─────────────────────────────────────────────────────────────

_raw_data_dir = os.environ.get('CDIE_DATA_DIR', '')
DATA_DIR: Path = Path(_raw_data_dir).resolve() if _raw_data_dir else (PROJECT_ROOT / 'data').resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Database Paths ────────────────────────────────────────────────────────────

SAFETY_MAP_DB = DATA_DIR / 'safety_map.db'
KNOWLEDGE_DB = DATA_DIR / 'knowledge.db'
DRIFT_DB = DATA_DIR / 'drift_history.db'

# ── API Settings ──────────────────────────────────────────────────────────────

ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:3000,http://localhost:8501').split(',')
RATE_LIMIT_STRING = os.environ.get('RATE_LIMIT', '100/minute')

# ── Secure Headers (Production Standards) ────────────────────────────────────

SECURE_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'Content-Security-Policy': "default-src 'self'; script-src 'self'; object-src 'none';",
}

# ── LLM / OPEA Integration ────────────────────────────────────────────────────

TGI_ENDPOINT = os.environ.get('TGI_ENDPOINT') or None
OPEA_LLM_ENDPOINT = os.environ.get('OPEA_LLM_ENDPOINT') or None
REDIS_URL = os.environ.get('REDIS_URL') or None

# ── Observability ─────────────────────────────────────────────────────────────

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
METRICS_ENABLED = os.environ.get('CDIE_ENABLE_METRICS', '1') == '1'

# ── SCM Configuration (Telecom SIM Box Fraud Domain) ─────────────────────────

VARIABLE_NAMES = [
    'CallDataRecordVolume',
    'SIMBoxFraudAttempts',
    'FraudPolicyStrictness',
    'SIMFraudDetectionRate',
    'RevenueLeakageVolume',
    'SubscriberRetentionScore',
    'ARPUImpact',
    'NetworkOpExCost',
    'CashFlowRisk',
    'NetworkLoad',
    'RegulatorySignal',
    'ITURegulatoryPressure',
]

GROUND_TRUTH_EDGES = [
    ('CallDataRecordVolume', 'SIMBoxFraudAttempts'),
    ('CallDataRecordVolume', 'NetworkLoad'),
    ('CallDataRecordVolume', 'ARPUImpact'),
    ('SIMBoxFraudAttempts', 'SIMFraudDetectionRate'),
    ('SIMBoxFraudAttempts', 'RevenueLeakageVolume'),
    ('FraudPolicyStrictness', 'SIMFraudDetectionRate'),
    ('FraudPolicyStrictness', 'NetworkOpExCost'),
    ('SIMFraudDetectionRate', 'RevenueLeakageVolume'),
    ('SIMFraudDetectionRate', 'SubscriberRetentionScore'),
    ('RevenueLeakageVolume', 'ARPUImpact'),
    ('RevenueLeakageVolume', 'CashFlowRisk'),
    ('SubscriberRetentionScore', 'ARPUImpact'),
    ('RegulatorySignal', 'SubscriberRetentionScore'),
    ('RegulatorySignal', 'ITURegulatoryPressure'),
    ('ITURegulatoryPressure', 'FraudPolicyStrictness'),
    ('NetworkOpExCost', 'CashFlowRisk'),
    ('NetworkLoad', 'NetworkOpExCost'),
]

# ── Versioning ────────────────────────────────────────────────────────────────

VERSION = '5.0.0'
APP_TITLE = 'CDIE v5 — Production Hardened'

# ── Magnitude Levels ──────────────────────────────────────────────────────────

# Programmatic generation of magnitude levels (5% to 50%)
MAGNITUDE_LEVELS = {f'increase_{i}': i / 100.0 for i in [5, 10, 15, 20, 25, 30, 40, 50]}
MAGNITUDE_LEVELS.update({f'decrease_{i}': -i / 100.0 for i in [5, 10, 15, 20, 25, 30, 40, 50]})

# Lookup map for API (percent int -> key)
MAGNITUDE_LOOKUP = {int(v * 100): k for k, v in MAGNITUDE_LEVELS.items()}

# ── Startup Validation ────────────────────────────────────────────────────────

_logger = logging.getLogger(__name__)


def _validate_config() -> None:
    """Run startup checks and emit warnings for misconfigured variables.

    Intentionally non-fatal: CDIE degrades gracefully when optional services
    are absent, so we warn rather than raise to keep the local dev experience smooth.
    """
    issues: list[str] = []

    # DATA_DIR must be writable
    try:
        test_file = DATA_DIR / '.write_test'
        test_file.touch()
        test_file.unlink()
    except OSError as exc:
        issues.append(f'DATA_DIR={DATA_DIR} is not writable: {exc}')

    # Warn about cloud-sync paths for RUNTIME_DIR
    runtime_raw = os.environ.get('CDIE_RUNTIME_DIR', '')
    if runtime_raw:
        runtime_path = Path(runtime_raw).resolve()
        _sync_keywords = ('onedrive', 'dropbox', 'icloud', 'google drive', 'googledrive', 'box')
        path_lower = str(runtime_path).lower()
        if any(kw in path_lower for kw in _sync_keywords):
            issues.append(
                f'CDIE_RUNTIME_DIR={runtime_path} appears to be inside a cloud-sync folder. '
                'This can cause SQLite corruption. Set CDIE_RUNTIME_DIR to a local path '
                'such as C:\\\\Temp\\\\cdie or /tmp/cdie.'
            )

    # Optional services — info level, not warnings
    if not OPEA_LLM_ENDPOINT:
        _logger.info('[config] OPEA_LLM_ENDPOINT not set — using template-based explanations.')
    if not TGI_ENDPOINT:
        _logger.debug('[config] TGI_ENDPOINT not set — will use OPEA_LLM_ENDPOINT fallback.')
    if not REDIS_URL:
        _logger.info('[config] REDIS_URL not set — RAG response caching disabled.')

    # Frontend check
    if not os.environ.get('NEXT_PUBLIC_API_URL'):
        _logger.info('[config] NEXT_PUBLIC_API_URL not set — frontend may use wrong API base.')

    # Emit consolidated warnings
    for issue in issues:
        warnings.warn(f'[CDIE config] {issue}', stacklevel=2)
        _logger.warning('[config] %s', issue)


# Run validation at import time (safe — only emits warnings/logs)
_validate_config()


# ── External YAML Configs ─────────────────────────────────────────────────────

def load_yaml_config(path: Path) -> dict:
    """Helper to load YAML configuration safely."""
    if not path.exists():
        _logger.debug('[config] YAML not found: %s', path)
        return {}
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        _logger.warning('[config] Failed to load YAML %s: %s', path, e)
        return {}


# Load CATE segments and heuristics from external files
CATE_CONFIG = load_yaml_config(PROJECT_ROOT / 'config' / 'cate_segments.yaml')
HEURISTICS_CONFIG = load_yaml_config(PROJECT_ROOT / 'config' / 'magnitude_heuristics.yaml')

