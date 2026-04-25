"""
CDIE v5 — Observability: Structured Logging + In-Process Metrics

Provides:
- get_logger(name) → structlog-backed logger (falls back to stdlib if not installed)
- Atomic in-process counters exposed via /metrics endpoint
- LOG_LEVEL and CDIE_ENABLE_METRICS driven by environment variables
"""

from __future__ import annotations

import logging
import os
import threading
from collections import defaultdict
from typing import Any

# ── Log Level ─────────────────────────────────────────────────────────────────

_LOG_LEVEL_STR = os.environ.get('LOG_LEVEL', 'INFO').upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_STR, logging.INFO)

# ── Structlog Setup ───────────────────────────────────────────────────────────

_STRUCTLOG_AVAILABLE = False

try:
    import structlog

    _is_dev = os.environ.get('NODE_ENV', 'development') != 'production'

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt='iso'),
            structlog.dev.ConsoleRenderer() if _is_dev else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_LOG_LEVEL),
        logger_factory=structlog.PrintLoggerFactory(),
    )
    _STRUCTLOG_AVAILABLE = True

except ImportError:
    # Fall back to stdlib logging with a simple format
    logging.basicConfig(
        level=_LOG_LEVEL,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
    )


def get_logger(name: str) -> Any:
    """Return a structured logger for the given module name.

    Uses structlog if available, otherwise falls back to stdlib logging.
    Usage::

        log = get_logger(__name__)
        log.info('Safety Map loaded', path=str(db_path), n_scenarios=1200)
    """
    if _STRUCTLOG_AVAILABLE:
        import structlog as _sl

        return _sl.get_logger(name)
    return logging.getLogger(name)


# ── In-Process Metrics ────────────────────────────────────────────────────────

_METRICS_ENABLED = os.environ.get('CDIE_ENABLE_METRICS', '1') == '1'

#: Public alias — import this in other modules for conditional metrics logic.
METRICS_ENABLED = _METRICS_ENABLED

_counters: dict[str, int] = defaultdict(int)
_lock = threading.Lock()


def increment(metric: str, amount: int = 1) -> None:
    """Atomically increment a named counter.

    Args:
        metric: Counter name, e.g. ``"query.total"``.
        amount: Increment amount (default 1).
    """
    if not _METRICS_ENABLED:
        return
    with _lock:
        _counters[metric] += amount


def get_metrics() -> dict[str, int]:
    """Return a snapshot of all current metric counters."""
    with _lock:
        return dict(_counters)


def reset_metrics() -> None:
    """Reset all counters (used in tests)."""
    with _lock:
        _counters.clear()


# ── Convenience Counter Names (import these rather than using raw strings) ────

METRIC_QUERY_TOTAL = 'query.total'
METRIC_QUERY_EXACT = 'query.match.exact'
METRIC_QUERY_NEAREST = 'query.match.nearest'
METRIC_QUERY_FALLBACK = 'query.match.fallback'
METRIC_CACHE_HIT = 'rag.cache.hit'
METRIC_CACHE_MISS = 'rag.cache.miss'
METRIC_LLM_CALL = 'rag.llm.call'
METRIC_LLM_FAIL = 'rag.llm.fail'
METRIC_OPEA_RETRY = 'opea.retry'
METRIC_OPEA_FAIL = 'opea.fail'
