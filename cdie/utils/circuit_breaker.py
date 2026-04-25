"""
CDIE v5 — OPEA HTTP Client with Retry + Circuit Breaker

Wraps ``requests.post`` for all outbound calls to OPEA microservices
(TextGen, TEI Embedding, TEI Reranking) with:

- 3 automatic retries with exponential backoff (1 s → 2 s → 4 s)
- Catches ``Timeout`` and ``ConnectionError`` only (not HTTP 4xx/5xx)
- Structured log on every retry attempt
- Returns ``None`` on final failure — callers must handle ``None``

Uses only ``tenacity`` which is already in the Python ecosystem and
requires no new direct dependency (installed transitively by several
CDIE packages). If ``tenacity`` is not available, falls back to a
simple single-attempt wrapper with the same interface.

Usage::

    from cdie.utils.circuit_breaker import opea_post

    response = opea_post(f'{endpoint}/embed', json={'inputs': texts}, timeout=30)
    if response is None:
        # OPEA unavailable — fall back to TF-IDF
        ...
    response.raise_for_status()
"""

from __future__ import annotations

import logging
from typing import Any

import requests  # type: ignore[import-untyped]

from cdie.observability import METRIC_OPEA_FAIL, METRIC_OPEA_RETRY, get_logger, increment

log = get_logger(__name__)

# ── Tenacity-based retry ──────────────────────────────────────────────────────

try:
    from tenacity import (
        RetryError,
        before_sleep_log,
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    _stdlib_log = logging.getLogger(__name__)

    @retry(
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        before_sleep=before_sleep_log(_stdlib_log, logging.WARNING),
        reraise=False,
    )
    def _post_with_retry(url: str, json: Any, timeout: int) -> requests.Response:
        increment(METRIC_OPEA_RETRY)
        return requests.post(url, json=json, timeout=timeout)

    def opea_post(url: str, json: Any, timeout: int = 10) -> requests.Response | None:
        """POST to an OPEA endpoint with automatic retry on transient failures.

        Args:
            url:     Full URL including path, e.g. ``http://tei-embedding:80/embed``.
            json:    Request body as a Python dict (serialised to JSON).
            timeout: Per-attempt timeout in seconds (default 10).

        Returns:
            A ``requests.Response`` on success, or ``None`` after all retries fail.
        """
        try:
            return _post_with_retry(url, json, timeout)
        except (RetryError, requests.RequestException) as exc:
            increment(METRIC_OPEA_FAIL)
            log.warning('[circuit_breaker] OPEA call failed after retries', url=url, error=str(exc))
            return None

except ImportError:
    # tenacity not available — single attempt, same interface
    log.info('[circuit_breaker] tenacity not installed; using single-attempt OPEA calls')

    def opea_post(url: str, json: Any, timeout: int = 10) -> requests.Response | None:
        """POST to an OPEA endpoint (single attempt — tenacity not installed).

        Args:
            url:     Full URL including path.
            json:    Request body as a Python dict.
            timeout: Request timeout in seconds (default 10).

        Returns:
            A ``requests.Response`` on success, or ``None`` on network error.
        """
        try:
            return requests.post(url, json=json, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            increment(METRIC_OPEA_FAIL)
            log.warning('[circuit_breaker] OPEA call failed', url=url, error=str(exc))
            return None
