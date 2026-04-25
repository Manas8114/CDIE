"""
CDIE v5 — RAG Cache Module

Handles Redis connection lifecycle, deterministic cache-key generation,
and typed read/write helpers. All cache operations are fire-and-forget:
failures are logged and the caller receives ``None`` / no-op.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import json
import os
from typing import Any

from cdie.observability import METRIC_CACHE_HIT, METRIC_CACHE_MISS, get_logger, increment

log = get_logger(__name__)


def build_redis_client() -> Any:
    """Create a Redis client from ``REDIS_URL``, or return ``None`` if not configured."""
    redis_url = os.environ.get('REDIS_URL', '').strip()
    if not redis_url:
        log.info('[rag.cache] REDIS_URL not set — caching disabled')
        return None
    try:
        import redis

        client = redis.from_url(redis_url, decode_responses=True)  # type: ignore
        # Ping to verify connectivity at startup
        client.ping()
        log.info('[rag.cache] Redis connected', url=redis_url)
        return client
    except Exception as exc:
        log.warning('[rag.cache] Redis connection failed — caching disabled', error=str(exc))
        return None


def make_cache_key(
    query_type: str, 
    source: str, 
    target: str, 
    effect: dict[str, Any], 
    refutation_status: dict[str, Any] | None = None,
    temporal_info: dict[str, Any] | None = None,
) -> str:
    """Generate a deterministic SHA-256 cache key for a causal explanation query."""
    payload = {
        'query_type': query_type,
        'source': source,
        'target': target,
        'point': effect.get('point_estimate', 0),
        'refutations': refutation_status or {},
        'temporal_info': temporal_info or {},
        'engine_version': 'v5.0',
    }
    encoded = json.dumps(payload, sort_keys=True)
    return f"cdie:expl:{hashlib.sha256(encoded.encode()).hexdigest()}"


def cache_get(client: Any, key: str) -> str | None:
    """Read a cached explanation string, returning ``None`` on miss or error."""
    if client is None:
        increment(METRIC_CACHE_MISS)
        return None
    with contextlib.suppress(Exception):
        value = client.get(key)
        if value:
            increment(METRIC_CACHE_HIT)
            log.debug('[rag.cache] HIT', key=key[:24])
            return str(value)
    increment(METRIC_CACHE_MISS)
    return None


def cache_set(client: Any, key: str, value: str, ttl_seconds: int = 86400) -> None:
    """Write an explanation string to Redis with a TTL (default 24 h)."""
    if client is None or not value:
        return
    with contextlib.suppress(Exception):
        client.setex(key, ttl_seconds, value)
        log.debug('[rag.cache] SET', key=key[:24], ttl=ttl_seconds)
