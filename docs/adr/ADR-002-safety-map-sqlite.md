# ADR-002 — Pre-computed Safety Map with SQLite Backend

**Date:** 2026-04-24  
**Status:** Accepted  
**Deciders:** CDIE v5 Architecture Team

---

## Context

CDIE must answer causal queries (intervention, counterfactual, root-cause, temporal)
**with sub-200 ms latency** in a production API. The causal inference pipeline
(GFCI + DoWhy + EconML DML) takes 5–15 minutes on a fresh dataset.

---

## Decision

Pre-compute all validated causal scenarios offline and store them in a **Safety Map**:
a structured index of ≈ 2 000 scenarios (12 variables × 10 edges × 16 magnitude levels).

The Safety Map has a **dual storage backend**:

```
Offline pipeline
   └── build_safety_map() → safety_map.json (canonical, always written)
                          ↓
                    _write_sqlite_safety_map()
                          ↓
         runtime/safety_map.runtime.db  (primary lookup, O(log n) queries)
                          ↓
         data/safety_map.db             (project mirror, for sharing/Git LFS)
```

At API startup, `SafetyMapLookup.load()` tries paths in order:
`data/safety_map.db` → `runtime/safety_map.runtime.db` → `data/safety_map.json`.

---

## Alternatives Considered

| Approach | Latency | Complexity | Why Rejected |
|----------|---------|------------|--------------|
| **Live inference per query** | 5–15 min | Low | Unacceptable for production |
| **In-memory dict cache** | < 1 ms | Low | Lost on restart; too large for RAM in full scenarios |
| **PostgreSQL** | ~5 ms | High | Overkill for read-only lookup; adds infra dependency |
| **DuckDB** | < 5 ms | Medium | No significant advantage over SQLite for this scale |
| **Pure JSON (no SQLite)** | ~50 ms (full parse) | Low | JSON file grows to ~40 MB; SQLite allows indexed queries |

---

## Consequences

**Positive:**
- Sub-5 ms p95 lookup latency (benchmarked: see `/benchmark/latency`).
- Zero inference cost at query time — all computation is offline.
- SQLite is a single file with no server, easy to back up and ship.
- JSON fallback ensures the API degrades gracefully even without SQLite.

**Negative:**
- Safety Map becomes stale when distribution shifts. Mitigated by:
  - KS-test staleness check on every query (`SafetyMapLookup.check_staleness`)
  - Drift Analyzer tracking DAG snapshots across pipeline runs
- Scenarios are discrete (16 magnitude levels). Queries with non-standard magnitudes
  use nearest-neighbour matching or heuristic interpolation.

---

## References

- `cdie/pipeline/safety_map.py` — builder
- `cdie/api/lookup.py` — query-time lookup
- `cdie/api/drift.py` — staleness tracking
- `cdie/runtime.py` — path resolution
