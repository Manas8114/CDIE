# CDIE v5 — Troubleshooting Guide

This guide covers the most common failure modes and their resolutions.

---

## 1. Safety Map Not Loading

**Symptoms:** API returns `503 Service Unavailable` on `/query`, `/graph`, or `/prescribe`.
Health endpoint shows `"safety_map_loaded": false`.

**Cause:** The offline pipeline has not been run or its output is in the wrong location.

**Fix:**
```bash
# Run the full offline pipeline (takes 5–15 min)
python -m cdie.pipeline.run_pipeline

# Verify output
ls data/safety_map.json     # canonical JSON
ls data/safety_map.db       # SQLite mirror
```

**Advanced:** If you see `SQLite save skipped` in logs, the pipeline ran but SQLite failed.
The API will still load from `safety_map.json` (slower startup). Check `CDIE_RUNTIME_DIR` path.

---

## 2. OPEA Services Unreachable

**Symptoms:** Logs show `[circuit_breaker] OPEA call failed after retries`. 
Explanations fall back to templates (still functional).

**Cause:** OPEA microservices not started, wrong endpoint URL, or port conflict.

**Port Reference:**

| Service | Container Port | Host Port | Variable |
|---------|---------------|-----------|----------|
| API | 8000 | 8000 | — |
| OPEA TextGen | 9000 | **8888** | `OPEA_LLM_ENDPOINT` points to container port 9000 |
| TGI Backend | 80 | 8008 | `TGI_ENDPOINT` |
| TEI Embedding | 80 | 6006 | `OPEA_EMBEDDING_ENDPOINT` |
| TEI Reranking | 80 | 8808 | `OPEA_RERANKING_ENDPOINT` |
| Redis | 6379 | 6379 | `REDIS_URL` |

> **Note:** `OPEA_LLM_ENDPOINT=http://localhost:9000` in `.env` is for local testing.
> Inside Docker Compose, services communicate on container ports:
> `OPEA_LLM_ENDPOINT=http://opea-llm-textgen:9000`.

**Fix:**
```bash
# Check running services
docker compose ps

# Check logs for a specific service
docker compose logs opea-llm-textgen --tail=50

# Restart a failed service
docker compose restart opea-llm-textgen
```

---

## 3. Windows Path / OneDrive Conflicts

**Symptoms:** SQLite errors like `disk I/O error`, `database is locked`, or pipeline crashes
when `data/` is inside an OneDrive-synced folder.

**Cause:** OneDrive (and other sync tools) can hold file locks on `.db` files during sync,
which SQLite interprets as corruption.

**Fix:** Set `CDIE_RUNTIME_DIR` to a local (non-synced) path in your `.env`:
```bash
# Windows
CDIE_RUNTIME_DIR=C:\Temp\cdie

# WSL / Linux
CDIE_RUNTIME_DIR=/tmp/cdie
```

If you see a startup warning `CDIE_RUNTIME_DIR looks like a cloud-sync path`, the system
has already detected this issue. The API still works via the `safety_map.json` fallback.

---

## 4. Port Conflicts

**Symptoms:** `docker compose up` fails with `address already in use` or `bind: address already in use`.

**Common conflicts:**

| Port | Common culprit |
|------|---------------|
| 8000 | Another FastAPI/uvicorn instance |
| 3000 | Another Next.js dev server |
| 6379 | System Redis instance |
| 8888 | Jupyter Notebook |

**Fix:**
```bash
# Find the conflicting process
netstat -ano | findstr :8000   # Windows
lsof -i :8000                  # Linux/Mac

# Kill the process (Windows)
taskkill /PID <pid> /F

# Or override the port in docker-compose.yml:
ports:
  - "8001:8000"  # map host 8001 → container 8000
```

---

## 5. Pipeline Memory Errors

**Symptoms:** `MemoryError` or OOM during `run_pipeline.py`, especially in GFCI discovery
or XGBoost/SHAP steps.

**Cause:** Default synthetic dataset is 5 000 rows × 12 columns — well within 4 GB.
This usually occurs with custom large datasets.

**Fix:**
1. Reduce dataset size in `cdie/pipeline/data_generator.py` (`N_SAMPLES = 1000`)
2. Set `OMP_NUM_THREADS=2` to reduce parallelism overhead
3. Run with GFCI timeout reduced: set `timeout_seconds=30` in `gfci_discovery.py`
4. If GFCI still fails, it falls back to PC algorithm automatically

---

## 6. Frontend Cannot Connect to API

**Symptoms:** Browser shows CORS errors or `Failed to fetch`.

**Fix:**
1. Ensure `NEXT_PUBLIC_API_URL` in `.env.local` matches the API URL:
   ```bash
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```
2. Ensure `ALLOWED_ORIGINS` in the API env includes the frontend URL:
   ```bash
   ALLOWED_ORIGINS=http://localhost:3000
   ```
3. Check that the API is running: `curl http://localhost:8000/health`

---

## 7. Slow First API Response

**Symptoms:** First `/query` request takes > 5 seconds.

**Cause:** Cold start — Safety Map loads from SQLite, TF-IDF index is built, optional OPEA
embedding index is constructed.

**This is expected.** Subsequent requests are sub-100 ms (served from the in-process cache).

For production, pre-warm the API with `GET /health` after deployment:
```bash
curl -f http://localhost:8000/health
```

---

## 8. `METRIC_OPEA_FAIL` Counter Increasing

**Check:** `GET /metrics` returns `{"metrics": {"opea.fail": N}, ...}`.

This means OPEA services are failing. See [Section 2](#2-opea-services-unreachable).
The system continues to function via template-based explanations.

---

## Getting Help

1. Check `GET /health` for the full system status payload.
2. Check `GET /metrics` for service call counters.
3. Review structured logs — search for `[WARNING]` or `[ERROR]` lines.
4. Open an issue with the output of: `docker compose logs api --tail=100`
