# CDIE v5 Operational Runbook

This guide outlines standard operating procedures for maintaining and troubleshooting the Causal Decision Intelligence Engine (CDIE) v5.

## 1. System Architecture Overview
CDIE consists of three main tiers:
- **Pipeline**: Offline causal discovery and Safety Map generation.
- **API**: High-throughput FastAPI server for causal lookups.
- **UI**: Streamlit-based executive dashboard.

## 2. Service Monitoring & Health Checks
Monitor the system health via the following endpoints:
- **API Health**: `GET /health` - Returns status of API and connected OPEA services.
- **Metrics**: `GET /metrics` - Prometheus-compatible structured metrics.
- **Log Level**: Default is `INFO`. Set `LOG_LEVEL=DEBUG` for deep troubleshooting.

## 3. Pipeline Recovery (Stale Safety Map)
If the UI displays a **"Causal Drift Detected"** warning or the `ks_statistic` is high (>0.2):
1. Ensure the raw data in `data/` is up to date.
2. Run the full pipeline to re-calibrate:
   ```powershell
   python -m cdie.pipeline.run_pipeline
   ```
3. Verify the new hash in the UI `Audit Ribbon`.

## 4. Service Restarts
If the API hangs or OPEA services are disconnected:
1. **Local Mode**:
   ```powershell
   # Kill stuck processes
   taskkill /F /IM python.exe
   # Restart
   ./start_local.cmd
   ```
2. **Docker Mode**:
   ```bash
   docker-compose restart api
   ```

## 5. Troubleshooting Common Issues
- **Redis Connection Failed**: Check `REDIS_URL` env var. If down, system falls back to in-memory rate limiting.
- **OPEA Offline**: Verify `OPEA_EMBEDDING_ENDPOINT` and `OPEA_LLM_ENDPOINT`. Ensure Docker containers for TEI and TGI are running.
- **JSON Serialization Errors**: Ensure all custom types are cast to native Python types before API response.

## 6. Database Maintenance
The Safety Map is stored in `data/safety_map.db`. 
- **Backup**: `cp data/safety_map.db data/safety_map.db.bak`
- **Integrity Check**: The API automatically verifies the SHA-256 hash at startup.
