# CDIE v5 — How to Demo

## Quick Demo (Recommended — no Docker, <100 MB RAM)

Runs in seconds on any machine with Python 3.11+. Uses pre-computed artifacts.

```bash
pip install fastapi pydantic pandas numpy scipy psutil scipy
cd cdie-v5
python -c "
import sys, json, time, os
sys.path.insert(0, os.getcwd())
from pathlib import Path
DATA_DIR = Path('data')
from cdie.api.lookup import SafetyMapLookup

lookup = SafetyMapLookup(safety_map_path=str(DATA_DIR / 'safety_map.json'))
if not lookup.loaded:
    print('Error: safety_map not found. Run the full pipeline first.')
    sys.exit(1)

queries = [
    ('SIMBoxFraudAttempts', 'ARPUImpact', 30),
    ('FraudPolicyStrictness', 'SIMFraudDetectionRate', 20),
    ('RevenueLeakageVolume', 'CashFlowRisk', 15),
]

print('=' * 70)
print('  CDIE v5 — Quick Demo')
print('=' * 70)
print(f'\nSafety Map loaded: {len(lookup.json_store.get(\"scenarios\", {}))} scenarios')

for src, tgt, mag in queries:
    start = time.perf_counter()
    result = lookup.find_best_scenario(src, tgt, mag)
    elapsed = (time.perf_counter() - start) * 1000
    eff = result.get('effect', {}) if result else {}
    print(f'\nQuery: What if {src} increases by {mag}%?')
    print(f'  -> {tgt} will shift by {eff.get(\"point_estimate\", \"N/A\")}')
    print(f'     95% CI: [{eff.get(\"ci_lower\", \"N/A\")}, {eff.get(\"ci_upper\", \"N/A\")}]')
    print(f'     Lookup time: {elapsed:.2f} ms')

print('\n' + '=' * 70)
"
```

**Expected output:**

```
======================================================================
  CDIE v5 — Quick Demo
======================================================================

Safety Map loaded: 640 scenarios

Query: What if SIMBoxFraudAttempts increases by 30%?
  -> ARPUImpact will shift by X.XXX
     95% CI: [X.XXX, X.XXX]
     Lookup time: 0.XX ms

Query: What if FraudPolicyStrictness increases by 20%?
  -> SIMFraudDetectionRate will shift by X.XXX
     95% CI: [X.XXX, X.XXX]
     Lookup time: 0.XX ms

Query: What if RevenueLeakageVolume increases by 15%?
  -> CashFlowRisk will shift by X.XXX
     95% CI: [X.XXX, X.XXX]
     Lookup time: 0.XX ms
```

## Full Pipeline (Requires 64 GB RAM)

The offline 8-step pipeline (Data → CATL → GFCI → Granger → DoWhy → LinearDML → MAPIE → Safety Map) is memory-intensive (~57 GB peak).

### With Docker
```bash
docker compose up -d
docker compose logs -f pipeline
```

### Without Docker
```bash
pip install -r requirements.txt
python -m cdie.pipeline.run_pipeline
```

## API Queries (Requires Docker or `uvicorn`)

```bash
# Start the API
python -m uvicorn cdie.api.main:app --host 0.0.0.0 --port 8000

# Benchmark endpoints
curl http://localhost:8000/benchmark/performance
curl http://localhost:8000/benchmark/latency
curl http://localhost:8000/benchmark/hardware

# Causal query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What happens if SIM box fraud attempts increase by 30%?"}'
```

## Frontend

```bash
cd frontend && npm install && npm run dev
# Open http://localhost:3000
```

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM (Quick Demo) | 2 GB | 4 GB |
| RAM (Full Pipeline) | 58 GB | 64 GB+ |
| CPU | 4-core x86 | 12-core Intel with AVX-512 |
| GPU | Not required | Optional (TGI benefits from GPU) |
| Storage | 2 GB | 10 GB (for Docker images) |
