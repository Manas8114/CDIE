# CDIE v5 — How to Demo

## 📖 Demo Script: SIM Box Fraud Detection Narrative

**Scenario**: A regional telecom operator noticed a sudden 15% drop in monthly revenue and a 20% increase in network congestion. Traditional analytics identified the symptoms but couldn't pinpoint the cause or the best intervention.

**Narrative Steps**:
1. **The Investigation**: "We're showing how CDIE detected SIM box fraud for a regional operator. The operator noticed dropped calls and revenue leakage. They suspected network issues or competitor pricing."
2. **The Query**: "Instead of guessing, the operator ran a causal query: *'What happens if SIM box fraud attempts increase by 30%?'*"
3. **The Discovery**: Open the Dashboard. Show the causal graph where `SIMBoxFraudAttempts` directly drives `RevenueLeakageVolume` and `NetworkLoad`.
4. **The Recommendation**: "CDIE recommended tightening the `FraudPolicyStrictness` by 20%. The causal model estimated this would recover 85% of the leaked revenue ($1.2M/month) without significantly impacting legitimate subscriber retention."
5. **The Outcome**: "The operator implemented the policy. Three weeks later, they validated the causal prediction: revenue recovered by $1.15M (96% accuracy), and network congestion normalized. CDIE turned a correlation into a multi-million dollar recovery strategy."

---

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

# Batch Causal Analysis (Bulk Lookup)
# Efficiently analyze multiple intervention paths in a single call
curl -X POST http://localhost:8000/batch_query \
  -H "Content-Type: application/json" \
  -d '{"queries": [
    {"source": "SIMBoxFraudAttempts", "target": "RevenueLeakageVolume", "magnitude": 30},
    {"source": "FraudPolicyStrictness", "target": "SIMFraudDetectionRate", "magnitude": 20},
    {"source": "RevenueLeakageVolume", "target": "CashFlowRisk", "magnitude": 15}
  ]}'

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
