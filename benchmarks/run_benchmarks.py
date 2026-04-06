import sys, os, json, time, psutil, statistics, platform

sys.path.insert(0, os.getcwd())

from pathlib import Path
DATA_DIR = Path('data')
os.environ.setdefault('CDIE_RUNTIME_DIR', os.path.join(os.getcwd(), 'cdie_runtime'))

from cdie.api.lookup import SafetyMapLookup

lookup = SafetyMapLookup(safety_map_path=str(DATA_DIR / 'safety_map.json'))

if not lookup.loaded:
    print('ERROR: Safety map not loaded')
    sys.exit(1)

# --- Lookup Latency ---
test_pairs = [
    ('SIMBoxFraudAttempts', 'ARPUImpact', 30),
    ('FraudPolicyStrictness', 'SIMFraudDetectionRate', 20),
    ('RevenueLeakageVolume', 'CashFlowRisk', 15),
    ('CallDataRecordVolume', 'NetworkLoad', 25),
]

latencies = []
for src, tgt, mag in test_pairs:
    start = time.perf_counter()
    result = lookup.find_best_scenario(src, tgt, mag)
    elapsed = (time.perf_counter() - start) * 1000
    latencies.append(elapsed)
    print(f"  {src} -> {tgt} (+{mag}%): {elapsed:.2f} ms")

print(f'\n=== Safety Map Lookup Latency ({len(latencies)} queries) ===')
print(f'  Mean:    {statistics.mean(latencies):.2f} ms')
print(f'  Median:  {statistics.median(latencies):.2f} ms')
sorted_lat = sorted(latencies)
print(f'  P95:     {sorted_lat[-1]:.2f} ms')
print(f'  Max:     {max(latencies):.2f} ms')
print(f'  Min:     {min(latencies):.2f} ms')

# --- Memory ---
proc = psutil.Process()
mem_mb = proc.memory_info().rss / 1024 / 1024
print(f'\n=== Memory ===')
print(f'  Process RSS: {mem_mb:.1f} MB')

# --- Training Data ---
import pandas as pd
df = pd.read_csv(DATA_DIR / 'scm_data.csv')
print(f'\n=== Training Data ===')
print(f'  Rows: {len(df):,}')
print(f'  Variables: {len(df.columns)}')

# --- Safety Map Stats ---
with open(DATA_DIR / 'safety_map.json') as f:
    smap = json.load(f)

own = smap['benchmarks'].get('own_scm', {})
ref = smap['refutation_summary']
disc = smap.get('discovery_metadata', {})

print(f'\n=== Causal Discovery (own SCM ground truth) ===')
print(f'  Precision: {own.get("precision", 0):.1%}')
print(f'  Recall:    {own.get("recall", 0):.1%}')
print(f'  F1 Score:  {own.get("f1", 0):.1%}')
print(f'  SHD:       {own.get("shd", 0)}')
print(f'  TP: {own.get("tp", 0)}, FP: {own.get("fp", 0)}, FN: {own.get("fn", 0)}')

print(f'\n=== Refutation Suite ===')
print(f'  Pass rate:   {ref["pass_rate"]:.1%}')
print(f'  Validated:   {ref["validated_count"]}')
print(f'  Quarantined: {ref["quarantined_count"]}')

print(f'\n=== Pipeline Stats ===')
print(f'  Pre-computed scenarios: {len(smap.get("scenarios", {}))}')
print(f'  Graph edges: {len(smap["graph"]["edges"])}')
print(f'  Edges discovered: {disc.get("n_edges_discovered", 0)}')

print(f'\n=== Artifact Sizes ===')
for path in [
    DATA_DIR / 'safety_map.db',
    DATA_DIR / 'safety_map.json',
    DATA_DIR / 'scm_data.csv',
]:
    sz = path.stat().st_size
    if sz > 1024*1024:
        print(f'  {path.name}: {sz / 1024 / 1024:.1f} MB')
    else:
        print(f'  {path.name}: {sz / 1024:.0f} KB')

print(f'\n=== Hardware ===')
print(f'  OS: {platform.system()} {platform.release()}')
print(f'  Processor: {platform.processor()}')
print(f'  Python: {platform.python_version()}')
print(f'  DNNL_MAX_CPU_ISA: {os.environ.get("DNNL_MAX_CPU_ISA", "not set")}')
print(f'  OMP_NUM_THREADS: {os.environ.get("OMP_NUM_THREADS", "not set")}')
