import json
import os
import sqlite3

from cdie.config import DATA_DIR

# 1. Read safety_map.json directly
json_path = DATA_DIR / 'safety_map.json'
with open(json_path) as f:
    smap = json.load(f)

print('=== Safety Map Overview ===')
print(f'Version: {smap.get("version", "?")}')
print(f'Variables: {smap.get("n_variables", 0)}')
print(f'Scenarios pre-computed: {len(smap.get("scenarios", {}))}')
print(f'Graph edges: {len(smap.get("graph", {}).get("edges", []))}')

# Discovery info
disc = smap.get('discovery_metadata', {})
print(f'Discovery algorithm: {disc.get("algorithm", "?")}')
print(f'Edges discovered: {disc.get("n_edges_discovered", 0)}')

# Refutation summary
ref = smap.get('refutation_summary', {})
print(f'Refutation pass rate: {ref.get("pass_rate", 0):.1%}')
print(f'Validated edges: {ref.get("validated_count", 0)}')
print(f'Quarantined edges: {ref.get("quarantined_count", 0)}')

# Benchmarks
bench = smap.get('benchmarks', {})
print('\n=== Benchmark Results ===')
if bench:
    for k, v in bench.items():
        print(f'  {k}: {v}')
else:
    print('  (empty)')

# Sample scenarios (first 3)
scenarios = smap.get('scenarios', {})
print('\n=== Sample Scenarios (first 5) ===')
for _i, (sid, s) in enumerate(list(scenarios.items())[:5]):
    eff = s.get('effect', {})
    print(
        f'  {sid}: ATE={eff.get("ate_used", 0):.4f}, '
        f'point={eff.get("point_estimate", 0):.4f}, '
        f'CI=[{eff.get("ci_lower", 0):.4f}, {eff.get("ci_upper", 0):.4f}]'
    )

# 2. File sizes
print('\n=== File Sizes ===')
for name in ['safety_map.db', 'safety_map.json', 'scm_data.csv', 'ground_truth.pkl']:
    p = DATA_DIR / name
    if os.path.exists(p):
        sz = os.path.getsize(p)
        if sz > 1024 * 1024:
            print(f'  {name}: {sz / 1024 / 1024:.1f} MB')
        else:
            print(f'  {name}: {sz / 1024:.1f} KB')

# 3. DB schema
print('\n=== SQLite tables ===')
conn = sqlite3.connect(DATA_DIR / 'safety_map.db')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
for t in c.fetchall():
    c.execute(f'SELECT COUNT(*) FROM [{t[0]}]')
    n = c.fetchone()[0]
    print(f'  {t[0]}: {n} rows')
conn.close()
