import time
import statistics
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdie.api.main import app

client = TestClient(app)

ITERATIONS = 20

def run_stress_test():
    print(f"🚀 Starting CDIE v4 Stress Test ({ITERATIONS} iterations)...")
    
    endpoints = {
        "health": ("/health", "GET"),
        "metadata": ("/metadata", "GET"),
        "info": ("/info", "GET"),
        "demo-queries": ("/demo-queries", "GET"),
        "graph": ("/graph", "GET"),
        "benchmark": ("/benchmark", "GET"),
        "catl": ("/catl", "GET"),
        "xgboost": ("/xgboost", "GET"),
        "temporal": ("/temporal", "GET"),
    }

    results = {name: [] for name in endpoints}
    errors = []

    for i in range(ITERATIONS):
        print(f"\nIteration {i+1}/{ITERATIONS}: ", end="")
        for name, (path, method) in endpoints.items():
            start = time.perf_counter()
            try:
                if method == "GET":
                    response = client.get(path)
                
                elapsed = (time.perf_counter() - start) * 1000
                
                if response.status_code not in [200, 503]:
                    errors.append(f"Iteration {i+1}: {name} ({path}) failed with status {response.status_code}: {response.text}")
                    print("❌", end="")
                else:
                    results[name].append(elapsed)
                    print(".", end="")
            except Exception as e:
                errors.append(f"Iteration {i+1}: {name} ({path}) raised exception: {e}")
                print("💥", end="")

        # Test /query with varying inputs
        query_text = "What is the impact of SIM box fraud on ARPU?"
        start = time.perf_counter()
        resp = client.post("/query", json={"query": query_text})
        elapsed = (time.perf_counter() - start) * 1000
        if resp.status_code not in [200, 503]:
            errors.append(f"Iteration {i+1}: /query failed with status {resp.status_code}")
        
        # Test /prescribe
        start = time.perf_counter()
        resp = client.post("/prescribe", json={"target": "ARPUImpact", "maximize": True, "limit": 3})
        elapsed = (time.perf_counter() - start) * 1000
        if resp.status_code not in [200, 503]:
             errors.append(f"Iteration {i+1}: /prescribe failed with status {resp.status_code}")

    print("\n\n📊 Stress Test Results (Mean Latency):")
    for name, latencies in results.items():
        if latencies:
            print(f"  - {name:15}: {statistics.mean(latencies):.2f}ms")
        else:
            print(f"  - {name:15}: NO SUCCESSFUL RUNS")

    if errors:
        print("\n❌ Errors Encountered:")
        for err in errors[:10]: # Show first 10
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors)-10} more.")
    else:
        print("\n✅ All iterations completed with no 500 errors.")

    return errors

if __name__ == "__main__":
    errors = run_stress_test()
    if errors:
        sys.exit(1)
