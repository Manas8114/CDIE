import time
import psutil
import os
import numpy as np
import pandas as pd
from typing import Any

def get_mem() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def benchmark_scaling():
    print("=== CDIE Scaling Benchmark ===")
    print(f"Base Memory: {get_mem():.2f} MB")
    
    scales = [1000, 10000, 50000, 100000]
    results = []
    
    for n in scales:
        print(f"\n--- Testing Scale: {n:,} rows ---")
        
        # Generation
        start_gen = time.perf_counter()
        df = pd.DataFrame({
            'source': np.random.choice(['FraudAttempts', 'ARPU', 'ChurnRate', 'NetworkLoad'], n),
            'target': np.random.choice(['Revenue', 'SafetyScore', 'Uptime'], n),
            'magnitude_value': np.random.uniform(-1, 1, n),
            'point_estimate': np.random.uniform(0, 5, n)
        })
        end_gen = time.perf_counter()
        gen_time = end_gen - start_gen
        gen_mem = get_mem()
        print(f"Data Generation: {gen_time:.4f}s | Memory: {gen_mem:.2f} MB")
        
        # Simulation of "Inference" (Searching for closest scenario)
        # In CDIE, this would be find_best_scenario
        start_inf = time.perf_counter()
        # Mock search: find mean point_estimate for a source/target
        _ = df[(df['source'] == 'FraudAttempts') & (df['target'] == 'Revenue')]['point_estimate'].mean()
        end_inf = time.perf_counter()
        inf_time = end_inf - start_inf
        print(f"Mock Inference: {inf_time:.4f}s")
        
        results.append({
            'scale': n,
            'gen_time': gen_time,
            'gen_mem': gen_mem,
            'inf_time': inf_time
        })
        
        del df # Clean up for next scale
        
    print("\n=== Results Summary ===")
    print(f"{'Scale':<10} | {'Gen Time':<10} | {'Memory (MB)':<12} | {'Inf Time':<10}")
    for r in results:
        print(f"{r['scale']:<10,} | {r['gen_time']:<10.4f} | {r['gen_mem']:<12.2f} | {r['inf_time']:<10.4f}")

if __name__ == "__main__":
    benchmark_scaling()
