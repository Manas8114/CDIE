import time
import sys
import os

print("Starting import timing...")
start = time.perf_counter()
try:
    import numpy as np
    print(f"NumPy imported in {time.perf_counter() - start:.4f}s")
except Exception as e:
    print(f"NumPy import failed: {e}")

start = time.perf_counter()
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    print(f"Scikit-learn imported in {time.perf_counter() - start:.4f}s")
except Exception as e:
    print(f"Scikit-learn import failed: {e}")

start = time.perf_counter()
try:
    sys.path.insert(0, os.getcwd())
    from cdie.api.rag.engine import ExplanationEngine
    print(f"ExplanationEngine imported in {time.perf_counter() - start:.4f}s")
    
    start = time.perf_counter()
    engine = ExplanationEngine()
    print(f"ExplanationEngine initialized in {time.perf_counter() - start:.4f}s")
except Exception as e:
    print(f"ExplanationEngine failed: {e}")
