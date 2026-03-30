import sys
from pathlib import Path
import io
import pandas as pd

sys.path.insert(0, str(Path(r"c:\Users\msgok\OneDrive\Desktop\Project\hackathon\Rename")))

from cdie.pipeline.data_generator import generate_scm_data
from cdie.pipeline.data_ingestion import DataIngestionRouter
from cdie.pipeline.catl import run_catl
from cdie.pipeline.data_generator import VARIABLE_NAMES

def main():
    print("--- 1. Generating SCM data ---")
    df_gen = generate_scm_data(n_samples=100)
    
    csv_buffer = io.BytesIO()
    df_gen.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    print("--- 2. Testing Data Ingestion Router ---")
    df_ingested, warnings = DataIngestionRouter.ingest(csv_buffer, "test_data.csv")
    print(f"Original shape: {df_gen.shape}")
    print(f"Ingested shape: {df_ingested.shape}")
    print(f"Warnings: {warnings}")

    print("\n--- 3. Testing CATL Positivity Gatekeeper (Pass) ---")
    results = run_catl(df_ingested, VARIABLE_NAMES)
    print(f"Positivity Status: {results['positivity']['status']}")
    print(f"Positivity Tooltip: {results['positivity']['tooltip']}")

    print("\n--- 4. Testing Adversarial Data (Zero-Variance Fail) ---")
    df_adv = df_ingested.copy()
    df_adv['SIMBoxFraudAttempts'] = 0  # Force zero variance
    
    adv_buffer = io.BytesIO()
    df_adv.to_csv(adv_buffer, index=False)
    adv_buffer.seek(0)
    
    df_bad, _ = DataIngestionRouter.ingest(adv_buffer, "bad_data.csv")
    bad_results = run_catl(df_bad, VARIABLE_NAMES)
    
    print(f"Positivity Status: {bad_results['positivity']['status']}")
    print(f"Positivity Tooltip: {bad_results['positivity']['tooltip']}")

if __name__ == "__main__":
    main()
