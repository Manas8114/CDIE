import io
import os
import pandas as pd
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdie.api.main import app

client = TestClient(app)

DATA_PATH = Path(__file__).parent.parent / "data" / "scm_data.csv"

def test_real_data_ingestion():
    print(f"📂 Validating ingestion with real data: {DATA_PATH}...")
    
    if not DATA_PATH.exists():
        print(f"❌ Error: {DATA_PATH} not found.")
        return False

    df = pd.read_csv(DATA_PATH)
    print(f"📊 Loaded {len(df)} rows and {len(df.columns)} columns.")

    # Convert to buffer
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    print("🚀 Triggering /ingest...")
    response = client.post(
        "/ingest",
        files={"file": ("scm_data.csv", csv_buffer, "text/csv")}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {data['status']} - {data.get('message', '')}")
        if data["status"] == "accepted":
            print("✅ Ingestion successfully accepted.")
            return True
        else:
            print(f"⚠️ Rejection reason: {data.get('reasons', []) or data.get('reason', 'Unknown')}")
            return False
    else:
        print(f"❌ Ingestion failed with status {response.status_code}: {response.text}")
        return False

if __name__ == "__main__":
    if not test_real_data_ingestion():
        sys.exit(1)
