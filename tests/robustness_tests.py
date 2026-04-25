import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdie.api.main import app

client = TestClient(app)


def test_robustness():
    print('🛡️ Running Robustness and Impact Analysis...')

    # 1. Malformed JSON to /query
    print('\n[Robustness] Sending malformed JSON to /query...')
    try:
        response = client.post('/query', content="{'invalid': 'json'}")
        if response.status_code == 422:
            print('✅ Handled correctly (422) - Local impact.')
        else:
            print(f'⚠️ Unexpected status {response.status_code}: {response.text}')
    except Exception as e:
        print(f'❌ System Crash detected: {e} - SYSTEM-WIDE impact.')

    # 2. Empty query
    print('\n[Robustness] Sending empty query to /query...')
    response = client.post('/query', json={'query': ''})
    if response.status_code == 400:
        print('✅ Handled correctly (400) - Local impact.')
    else:
        print(f'⚠️ Unexpected status {response.status_code}: {response.text}')

    # 3. Invalid variable names to /prescribe
    print('\n[Robustness] Sending invalid variable to /prescribe...')
    response = client.post('/prescribe', json={'target': 'NON_EXISTENT_VAR', 'maximize': True, 'limit': 3})
    if response.status_code == 200:
        # Should resolve it via fuzzy matching or return an empty list
        print('✅ Handled correctly (200 with fallback) - Local impact.')
    elif response.status_code == 500:
        print('❌ System Crash (500) - Potential SYSTEM-WIDE impact.')
    else:
        print(f'⚠️ Status {response.status_code}')

    # 4. Large CSV Ingestion (checking memory limit)
    import io

    import pandas as pd

    print('\n[Robustness] Testing large CSV ingestion...')
    large_df = pd.DataFrame({'A': list(range(10000)), 'B': [i * 2 for i in range(10000)]})
    csv_buffer = io.BytesIO()
    large_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    try:
        response = client.post('/ingest', files={'file': ('too_large.csv', csv_buffer, 'text/csv')})
        if response.status_code == 200:
            print('✅ Handled correctly (accepted or rejected gracefully) - Local impact.')
        else:
            print(f'⚠️ Status {response.status_code}: {response.text}')
    except Exception as e:
        print(f'❌ System Crash: {e} - SYSTEM-WIDE impact.')

    print('\n🏁 Robustness tests completed.')


if __name__ == '__main__':
    test_robustness()
