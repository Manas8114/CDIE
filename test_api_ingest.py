import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(r'c:\Users\msgok\OneDrive\Desktop\Project\hackathon\Rename')))

from cdie.api.main import app
from cdie.pipeline.data_generator import generate_scm_data

client = TestClient(app)


def main():
    print('Generating SCM data...')
    df_gen = generate_scm_data(n_samples=100)

    csv_buffer = io.BytesIO()
    df_gen.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    print('Testing /ingest endpoint with valid data...')
    response = client.post('/ingest', files={'file': ('test.csv', csv_buffer, 'text/csv')})

    print(f'Status Code: {response.status_code}')
    print(f'Response: {response.json()}')

    assert response.status_code == 200
    assert response.json()['status'] == 'accepted'

    print('\nTesting /ingest endpoint with adversarial data (Zero variance)...')
    df_adv = df_gen.copy()
    df_adv['SIMBoxFraudAttempts'] = 0  # Zero variance

    adv_buffer = io.BytesIO()
    df_adv.to_csv(adv_buffer, index=False)
    adv_buffer.seek(0)

    response_bad = client.post('/ingest', files={'file': ('bad_data.csv', adv_buffer, 'text/csv')})

    print(f'Status Code: {response_bad.status_code}')
    print(f'Response: {response_bad.json()}')

    assert response_bad.status_code == 200
    assert response_bad.json()['status'] == 'rejected'
    assert 'reasons' in response_bad.json()
    assert 'Positivity check failed: Zero variance detected.' in response_bad.json()['reasons']

    print('\nALL API INTEGRATION TESTS PASSED ✓')


if __name__ == '__main__':
    main()
