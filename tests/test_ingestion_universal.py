import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Provide standard project path access
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdie.pipeline.data_ingestion import DataIngestionRouter
from cdie.pipeline.datastore import DataStoreManager


@pytest.fixture
def store_manager(tmp_path):
    master_csv_path = tmp_path / 'test_scm_data.csv'
    return DataStoreManager(master_csv_path=master_csv_path)


def test_json_flattening():
    nested_json = {
        'records': [
            {'cdr_volume': 1000, 'simbox_attempts': 20, 'extra': 'info'},
            {'cdr_volume': 1100, 'simbox_attempts': 25, 'extra': 'info'},
        ]
    }
    file_obj = io.BytesIO(json.dumps(nested_json).encode())
    df, warnings = DataIngestionRouter.ingest(file_obj, 'test.json')

    assert 'CallDataRecordVolume' in df.columns
    assert 'SIMBoxFraudAttempts' in df.columns
    assert len(df) == 2
    assert any('flattening' in w for w in warnings)


def test_fuzzy_pdf_extraction(monkeypatch):
    # Mocking a text-based PDF format using a regex-friendly string
    dummy_text = 'CallVolume    SIMBoxAttempts    NetworkUtilization\n1000    50    0.8\n1200    60    0.85'

    class FakePage:
        def extract_text(self):
            return dummy_text

    class FakePDF:
        def __init__(self):
            self.pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class MockPdfPlumber:
        def open(self, *args):
            return FakePDF()

    # Use monkeypatch to replace pdfplumber in the ingestion module
    import cdie.pipeline.data_ingestion as di

    monkeypatch.setattr(di, 'pdfplumber', MockPdfPlumber())

    file_obj = io.BytesIO(b'fake_pdf_content')
    df = DataIngestionRouter._extract_fuzzy_table(file_obj)

    assert not df.empty
    assert len(df) >= 2
    # Fuzzy parsing should have col_0, col_1, col_2 initially
    assert 'col_0' in df.columns


def test_data_merging_and_deduplication(store_manager):
    # 1. First ingest
    df1 = pd.DataFrame(
        {
            'CallDataRecordVolume': [1000, 2000],
            'SIMBoxFraudAttempts': [10, 20],
            'CustomerSegment': ['Consumer', 'Consumer'],
        }
    )

    merged1, warnings1 = store_manager.merge_to_master(df1)
    assert len(merged1) == 2
    assert store_manager.master_csv_path.exists()

    # 2. Duplicate ingest
    df2 = pd.DataFrame(
        {
            'CallDataRecordVolume': [1000, 3000],
            'SIMBoxFraudAttempts': [10, 30],
            'CustomerSegment': ['Consumer', 'Consumer'],
        }
    )

    merged2, warnings2 = store_manager.merge_to_master(df2)
    # Total should be 3 (1000 is duplicate, 3000 is new)
    assert len(merged2) == 3
    assert any('duplicate' in w for w in warnings2)


def test_schema_alignment(store_manager):
    # Uploading data with missing columns should pad with NaN
    partial_df = pd.DataFrame({'CallDataRecordVolume': [1500], 'ARPUImpact': [0.5]})

    merged, warnings = store_manager.merge_to_master(partial_df)
    assert 'SIMBoxFraudAttempts' in merged.columns
    assert merged['SIMBoxFraudAttempts'].isna().all()
