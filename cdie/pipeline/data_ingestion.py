"""
CDIE v5 — Universal Data Ingestion Pipeline
Handles multi-modal data ingestion (CSV, Parquet, JSON, Excel, PDF).
Routes by MIME type, applies schema contract, and produces normalized DataFrames.
"""

import contextlib
import mimetypes
from typing import Any, BinaryIO, cast

import pandas as pd

from cdie.pipeline.schema_contract import validate_schema

with contextlib.suppress(ImportError):
    import pdfplumber

try:
    import openpyxl  # noqa: F401

    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


class DataIngestionRouter:
    @staticmethod
    def _flatten_json(data: Any) -> pd.DataFrame:
        """Recursive flattening for nested JSON telemetry data."""
        if isinstance(data, list):
            return pd.json_normalize(data)
        elif isinstance(data, dict):
            # Check if it's a 'records' or 'data' wrapper
            for key in ['records', 'data', 'results']:
                if key in data and isinstance(data[key], list):
                    return pd.json_normalize(data[key])
            return pd.json_normalize([data])
        return pd.DataFrame()

    @staticmethod
    def _extract_fuzzy_table(pdf_path: Any) -> pd.DataFrame:
        """
        Fallback for PDFs without explicit tables.
        Attempts to parse whitespace-separated text into columns.
        """
        import re

        rows = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                for line in text.split('\n'):
                    # Split by 2+ spaces or tabs
                    parts = re.split(r'\s{2,}|\t+', line.strip())
                    if len(parts) > 1:
                        rows.append(parts)

        if not rows:
            return pd.DataFrame()

        # Determine max columns
        max_cols = max(len(r) for r in rows)
        headers = [f'col_{i}' for i in range(max_cols)]
        return pd.DataFrame([r + [None] * (max_cols - len(r)) for r in rows], columns=headers)

    @staticmethod
    def ingest_from_sql(uri: str, query: str) -> tuple[pd.DataFrame, list[str]]:
        """Ingest data from a SQL database."""
        from sqlalchemy import create_engine

        engine = create_engine(uri)
        df = pd.read_sql(query, engine)

        # Enforce schema contract
        df, warnings = validate_schema(df)
        return df, warnings

    @staticmethod
    def ingest(file_obj: BinaryIO, filename: str) -> tuple[pd.DataFrame, list[str]]:
        """
        Routes the file buffer to the appropriate parser based on MIME type or extension.
        Returns the normalized DataFrame and a list of warnings.
        """
        import json

        mime_type, _ = mimetypes.guess_type(filename)
        mime_type = mime_type or ''
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

        warnings = []
        df = pd.DataFrame()

        try:
            if mime_type == 'text/csv' or ext == 'csv':
                df = pd.read_csv(file_obj)

            elif ext == 'parquet' or 'parquet' in mime_type:
                df = pd.read_parquet(file_obj)

            elif mime_type == 'application/json' or ext == 'json':
                raw_data = json.load(file_obj)
                df = DataIngestionRouter._flatten_json(raw_data)
                warnings.append(f'Applied deep JSON flattening to {filename}')

            elif ext in ('xlsx', 'xls'):
                if not HAS_OPENPYXL:
                    raise ImportError('openpyxl required for Excel.')
                df = pd.read_excel(file_obj, engine='openpyxl')
                warnings.append(f'Parsed Excel file: {filename}')

            elif mime_type == 'application/pdf' or ext == 'pdf':
                if pdfplumber is None:
                    raise ImportError('pdfplumber required for PDFs.')

                # 1. Try tabular extraction
                all_data = []
                with pdfplumber.open(cast(Any, file_obj)) as pdf:
                    for page in pdf.pages:
                        table = page.extract_table()
                        if table:
                            all_data.extend(table)

                if all_data and len(all_data) >= 2:
                    headers = [str(h).strip() if h else f'col_{i}' for i, h in enumerate(all_data[0])]
                    df = pd.DataFrame(all_data[1:], columns=headers)
                    warnings.append(f'Extracted {len(df)} rows from PDF tables.')
                else:
                    # 2. Try fuzzy text extraction
                    file_obj.seek(0)
                    df = DataIngestionRouter._extract_fuzzy_table(file_obj)
                    if not df.empty:
                        warnings.append(f'Recovered {len(df)} rows via fuzzy PDF parsing.')
                    else:
                        raise ValueError(f'No parseable data found in PDF: {filename}')

            else:
                raise ValueError(f'Unsupported type: {ext} for {filename}.')

        except Exception as e:
            raise ValueError(f"Ingestion failed for {filename}: {str(e)}") from e

        # Enforce SCM schema contract
        df, schema_warnings = validate_schema(df)
        warnings.extend(schema_warnings)

        return df, warnings
