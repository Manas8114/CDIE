"""
CDIE v5 — Universal Data Ingestion Pipeline
Handles multi-modal data ingestion (CSV, Parquet, JSON, Excel, PDF).
Routes by MIME type, applies schema contract, and produces normalized DataFrames.
"""
import io
import mimetypes
import pandas as pd
from typing import Tuple, List, BinaryIO
from cdie.pipeline.schema_contract import validate_schema

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import openpyxl  # noqa: F401
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


class DataIngestionRouter:
    @staticmethod
    def ingest(file_obj: BinaryIO, filename: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Routes the file buffer to the appropriate parser based on MIME type or extension.
        Returns the normalized DataFrame and a list of warnings.
        """
        mime_type, _ = mimetypes.guess_type(filename)
        mime_type = mime_type or ""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        warnings = []

        if mime_type == "text/csv" or ext == "csv":
            df = pd.read_csv(file_obj)

        elif ext == "parquet" or "parquet" in mime_type:
            df = pd.read_parquet(file_obj)

        elif mime_type == "application/json" or ext == "json":
            df = pd.read_json(file_obj)

        elif ext in ("xlsx", "xls"):
            if not HAS_OPENPYXL:
                raise ImportError(
                    "openpyxl is required to parse Excel files. "
                    "Install via `pip install openpyxl`."
                )
            df = pd.read_excel(file_obj, engine="openpyxl")
            warnings.append(f"Parsed Excel file: {filename}")

        elif mime_type == "application/pdf" or ext == "pdf":
            if pdfplumber is None:
                raise ImportError(
                    "pdfplumber is required to parse PDFs. "
                    "Install via `pip install pdfplumber`."
                )
            all_data = []
            with pdfplumber.open(file_obj) as pdf:
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        all_data.extend(table)

            if not all_data or len(all_data) < 2:
                raise ValueError(
                    f"No valid tabular data found in PDF: {filename}. "
                    "Needs at least a header and one row."
                )
            headers = [
                str(h).strip() if h else f"col_{i}"
                for i, h in enumerate(all_data[0])
            ]
            df = pd.DataFrame(all_data[1:], columns=headers)
            warnings.append(f"Parsed {len(df)} rows from PDF tables.")

        else:
            raise ValueError(
                f"Unsupported file type: '{ext}' (MIME: {mime_type}) for {filename}. "
                f"Supported: csv, parquet, json, xlsx, pdf"
            )

        # Enforce SCM schema contract (includes alias mapping, range checks, adversarial detection)
        df, schema_warnings = validate_schema(df)
        warnings.extend(schema_warnings)

        # Build pre-ingestion report
        adversarial_flags = [w for w in warnings if "ADVERSARIAL" in w]
        if adversarial_flags:
            warnings.insert(0,
                f"SECURITY_ALERT: {len(adversarial_flags)} adversarial injection pattern(s) detected. "
                "Review warnings before proceeding."
            )

        return df, warnings
