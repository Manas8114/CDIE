"""
CDIE v5 — Statefull Data Store Manager
Manages the master dataset (scm_data.csv), handling merging, 
schema alignment, and deduplication of incoming data streams.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from cdie.pipeline.data_generator import VARIABLE_NAMES, DATA_DIR

class DataStoreManager:
    def __init__(self, master_path: Optional[Path] = None):
        self.master_path = master_path or (DATA_DIR / "scm_data.csv")
        self.master_path.parent.mkdir(parents=True, exist_ok=True)

    def load_master(self) -> pd.DataFrame:
        """Load the current master dataset."""
        if self.master_path.exists():
            return pd.read_csv(self.master_path)
        return pd.DataFrame(columns=VARIABLE_NAMES + ["CustomerSegment", "TimeIndex"])

    def merge_data(self, new_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Merges new data into the master store.
        - Higher-level schema alignment
        - Deduplication
        - Persistence
        """
        warnings = []
        master_df = self.load_master()

        # 1. Ensure new_df has all required columns (padded with NaN if missing)
        for col in master_df.columns:
            if col not in new_df.columns:
                new_df[col] = None
        
        # 2. Select only relevant columns in correct order
        new_df = new_df[master_df.columns]

        # 3. Deduplication (Row-based)
        initial_len = len(new_df)
        new_df = new_df.reset_index(drop=True)
        
        if not master_df.empty:
            # 3.1. Identify feature subset for duplicate detection
            subset_cols = [c for c in VARIABLE_NAMES if c in master_df.columns and c in new_df.columns]
            
            # Combine temporarily to find duplicates
            combined = pd.concat([master_df, new_df], ignore_index=True)
            
            # 3.2. Compare rows as normalized strings to avoid dtype-dependent NA warnings.
            comp_df = combined[subset_cols].astype("string").fillna("__MISSING__")
            
            # Find rows in new_df that are already in master_df
            is_dup = comp_df.duplicated(keep='first')
            new_rows_mask = ~is_dup[len(master_df):]
            # Use .values to avoid index alignment issues
            new_df = new_df[new_rows_mask.values]
            
            dup_count = initial_len - len(new_df)
            if dup_count > 0:
                warnings.append(f"Ignored {dup_count} duplicate rows already present in master store.")

        # 4. Append and Save
        if master_df.empty:
            result_df = new_df.copy()
        else:
            result_df = pd.concat([master_df, new_df], ignore_index=True)
        
        # 5. Handle TimeIndex (ensure it remains sequential if missing)
        if "TimeIndex" in result_df.columns:
            if result_df["TimeIndex"].isnull().any():
                result_df["TimeIndex"] = range(len(result_df))

        result_df.to_csv(self.master_path, index=False)
        
        warnings.append(f"Master store updated: {len(master_df)} -> {len(result_df)} rows.")
        return result_df, warnings

    def purge(self):
        """Reset the master store (delete the file)."""
        if self.master_path.exists():
            os.remove(self.master_path)
            return True
        return False
