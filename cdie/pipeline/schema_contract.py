"""
CDIE v5 — Schema Contract Validator
Enforces the 12-column SCM schema with hardened validation:
  - Timestamp granularity detection
  - Value range sanity checks
  - Adversarial injection detection
  - Column alias mapping
"""

import pandas as pd
from typing import Dict, List, Tuple
from cdie.pipeline.data_generator import VARIABLE_NAMES


class SchemaContractError(Exception):
    pass


COLUMN_ALIASES: Dict[str, str] = {
    "cdr_volume": "CallDataRecordVolume",
    "call_volume": "CallDataRecordVolume",
    "records_count": "CallDataRecordVolume",
    "total_calls": "CallDataRecordVolume",
    
    "sim_box_fraud": "SIMBoxFraudAttempts",
    "simbox_attempts": "SIMBoxFraudAttempts",
    "fraud_events": "SIMBoxFraudAttempts",
    "detected_fraud": "SIMBoxFraudAttempts",
    
    "fraud_policy": "FraudPolicyStrictness",
    "policy_level": "FraudPolicyStrictness",
    "strictness_index": "FraudPolicyStrictness",
    "security_threshold": "FraudPolicyStrictness",
    
    "detection_rate": "SIMFraudDetectionRate",
    "accuracy_score": "SIMFraudDetectionRate",
    "capture_rate": "SIMFraudDetectionRate",
    "mitigation_efficiency": "SIMFraudDetectionRate",
    
    "revenue_leakage": "RevenueLeakageVolume",
    "leakage_amount": "RevenueLeakageVolume",
    "financial_loss": "RevenueLeakageVolume",
    "unbilled_revenue": "RevenueLeakageVolume",
    
    "subscriber_retention": "SubscriberRetentionScore",
    "loyalty_score": "SubscriberRetentionScore",
    "churn_rate_inverse": "SubscriberRetentionScore",
    "retention_index": "SubscriberRetentionScore",
    
    "arpu": "ARPUImpact",
    "average_revenue": "ARPUImpact",
    "billing_impact": "ARPUImpact",
    "wallet_share": "ARPUImpact",
    
    "opex": "NetworkOpExCost",
    "operating_expenditure": "NetworkOpExCost",
    "maintenance_cost": "NetworkOpExCost",
    "efficiency_drain": "NetworkOpExCost",
    
    "cash_flow_risk": "CashFlowRisk",
    "liquidity_risk": "CashFlowRisk",
    "bad_debt_exposure": "CashFlowRisk",
    "default_prob": "CashFlowRisk",
    
    "network_load": "NetworkLoad",
    "traffic_mb": "NetworkLoad",
    "utilization_pct": "NetworkLoad",
    "bandwidth_demand": "NetworkLoad",
    
    "regulatory_signal": "RegulatorySignal",
    "compliance_alert": "RegulatorySignal",
    "government_mandate": "RegulatorySignal",
    "legal_signal": "RegulatorySignal",
    
    "itu_pressure": "ITURegulatoryPressure",
    "un_sdg_pressure": "ITURegulatoryPressure",
    "global_standard_gap": "ITURegulatoryPressure",
    "compliance_burden": "ITURegulatoryPressure",
}

# Physical value constraints per variable
VALUE_RANGES: Dict[str, Tuple[float, float]] = {
    "SIMFraudDetectionRate": (0.0, 1.0),
    "FraudPolicyStrictness": (0.0, 10.0),
    "SIMBoxFraudAttempts": (0.0, float("inf")),
    "CallDataRecordVolume": (0.0, float("inf")),
    "RevenueLeakageVolume": (0.0, float("inf")),
    "NetworkLoad": (0.0, float("inf")),
    "SubscriberRetentionScore": (0.0, 1.0),
    "CashFlowRisk": (0.0, float("inf")),
}


def _apply_alias_mapping(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Map common column name variants to canonical CDIE variable names."""
    warnings = []
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower().strip().replace(" ", "_")
        if col not in VARIABLE_NAMES and col_lower in COLUMN_ALIASES:
            canonical = COLUMN_ALIASES[col_lower]
            if canonical not in df.columns:
                rename_map[col] = canonical
                warnings.append(f"Alias mapped: '{col}' -> '{canonical}'")
    if rename_map:
        df = df.rename(columns=rename_map)
    return df, warnings


def _check_timestamp_granularity(df: pd.DataFrame) -> List[str]:
    """Detect timestamp columns and warn about mixed granularities."""
    warnings: list[str] = []
    time_cols = [
        c
        for c in df.columns
        if c.lower() in ("timeindex", "time_index", "timestamp", "date", "period")
    ]
    if not time_cols:
        return warnings

    for col in time_cols:
        try:
            ts = pd.to_datetime(df[col], errors="coerce")
            valid = ts.dropna()
            if len(valid) < 2:
                continue
            diffs = valid.diff().dropna()
            unique_diffs = diffs.dt.total_seconds().round(0).unique()
            if len(unique_diffs) > 3:
                warnings.append(
                    f"TIMESTAMP_MIXED: Column '{col}' has {len(unique_diffs)} distinct intervals. "
                    f"Range: {diffs.min()} to {diffs.max()}. Temporal lags may be corrupted."
                )
        except Exception:
            pass
    return warnings


def _check_value_ranges(df: pd.DataFrame) -> List[str]:
    """Flag columns with physically impossible values."""
    warnings = []
    for col, (lo, hi) in VALUE_RANGES.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        below = (series < lo).sum()
        above = (series > hi).sum() if hi != float("inf") else 0
        if below > 0:
            warnings.append(
                f"VALUE_RANGE: '{col}' has {below} values below minimum {lo}"
            )
        if above > 0:
            warnings.append(
                f"VALUE_RANGE: '{col}' has {above} values above maximum {hi}"
            )
    return warnings


def _detect_adversarial_injection(df: pd.DataFrame, variable_names: list) -> List[str]:
    """
    Detect patterns consistent with deliberate adversarial injection:
      - Sudden zero-variance columns (designed to break positivity)
      - Extreme outlier clusters (>6 sigma) in isolated rows
      - Suspiciously identical rows (copy-paste injection)
    """
    warnings = []
    numeric_cols = [c for c in variable_names if c in df.columns]

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 10:
            continue

        # Zero variance detection
        if series.std() < 1e-10:
            warnings.append(
                f"ADVERSARIAL_SUSPECTED: '{col}' has zero variance. "
                "This will break CATL positivity and may be deliberate injection."
            )

        # Extreme outlier cluster detection (>6 sigma, >5% of data)
        mean, std = series.mean(), series.std()
        if std > 0:
            outliers = ((series - mean).abs() > 6 * std).sum()
            if outliers > len(series) * 0.05:
                warnings.append(
                    f"ADVERSARIAL_SUSPECTED: '{col}' has {outliers} extreme outliers (>6 sigma, "
                    f"{outliers / len(series) * 100:.1f}% of data). Possible injection attack."
                )

    # Duplicate row detection (>20% duplicates is suspicious)
    n_dupes = df[numeric_cols].duplicated().sum()
    dupe_pct = n_dupes / len(df) * 100 if len(df) > 0 else 0
    if dupe_pct > 20:
        warnings.append(
            f"ADVERSARIAL_SUSPECTED: {n_dupes} duplicate rows ({dupe_pct:.1f}%). "
            "Possible copy-paste injection to manipulate causal discovery."
        )

    return warnings


def validate_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Full schema contract validation pipeline.
    Returns (normalized_df, warnings_list).
    """
    warnings: List[str] = []

    # 1. Column alias mapping
    df, alias_warnings = _apply_alias_mapping(df)
    warnings.extend(alias_warnings)

    # 2. Missing columns
    missing_cols = [col for col in VARIABLE_NAMES if col not in df.columns]
    if missing_cols:
        warnings.append(f"Missing columns filled with NaN: {missing_cols}")
        for col in missing_cols:
            df[col] = float("nan")

    # 3. Extra columns (keep TimeIndex and CustomerSegment)
    allowed_cols = set(
        VARIABLE_NAMES + ["CustomerSegment", "TimeIndex", "timestamp", "date", "period"]
    )
    extra_cols = [col for col in df.columns if col not in allowed_cols]
    if extra_cols:
        warnings.append(f"Dropping extra columns: {extra_cols}")
        df = df.drop(columns=extra_cols)

    # 4. Type coercion
    for col in VARIABLE_NAMES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Timestamp granularity check
    ts_warnings = _check_timestamp_granularity(df)
    warnings.extend(ts_warnings)

    # 6. Value range checks
    range_warnings = _check_value_ranges(df)
    warnings.extend(range_warnings)

    # 7. Adversarial injection detection
    adv_warnings = _detect_adversarial_injection(df, VARIABLE_NAMES)
    warnings.extend(adv_warnings)

    return df, warnings
