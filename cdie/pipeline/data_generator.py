"""
CDIE v4 — Synthetic SCM Data Generator
Generates a 12-node telecom billing fraud structural causal model dataset
with known ground-truth causal structure using pgmpy.
Domain: Telecom SIM Box Fraud Detection (ITU AI4Good / OPEA Challenge)
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(os.environ.get("CDIE_DATA_DIR", Path(__file__).parent.parent.parent / "data"))

VARIABLE_NAMES = [
    "CallDataRecordVolume",
    "SIMBoxFraudAttempts",
    "FraudPolicyStrictness",
    "SIMFraudDetectionRate",
    "RevenueLeakageVolume",
    "SubscriberRetentionScore",
    "ARPUImpact",
    "NetworkOpExCost",
    "CashFlowRisk",
    "NetworkLoad",
    "RegulatorySignal",
    "ITURegulatoryPressure",
]

GROUND_TRUTH_EDGES = [
    ("CallDataRecordVolume", "SIMBoxFraudAttempts"),
    ("CallDataRecordVolume", "NetworkLoad"),
    ("CallDataRecordVolume", "ARPUImpact"),
    ("SIMBoxFraudAttempts", "SIMFraudDetectionRate"),
    ("SIMBoxFraudAttempts", "RevenueLeakageVolume"),
    ("FraudPolicyStrictness", "SIMFraudDetectionRate"),
    ("FraudPolicyStrictness", "NetworkOpExCost"),
    ("SIMFraudDetectionRate", "RevenueLeakageVolume"),
    ("SIMFraudDetectionRate", "SubscriberRetentionScore"),
    ("RevenueLeakageVolume", "ARPUImpact"),
    ("RevenueLeakageVolume", "CashFlowRisk"),
    ("SubscriberRetentionScore", "ARPUImpact"),
    ("RegulatorySignal", "SubscriberRetentionScore"),
    ("RegulatorySignal", "ITURegulatoryPressure"),
    ("ITURegulatoryPressure", "FraudPolicyStrictness"),
    ("NetworkOpExCost", "CashFlowRisk"),
    ("NetworkLoad", "NetworkOpExCost"),
]


def generate_ground_truth_dag():
    """Build the ground truth DAG as an adjacency dict."""
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(VARIABLE_NAMES)
    G.add_edges_from(GROUND_TRUTH_EDGES)
    assert nx.is_directed_acyclic_graph(G), "Ground truth must be a DAG"
    return G


def generate_scm_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data from the SCM using structural equations.
    Each variable is a linear combination of its parents + noise.
    """
    rng = np.random.default_rng(seed)

    data = {}

    # Exogenous roots
    data["RegulatorySignal"] = rng.normal(50, 15, n_samples)
    data["CallDataRecordVolume"] = rng.normal(1000, 200, n_samples)

    # ITURegulatoryPressure <- RegulatorySignal
    data["ITURegulatoryPressure"] = (
        0.4 * data["RegulatorySignal"]
        + rng.normal(30, 8, n_samples)
    )

    # FraudPolicyStrictness <- ITURegulatoryPressure (Discrete levels 1-5)
    raw_strictness = (
        0.5 * data["ITURegulatoryPressure"]
        + rng.normal(40, 10, n_samples)
    )
    # Map to 5 discrete levels
    data["FraudPolicyStrictness"] = pd.cut(
        raw_strictness, bins=5, labels=[1, 2, 3, 4, 5]
    ).astype(float)

    # SIMBoxFraudAttempts <- CallDataRecordVolume
    data["SIMBoxFraudAttempts"] = (
        0.03 * data["CallDataRecordVolume"]
        + rng.normal(20, 8, n_samples)
    )
    data["SIMBoxFraudAttempts"] = np.maximum(data["SIMBoxFraudAttempts"], 0)

    # NetworkLoad <- CallDataRecordVolume
    data["NetworkLoad"] = (
        0.05 * data["CallDataRecordVolume"]
        + rng.normal(30, 10, n_samples)
    )

    # SIMFraudDetectionRate <- SIMBoxFraudAttempts + FraudPolicyStrictness
    data["SIMFraudDetectionRate"] = (
        -0.3 * data["SIMBoxFraudAttempts"]
        + 0.6 * data["FraudPolicyStrictness"]
        + rng.normal(50, 10, n_samples)
    )
    data["SIMFraudDetectionRate"] = np.clip(data["SIMFraudDetectionRate"], 0, 100)

    # NetworkOpExCost <- FraudPolicyStrictness + NetworkLoad
    data["NetworkOpExCost"] = (
        0.35 * data["FraudPolicyStrictness"]
        + 0.25 * data["NetworkLoad"]
        + rng.normal(100, 20, n_samples)
    )

    # RevenueLeakageVolume <- SIMBoxFraudAttempts + SIMFraudDetectionRate
    data["RevenueLeakageVolume"] = (
        0.5 * data["SIMBoxFraudAttempts"]
        - 0.2 * data["SIMFraudDetectionRate"]
        + rng.normal(15, 5, n_samples)
    )
    data["RevenueLeakageVolume"] = np.maximum(data["RevenueLeakageVolume"], 0)

    # SubscriberRetentionScore <- SIMFraudDetectionRate + RegulatorySignal
    data["SubscriberRetentionScore"] = (
        0.3 * data["SIMFraudDetectionRate"]
        + 0.2 * data["RegulatorySignal"]
        + rng.normal(60, 10, n_samples)
    )
    data["SubscriberRetentionScore"] = np.clip(data["SubscriberRetentionScore"], 0, 100)

    # ARPUImpact <- CallDataRecordVolume + RevenueLeakageVolume + SubscriberRetentionScore
    data["ARPUImpact"] = (
        0.5 * data["CallDataRecordVolume"]
        - 2.0 * data["RevenueLeakageVolume"]
        + 1.5 * data["SubscriberRetentionScore"]
        + rng.normal(0, 30, n_samples)
    )

    # CashFlowRisk <- RevenueLeakageVolume + NetworkOpExCost
    data["CashFlowRisk"] = (
        0.4 * data["RevenueLeakageVolume"]
        + 0.3 * data["NetworkOpExCost"]
        + rng.normal(20, 8, n_samples)
    )

    df = pd.DataFrame(data)[VARIABLE_NAMES]

    # Add synthetic subscriber segment for CATE analysis
    df["CustomerSegment"] = rng.choice(
        ["Consumer", "Enterprise", "MVNO"], n_samples, p=[0.5, 0.35, 0.15]
    )

    # Add time index for temporal analysis
    df["TimeIndex"] = np.arange(n_samples)

    return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Preprocess data: impute missing values, detect outliers,
    check multicollinearity, run ADF stationarity tests.
    Returns cleaned data and preprocessing report.
    """
    report = {
        "imputed_counts": {},
        "winsorized_counts": {},
        "collinear_pairs": [],
        "adf_results": {},
        "constant_variables": [],
    }

    numeric_cols = [c for c in VARIABLE_NAMES if c in df.columns]

    # 1. Missing value imputation
    for col in numeric_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(df[col].mean())
            report["imputed_counts"][col] = int(n_missing)

    # 2. Constant variable detection
    for col in numeric_cols:
        if df[col].std() == 0:
            report["constant_variables"].append(col)

    # 3. IQR outlier winsorization
    for col in numeric_cols:
        if col in report["constant_variables"]:
            continue
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        mask = (df[col] < q1) | (df[col] > q99)
        n_winsorized = mask.sum()
        if n_winsorized > 0:
            df[col] = df[col].clip(lower=q1, upper=q99)
            report["winsorized_counts"][col] = int(n_winsorized)

    # 4. Multicollinearity detection
    corr_matrix = df[numeric_cols].corr().abs()
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            if corr_matrix.iloc[i, j] > 0.95:
                report["collinear_pairs"].append(
                    (numeric_cols[i], numeric_cols[j], float(corr_matrix.iloc[i, j]))
                )

    # 5. ADF stationarity tests
    from statsmodels.tsa.stattools import adfuller
    for col in numeric_cols:
        if col in report["constant_variables"]:
            continue
        try:
            result = adfuller(df[col].dropna(), maxlag=10)
            report["adf_results"][col] = {
                "adf_statistic": float(result[0]),
                "p_value": float(result[1]),
                "stationary": result[1] < 0.05,
            }
        except Exception:
            report["adf_results"][col] = {"adf_statistic": None, "p_value": None, "stationary": None}

    return df, report


def save_data(df: pd.DataFrame, ground_truth_dag, output_dir: Path = None):
    """Save dataset and ground truth DAG to disk."""
    out = output_dir or DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "scm_data.csv", index=False)
    with open(out / "ground_truth.pkl", "wb") as f:
        pickle.dump(ground_truth_dag, f)

    return out / "scm_data.csv", out / "ground_truth.pkl"


def run(output_dir: Path = None) -> tuple[pd.DataFrame, dict, object]:
    """Main entry point: generate, preprocess, save."""
    print("[DataGen] Generating 12-node telecom billing fraud SCM...")
    dag = generate_ground_truth_dag()
    df = generate_scm_data()
    print(f"[DataGen] Generated {len(df)} samples × {len(df.columns)} columns")

    df, report = preprocess_data(df)
    print(f"[DataGen] Preprocessing complete. Winsorized: {sum(report['winsorized_counts'].values())} values")

    csv_path, dag_path = save_data(df, dag, output_dir)
    print(f"[DataGen] Saved to {csv_path} and {dag_path}")

    return df, report, dag


if __name__ == "__main__":
    run()
