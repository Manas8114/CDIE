"""
CDIE v4 — Synthetic SCM Data Generator
Generates a 12-node telecom billing fraud structural causal model dataset
with known ground-truth causal structure using pgmpy.
Domain: Telecom SIM Box Fraud Detection (ITU AI4Good / OPEA Challenge)
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cdie.config import DATA_DIR, GROUND_TRUTH_EDGES, VARIABLE_NAMES

__all__ = ['DATA_DIR', 'GROUND_TRUTH_EDGES', 'VARIABLE_NAMES', 'generate_ground_truth_dag', 'generate_scm_data', 'preprocess_data', 'run']


def generate_ground_truth_dag() -> Any:
    """Build the ground truth DAG as an adjacency dict."""
    import networkx as nx

    graph_gt = nx.DiGraph()
    graph_gt.add_nodes_from(VARIABLE_NAMES)
    graph_gt.add_edges_from(GROUND_TRUTH_EDGES)
    assert nx.is_directed_acyclic_graph(graph_gt), "Ground truth must be a DAG"
    return graph_gt


def generate_scm_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data from the SCM using structural equations.
    Each variable is a linear combination of its parents + noise.
    """
    rng = np.random.default_rng(seed)

    data = {}

    # Exogenous roots
    data['RegulatorySignal'] = rng.normal(50, 15, n_samples)
    data['CallDataRecordVolume'] = rng.normal(1000, 200, n_samples)

    # ITURegulatoryPressure <- RegulatorySignal
    data['ITURegulatoryPressure'] = 0.4 * data['RegulatorySignal'] + rng.normal(30, 8, n_samples)

    # FraudPolicyStrictness <- ITURegulatoryPressure (Discrete levels 1-5)
    raw_strictness = 0.5 * data['ITURegulatoryPressure'] + rng.normal(40, 10, n_samples)
    # Map to 5 discrete levels
    data['FraudPolicyStrictness'] = pd.cut(raw_strictness, bins=5, labels=[1, 2, 3, 4, 5]).astype(float)

    # SIMBoxFraudAttempts <- CallDataRecordVolume
    data['SIMBoxFraudAttempts'] = 0.03 * data['CallDataRecordVolume'] + rng.normal(20, 8, n_samples)
    data['SIMBoxFraudAttempts'] = np.maximum(data['SIMBoxFraudAttempts'], 0)

    # NetworkLoad <- CallDataRecordVolume
    data['NetworkLoad'] = 0.05 * data['CallDataRecordVolume'] + rng.normal(30, 10, n_samples)

    # SIMFraudDetectionRate <- SIMBoxFraudAttempts + FraudPolicyStrictness
    data['SIMFraudDetectionRate'] = (
        -0.3 * data['SIMBoxFraudAttempts'] + 0.6 * data['FraudPolicyStrictness'] + rng.normal(50, 10, n_samples)
    )
    data['SIMFraudDetectionRate'] = np.clip(data['SIMFraudDetectionRate'], 0, 100)

    # NetworkOpExCost <- FraudPolicyStrictness + NetworkLoad
    data['NetworkOpExCost'] = (
        0.35 * data['FraudPolicyStrictness'] + 0.25 * data['NetworkLoad'] + rng.normal(100, 20, n_samples)
    )

    # RevenueLeakageVolume <- SIMBoxFraudAttempts + SIMFraudDetectionRate
    data['RevenueLeakageVolume'] = (
        0.5 * data['SIMBoxFraudAttempts'] - 0.2 * data['SIMFraudDetectionRate'] + rng.normal(15, 5, n_samples)
    )
    data['RevenueLeakageVolume'] = np.maximum(data['RevenueLeakageVolume'], 0)

    # SubscriberRetentionScore <- SIMFraudDetectionRate + RegulatorySignal
    data['SubscriberRetentionScore'] = (
        0.3 * data['SIMFraudDetectionRate'] + 0.2 * data['RegulatorySignal'] + rng.normal(60, 10, n_samples)
    )
    data['SubscriberRetentionScore'] = np.clip(data['SubscriberRetentionScore'], 0, 100)

    # ARPUImpact <- CallDataRecordVolume + RevenueLeakageVolume + SubscriberRetentionScore
    data['ARPUImpact'] = (
        0.5 * data['CallDataRecordVolume']
        - 2.0 * data['RevenueLeakageVolume']
        + 1.5 * data['SubscriberRetentionScore']
        + rng.normal(0, 30, n_samples)
    )

    # CashFlowRisk <- RevenueLeakageVolume + NetworkOpExCost
    data['CashFlowRisk'] = (
        0.4 * data['RevenueLeakageVolume'] + 0.3 * data['NetworkOpExCost'] + rng.normal(20, 8, n_samples)
    )

    df = pd.DataFrame(data)[VARIABLE_NAMES]

    # Add synthetic subscriber segment for CATE analysis
    df['CustomerSegment'] = rng.choice(['Consumer', 'Enterprise', 'MVNO'], n_samples, p=[0.5, 0.35, 0.15])

    # --- HTE enrichment: subscriber-level context attributes ---
    # SubscriberTenureMonths: longer tenure → better policy response (non-linear)
    df['SubscriberTenureMonths'] = rng.integers(1, 120, n_samples).astype(float)

    # DeviceTier: 0=Budget, 1=Mid, 2=Premium — affects detection capability
    df['DeviceTier'] = rng.integers(0, 3, n_samples).astype(float)

    # RegionalRiskScore: regional fraud baseline 0-100
    df['RegionalRiskScore'] = np.clip(rng.normal(50, 20, n_samples), 0, 100)

    # --- Inject non-linear heterogeneous interactions ---
    # Enterprise segments respond 1.8x better to policy tightening
    enterprise_mask = df['CustomerSegment'] == 'Enterprise'
    mvno_mask = df['CustomerSegment'] == 'MVNO'

    interaction_boost = np.where(enterprise_mask, 1.8, np.where(mvno_mask, 0.6, 1.0))
    tenure_boost = 1 + 0.004 * df['SubscriberTenureMonths']  # linear tenure effect

    df['SIMFraudDetectionRate'] = np.clip(df['SIMFraudDetectionRate'] * interaction_boost * tenure_boost, 0, 100)
    # High-risk regions amplify revenue leakage
    region_factor = 1 + 0.006 * df['RegionalRiskScore']
    df['RevenueLeakageVolume'] = np.maximum(df['RevenueLeakageVolume'] * region_factor, 0)

    # Add time index for temporal analysis
    df['TimeIndex'] = np.arange(n_samples)

    return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Preprocess data: impute missing values, detect outliers,
    check multicollinearity, run ADF stationarity tests.
    Returns cleaned data and preprocessing report.
    """
    report: dict[str, Any] = {
        'imputed_counts': {},
        'winsorized_counts': {},
        'collinear_pairs': [],
        'adf_results': {},
        'constant_variables': [],
    }

    numeric_cols = [c for c in VARIABLE_NAMES if c in df.columns]

    # 1. Missing value imputation
    for col in numeric_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(df[col].mean())
            report['imputed_counts'][col] = int(n_missing)

    # 2. Constant variable detection
    for col in numeric_cols:
        if df[col].std() == 0:
            report['constant_variables'].append(col)

    # 3. IQR outlier winsorization
    for col in numeric_cols:
        if col in report['constant_variables']:
            continue
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        mask = (df[col] < q1) | (df[col] > q99)
        n_winsorized = mask.sum()
        if n_winsorized > 0:
            df[col] = df[col].clip(lower=q1, upper=q99)
            report['winsorized_counts'][col] = int(n_winsorized)

    # 4. Multicollinearity detection
    corr_matrix = df[numeric_cols].corr().abs()
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            if corr_matrix.iloc[i, j] > 0.95:
                report['collinear_pairs'].append((numeric_cols[i], numeric_cols[j], float(corr_matrix.iloc[i, j])))

    # 5. ADF stationarity tests
    from statsmodels.tsa.stattools import adfuller

    for col in numeric_cols:
        if col in report['constant_variables']:
            continue
        try:
            result = adfuller(df[col].dropna(), maxlag=10)
            report['adf_results'][col] = {
                'adf_statistic': float(result[0]),
                'p_value': float(result[1]),
                'stationary': result[1] < 0.05,
            }
        except Exception:
            report['adf_results'][col] = {
                'adf_statistic': None,
                'p_value': None,
                'stationary': None,
            }

    return df, report


def run(output_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any], Any]:
    """Main entry point: generate, preprocess, save."""
    from cdie.pipeline.datastore import DataStoreManager

    print('[DataGen] Generating 12-node telecom billing fraud SCM...')
    dag = generate_ground_truth_dag()
    df = generate_scm_data()
    print(f'[DataGen] Generated {len(df)} samples × {len(df.columns)} columns')

    df, report = preprocess_data(df)
    print(f'[DataGen] Preprocessing complete. Winsorized: {sum(report["winsorized_counts"].values())} values')

    store = DataStoreManager(master_csv_path=(output_dir or DATA_DIR) / 'scm_data.csv')
    csv_path, dag_path = store.save_pipeline_data(df, dag)
    print(f'[DataGen] Saved to {csv_path} and {dag_path}')

    return df, report, dag


if __name__ == '__main__':
    run()
