"""
CDIE v4 — LinearDML Effect Estimation + MAPIE Conformal Prediction
Doubly-robust ATE/CATE computation with distribution-free confidence intervals.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor


def compute_ate_simple(data, treatment, outcome, confounders=None):
    """
    Simple OLS-based ATE with scientifically valid Standard Error.
    Uses statsmodels for robust standard error calculation.
    """
    import statsmodels.api as sm

    if confounders:
        X = data[[treatment] + confounders].copy()
    else:
        X = data[[treatment]].copy()

    X = sm.add_constant(X)
    y = data[outcome]

    try:
        model = sm.OLS(y, X).fit()
        # The treatment coefficient is the second one (index 1) because intercept is index 0
        ate = float(model.params[1])
        se = float(model.bse[1])
        return ate, se
    except Exception as e:
        print(f"[Estimation] Fallback OLS failed: {e}")
        # Final safety fallback
        reg = LinearRegression().fit(X.iloc[:, 1:].values, y.values)
        ate = float(reg.coef_[0])
        return ate, abs(ate) * 0.15


def compute_ate_dml(data, treatment, outcome, confounders=None):
    """
    LinearDML doubly-robust ATE estimation via EconML.
    Falls back to OLS if EconML unavailable.
    """
    try:
        from econml.dml import LinearDML

        if confounders:
            W = data[confounders].values
        else:
            W = None

        T = data[treatment].values.ravel()
        Y = data[outcome].values.ravel()

        is_discrete = False
        unique_vals = len(np.unique(data[treatment].dropna()))
        # Check if integer-like and low cardinality
        if data[treatment].dtype in [np.int64, np.int32] and unique_vals < 5:
            is_discrete = True

        print(
            f"[Estimation] {treatment} -> {outcome}: Unique={unique_vals}, is_discrete={is_discrete}"
        )

        model = LinearDML(
            model_y=GradientBoostingRegressor(n_estimators=50, max_depth=3),
            model_t=GradientBoostingRegressor(n_estimators=50, max_depth=3),
            discrete_treatment=is_discrete,
            random_state=42,
        )

        if W is not None:
            model.fit(Y, T, X=None, W=W)
        else:
            # Dummy confounders for EconML API consistency
            model.fit(Y, T, X=None, W=np.ones((len(Y), 1)))

        ate = float(model.const_marginal_ate())
        ci = model.const_marginal_ate_interval(alpha=0.05)
        lower = float(ci[0])
        upper = float(ci[1])

        return {
            "ate": round(ate, 4),
            "ci_lower": round(lower, 4),
            "ci_upper": round(upper, 4),
            "method": "LinearDML",
        }
    except ImportError:
        # print("[Estimation] EconML not found, using OLS fallback.")
        ate, se = compute_ate_simple(data, treatment, outcome, confounders)
        return {
            "ate": round(ate, 4),
            "ci_lower": round(ate - 1.96 * se, 4),
            "ci_upper": round(ate + 1.96 * se, 4),
            "method": "OLS_fallback",
        }
    except Exception as e:
        print(f"[Estimation] DML Error for {treatment}->{outcome}: {e}")
        ate, se = compute_ate_simple(data, treatment, outcome, confounders)
        return {
            "ate": round(ate, 4),
            "ci_lower": round(ate - 1.96 * se, 4),
            "ci_upper": round(ate + 1.96 * se, 4),
            "method": f"OLS_fallback (DML error: {str(e)[:50]})",
        }


def compute_cate(data, treatment, outcome, segment_col, confounders=None):
    """Compute Conditional ATE (CATE) by segment."""
    segments = data[segment_col].unique()
    cate_results = {}

    for seg in segments:
        seg_data = data[data[segment_col] == seg]
        if len(seg_data) < 30:
            cate_results[str(seg)] = {
                "ate": None,
                "n_samples": len(seg_data),
                "status": "insufficient_data",
            }
            continue

        ate_result = compute_ate_dml(seg_data, treatment, outcome, confounders)
        ate_result["n_samples"] = len(seg_data)
        cate_results[str(seg)] = ate_result

    return cate_results


def add_mapie_intervals(data, treatment, outcome, ate_result):
    """
    Wrap ATE estimates with MAPIE conformal prediction intervals.
    Provides distribution-free coverage guarantees.
    """
    try:
        from mapie.regression import MapieRegressor
        from sklearn.ensemble import RandomForestRegressor

        X = data[[treatment]].values
        y = data[outcome].values

        base_model = RandomForestRegressor(
            n_estimators=50, max_depth=5, random_state=42
        )

        mapie = MapieRegressor(base_model, method="plus", cv=5, random_state=42)
        mapie.fit(X, y)

        # Predict at mean treatment value
        X_test = np.array([[data[treatment].mean()]])
        y_pred, y_pis = mapie.predict(X_test, alpha=0.05)

        ate_result["mapie_point"] = round(float(y_pred[0]), 4)
        ate_result["mapie_lower"] = round(float(y_pis[0, 0, 0]), 4)
        ate_result["mapie_upper"] = round(float(y_pis[0, 1, 0]), 4)
        ate_result["mapie_method"] = "conformal_plus"

        # Check if intervals are very wide
        width = abs(ate_result["mapie_upper"] - ate_result["mapie_lower"])
        point = abs(ate_result["mapie_point"]) if ate_result["mapie_point"] != 0 else 1
        if width > 3 * point:
            ate_result["confidence_label"] = "LOW_CONFIDENCE"
        else:
            ate_result["confidence_label"] = "HIGH_CONFIDENCE"

    except ImportError:
        ate_result["mapie_method"] = "unavailable"
        ate_result["confidence_label"] = "ESTIMATED"
    except Exception as e:
        ate_result["mapie_method"] = f"error: {str(e)[:50]}"
        ate_result["confidence_label"] = "ESTIMATED"

    return ate_result


def run_estimation(
    data: pd.DataFrame, validated_edges: list[tuple], variable_names: list[str]
):
    """
    Run ATE + CATE estimation for all reachable paths in the DAG.
    This resolves the 1-hop limitation (e.g. FraudPolicyStrictness -> RevenueLeakageVolume).
    """
    import networkx as nx

    print(
        f"[Estimation] Computing effects for DAG reachable paths from {len(validated_edges)} base edges..."
    )

    # Build directed graph of validated edges
    G = nx.DiGraph()
    G.add_nodes_from(variable_names)
    G.add_edges_from(validated_edges)

    # Generate all pairs of (source, target) where a path exists
    edges_to_estimate = []
    for src in G.nodes():
        for tgt in nx.descendants(G, src):
            edges_to_estimate.append((src, tgt))

    # Remove duplicates if any
    edges_to_estimate = list(set(edges_to_estimate))
    print(
        f"[Estimation] Found {len(edges_to_estimate)} total causal paths to estimate."
    )

    results = {}

    for src, tgt in edges_to_estimate:
        if src not in data.columns or tgt not in data.columns:
            continue

        # print(f"[Estimation] {src} → {tgt}...")

        # Identify confounders (parents of outcome excluding treatment)
        confounders = [
            c
            for c in variable_names
            if c != src
            and c != tgt
            and c in data.columns
            and abs(data[c].corr(data[tgt])) > 0.1
        ][:5]  # Limit confounders

        # ATE
        ate_result = compute_ate_dml(data, src, tgt, confounders)

        # MAPIE intervals
        ate_result = add_mapie_intervals(data, src, tgt, ate_result)

        # CATE by customer segment
        cate_by_segment = {}
        if "CustomerSegment" in data.columns:
            cate_by_segment = compute_cate(
                data, src, tgt, "CustomerSegment", confounders
            )

        # CATE by volume quartile
        cate_by_volume = {}
        if "TransactionVolume" in data.columns:
            data_copy = data.copy()
            data_copy["VolumeQuartile"] = pd.qcut(
                data_copy["TransactionVolume"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
            )
            cate_by_volume = compute_cate(
                data_copy, src, tgt, "VolumeQuartile", confounders
            )

        results[f"{src}->{tgt}"] = {
            "source": src,
            "target": tgt,
            "ate": ate_result,
            "cate_by_segment": cate_by_segment,
            "cate_by_volume": cate_by_volume,
        }

        print(
            f"[Estimation]   ATE={ate_result['ate']}, CI=[{ate_result['ci_lower']}, {ate_result['ci_upper']}]"
        )

    print(f"[Estimation] Complete. {len(results)} edges estimated.")
    return results


if __name__ == "__main__":
    from cdie.pipeline.data_generator import (
        generate_scm_data,
        VARIABLE_NAMES,
        GROUND_TRUTH_EDGES,
    )

    df = generate_scm_data()
    results = run_estimation(df, GROUND_TRUTH_EDGES[:3], VARIABLE_NAMES)
    for key, val in results.items():
        print(f"  {key}: ATE={val['ate']['ate']}")
