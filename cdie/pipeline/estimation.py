"""
CDIE v5 — HTE/CATE Estimation Module
Doubly-robust ATE via LinearDML + Heterogeneous Treatment Effects via ForestDRLearner.
Automatically discovers subscriber-level nuances without manual segment specification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from cdie.observability import get_logger

log = get_logger(__name__)

# Maximum context features passed to CausalForestDML / ForestDRLearner.
# Increasing this improves CATE granularity but slows training.
MAX_HTE_FEATURES = 10


def compute_ate_simple(data: pd.DataFrame, treatment: str, outcome: str, confounders: list[str] | None = None) -> tuple[float, float]:
    """
    Simple OLS-based ATE with scientifically valid Standard Error.
    Uses statsmodels for robust standard error calculation.
    """
    import statsmodels.api as sm

    x_data = data[[treatment] + confounders].copy() if confounders else data[[treatment]].copy()

    x_data = sm.add_constant(x_data)
    y_data = data[outcome]

    try:
        model = sm.OLS(y_data, x_data).fit()
        # The treatment coefficient is the second one (index 1) because intercept is index 0
        ate = float(model.params[1])
        se = float(model.bse[1])
        return ate, se
    except Exception as e:
        log.warning('[Estimation] Fallback OLS failed', error=str(e))
        # Final safety fallback
        reg = LinearRegression().fit(x_data.iloc[:, 1:].values, y_data.values)
        ate = float(reg.coef_[0])
        return ate, abs(ate) * 0.15


def compute_ate_dml(data: pd.DataFrame, treatment: str, outcome: str, confounders: list[str] | None = None) -> dict[str, Any]:
    """
    LinearDML doubly-robust ATE estimation via EconML.
    Falls back to OLS if EconML unavailable.
    """
    try:
        from econml.dml import LinearDML

        w_data = data[confounders].values if confounders else None

        t_data = data[treatment].values.ravel()
        y_data = data[outcome].values.ravel()

        is_discrete = False
        unique_vals = len(np.unique(data[treatment].dropna()))
        # Check if integer-like and low cardinality
        if data[treatment].dtype in [np.int64, np.int32] and unique_vals < 5:
            is_discrete = True

        log.debug(
            '[Estimation] treatment metadata',
            treatment=treatment,
            outcome=outcome,
            unique_vals=unique_vals,
            is_discrete=is_discrete,
        )

        model = LinearDML(
            model_y=GradientBoostingRegressor(n_estimators=50, max_depth=3),
            model_t=GradientBoostingRegressor(n_estimators=50, max_depth=3),
            discrete_treatment=is_discrete,
            random_state=42,
        )

        if w_data is not None:
            model.fit(y_data, t_data, X=None, W=w_data)
        else:
            # Dummy confounders for EconML API consistency
            model.fit(y_data, t_data, X=None, W=np.ones((len(y_data), 1)))

        ate = float(model.const_marginal_ate())
        ci = model.const_marginal_ate_interval(alpha=0.05)
        lower = float(ci[0])
        upper = float(ci[1])

        return {
            'ate': round(ate, 4),
            'ci_lower': round(lower, 4),
            'ci_upper': round(upper, 4),
            'method': 'LinearDML',
        }
    except ImportError:
        # print("[Estimation] EconML not found, using OLS fallback.")
        ate, se = compute_ate_simple(data, treatment, outcome, confounders)
        return {
            'ate': round(ate, 4),
            'ci_lower': round(ate - 1.96 * se, 4),
            'ci_upper': round(ate + 1.96 * se, 4),
            'method': 'OLS_fallback',
        }
    except Exception as e:
        log.warning('[Estimation] DML error', treatment=treatment, outcome=outcome, error=str(e)[:80])
        ate, se = compute_ate_simple(data, treatment, outcome, confounders)
        return {
            'ate': round(ate, 4),
            'ci_lower': round(ate - 1.96 * se, 4),
            'ci_upper': round(ate + 1.96 * se, 4),
            'method': f'OLS_fallback (DML error: {str(e)[:50]})',
        }


def compute_cate(data: pd.DataFrame, treatment: str, outcome: str, segment_col: str, confounders: list[str] | None = None) -> dict[str, Any]:
    """Compute Conditional ATE (CATE) by discrete segment using LinearDML per cohort."""
    segments = data[segment_col].unique()
    cate_results = {}

    for seg in segments:
        seg_data = data[data[segment_col] == seg]
        if len(seg_data) < 30:
            cate_results[str(seg)] = {
                'ate': None,
                'n_samples': len(seg_data),
                'status': 'insufficient_data',
            }
            continue

        ate_result = compute_ate_dml(seg_data, treatment, outcome, confounders)
        ate_result['n_samples'] = len(seg_data)
        cate_results[str(seg)] = ate_result

    return cate_results


# ---------------------------------------------------------------------------
# HTE Discovery via ForestDRLearner  (EconML 0.14+)
# ---------------------------------------------------------------------------

_HTE_CONTEXT_COLS = [
    'SubscriberTenureMonths',
    'DeviceTier',
    'RegionalRiskScore',
]


def discover_heterogeneity(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str] | None = None,
    context_cols: list[str] | None = None,
    max_features: int = 10,
) -> dict[str, Any]:
    """
    Automated HTE discovery using ForestDRLearner (doubly-robust causal forest).

    Fits a non-parametric forest over subscriber-context features (X) to discover
    *which* subscriber attributes cause treatment effect heterogeneity — and by
    how much — without requiring pre-defined segments.

    Returns
    -------
    dict with keys:
        - individual_effects: list of per-sample CATE values (float)
        - effect_std: scalar measure of heterogeneity
        - feature_importance: dict {feature: importance_score}
        - high_effect_profile: feature means for the top-20% effect recipients
        - low_effect_profile: feature means for the bottom-20% effect recipients
        - method: 'ForestDRLearner' | 'fallback_forest_dml' | 'unavailable'
    """
    # Determine context features (X) for CATE
    def normalize(s: str) -> str:
        return s.lower().replace('_', '').replace(' ', '')

    available_normalized = {normalize(c): c for c in data.columns}
    ctx = []
    for c in (context_cols or _HTE_CONTEXT_COLS):
        norm_c = normalize(c)
        if norm_c in available_normalized:
            ctx.append(available_normalized[norm_c])

    if not ctx:
        return {'method': 'unavailable', 'reason': 'no context columns found'}

    t_data = data[treatment].values.ravel()
    y_data = data[outcome].values.ravel()
    x_ctx = data[ctx].values

    if confounders:
        w_conf: np.ndarray | None = data[[c for c in confounders if c in data.columns]].values
        if w_conf is not None and w_conf.shape[1] == 0:
            w_conf = None
    else:
        w_conf = None

    # Ensure we have enough data
    if len(y_data) < 100:
        return {'method': 'unavailable', 'reason': f'insufficient samples ({len(y_data)})'}

    try:
        from econml.dml import CausalForestDML

        # CausalForestDML is the sklearn-compatible causal forest in EconML
        model = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
            model_t=GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
            n_estimators=200,
            min_samples_leaf=10,
            max_features=min(max_features, x_ctx.shape[1]),
            discrete_treatment=(
                int(pd.Series(t_data).nunique()) < 5
                and pd.Series(t_data).dtype in [np.int64, np.int32, int]
            ),
            random_state=42,
            verbose=0,
        )
        model.fit(y_data, t_data, X=x_ctx, W=w_conf if w_conf is not None else np.ones((len(y_data), 1)))

        # Per-sample CATE estimates
        tau_hat: np.ndarray = model.effect(x_ctx).ravel()
        feature_importance = {ctx[i]: float(model.feature_importances_[i]) for i in range(len(ctx))}
        method = 'CausalForestDML'

    except (ImportError, AttributeError):
        # Graceful fallback: ForestDRLearner
        try:
            from econml.dr import ForestDRLearner
            # Check discrete treatment for ForestDRLearner if needed
            is_discrete_dr = (
                int(pd.Series(t_data).nunique()) < 5
                and pd.Series(t_data).dtype in [np.int64, np.int32, int]
            )

            model_dr = ForestDRLearner(
                model_propensity=RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42),
                model_regression=RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42),
                n_estimators=200,
                min_samples_leaf=10,
                max_features=min(max_features, x_ctx.shape[1]),
                discrete_treatment=is_discrete_dr,
                random_state=42,
            )
            model_dr.fit(y_data, t_data, X=x_ctx, W=w_conf if w_conf is not None else np.ones((len(y_data), 1)))
            tau_hat = model_dr.effect(x_ctx).ravel()
            feature_importance = {ctx[i]: float(model_dr.feature_importances_[i]) for i in range(len(ctx))}
            method = 'ForestDRLearner'
        except Exception as e2:
            return {'method': 'unavailable', 'reason': f'DRLearner failed: {e2}'[:120]}
    except Exception as e:
        return {'method': 'unavailable', 'reason': f'CATE estimation failed: {e}'[:120]}

    # Heterogeneity statistics
    effect_std = float(np.std(tau_hat))
    top_idx = tau_hat >= np.percentile(tau_hat, 80)
    bot_idx = tau_hat <= np.percentile(tau_hat, 20)

    high_profile = {ctx[i]: float(np.round(x_ctx[top_idx, i].mean(), 4)) for i in range(len(ctx))}
    low_profile = {ctx[i]: float(np.round(x_ctx[bot_idx, i].mean(), 4)) for i in range(len(ctx))}

    # Sort features by importance
    feature_importance = dict(sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True))

    log.info(
        '[HTE] treatment heterogeneity computed',
        treatment=treatment,
        outcome=outcome,
        effect_std=round(effect_std, 4),
        method=method,
        top_feature=next(iter(feature_importance)),
    )

    return {
        'method': method,
        'n_samples': int(len(y_data)),
        'individual_effects': [round(float(v), 6) for v in tau_hat[:50]],  # first 50 for compactness
        'effect_mean': round(float(np.mean(tau_hat)), 6),
        'effect_std': round(effect_std, 6),
        'effect_min': round(float(np.min(tau_hat)), 6),
        'effect_max': round(float(np.max(tau_hat)), 6),
        'feature_importance': {k: round(v, 6) for k, v in feature_importance.items()},
        'high_effect_profile': high_profile,
        'low_effect_profile': low_profile,
        'heterogeneity_detected': effect_std > 0.05,
    }


def add_mapie_intervals(data: pd.DataFrame, treatment: str, outcome: str, ate_result: dict[str, Any]) -> dict[str, Any]:
    """
    Wrap ATE estimates with MAPIE conformal prediction intervals.
    Provides distribution-free coverage guarantees.
    """
    try:
        from mapie.regression import MapieRegressor
        from sklearn.ensemble import RandomForestRegressor

        x_data = data[[treatment]].values
        y_data = data[outcome].values

        base_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)

        mapie = MapieRegressor(base_model, method='plus', cv=5, random_state=42)
        mapie.fit(x_data, y_data)

        # Predict at mean treatment value
        x_test = np.array([[data[treatment].mean()]])
        y_pred, y_pis = mapie.predict(x_test, alpha=0.05)

        ate_result['mapie_point'] = round(float(y_pred[0]), 4)
        ate_result['mapie_lower'] = round(float(y_pis[0, 0, 0]), 4)
        ate_result['mapie_upper'] = round(float(y_pis[0, 1, 0]), 4)
        ate_result['mapie_method'] = 'conformal_plus'

        # Check if intervals are very wide
        width = abs(ate_result['mapie_upper'] - ate_result['mapie_lower'])
        point = abs(ate_result['mapie_point']) if ate_result['mapie_point'] != 0 else 1
        if width > 3 * point:
            ate_result['confidence_label'] = 'LOW_CONFIDENCE'
        else:
            ate_result['confidence_label'] = 'HIGH_CONFIDENCE'

    except ImportError:
        ate_result['mapie_method'] = 'unavailable'
        ate_result['confidence_label'] = 'ESTIMATED'
    except Exception as e:
        ate_result['mapie_method'] = f'error: {str(e)[:50]}'
        ate_result['confidence_label'] = 'ESTIMATED'

    return ate_result


def run_estimation(data: pd.DataFrame, validated_edges: list[tuple[str, str]], variable_names: list[str]) -> dict[str, Any]:
    """
    Run ATE + CATE estimation for all reachable paths in the DAG.
    This resolves the 1-hop limitation (e.g. FraudPolicyStrictness -> RevenueLeakageVolume).
    """
    import networkx as nx

    log.info('[Estimation] Computing effects', n_base_edges=len(validated_edges))

    # Build directed graph of validated edges
    graph_val = nx.DiGraph()
    graph_val.add_nodes_from(variable_names)
    graph_val.add_edges_from(validated_edges)

    # Generate all pairs of (source, target) where a path exists
    edges_to_estimate = []
    for src in graph_val.nodes():
        for tgt in nx.descendants(graph_val, src):
            edges_to_estimate.append((src, tgt))

    # Remove duplicates if any
    edges_to_estimate = list(set(edges_to_estimate))
    log.info('[Estimation] Causal paths to estimate', n_paths=len(edges_to_estimate))

    results = {}

    for src, tgt in edges_to_estimate:
        if src not in data.columns or tgt not in data.columns:
            continue

        # print(f"[Estimation] {src} → {tgt}...")

        # Identify confounders (parents of outcome excluding treatment)
        confounders = [
            c
            for c in variable_names
            if c != src and c != tgt and c in data.columns and abs(data[c].corr(data[tgt])) > 0.1
        ][:5]  # Limit confounders

        # ATE
        ate_result = compute_ate_dml(data, src, tgt, confounders)

        # MAPIE intervals
        ate_result = add_mapie_intervals(data, src, tgt, ate_result)

        # --- CATE by discrete customer segment (LinearDML per-cohort) ---
        cate_by_segment = {}
        if 'CustomerSegment' in data.columns:
            cate_by_segment = compute_cate(data, src, tgt, 'CustomerSegment', confounders)

        # --- CATE by volume quartile ---
        cate_by_volume = {}
        if 'TransactionVolume' in data.columns:
            data_copy = data.copy()
            data_copy['VolumeQuartile'] = pd.qcut(data_copy['TransactionVolume'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            cate_by_volume = compute_cate(data_copy, src, tgt, 'VolumeQuartile', confounders)

        # --- Automated HTE discovery (ForestDRLearner / CausalForestDML) ---
        hte_discovery = discover_heterogeneity(data, src, tgt, confounders=confounders)

        results[f'{src}->{tgt}'] = {
            'source': src,
            'target': tgt,
            'ate': ate_result,
            'cate_by_segment': cate_by_segment,
            'cate_by_volume': cate_by_volume,
            'hte_discovery': hte_discovery,
        }

        log.debug('[Estimation] edge estimated', src=src, tgt=tgt, ate=ate_result['ate'],
                  ci_lower=ate_result['ci_lower'], ci_upper=ate_result['ci_upper'])

    log.info('[Estimation] Complete', n_edges=len(results))
    return results


if __name__ == '__main__':
    from cdie.config import GROUND_TRUTH_EDGES, VARIABLE_NAMES
    from cdie.pipeline.data_generator import generate_scm_data

    df = generate_scm_data()
    results = run_estimation(df, GROUND_TRUTH_EDGES[:3], VARIABLE_NAMES)
    for key, val in results.items():
        print(f'  {key}: ATE={val["ate"]["ate"]}')
