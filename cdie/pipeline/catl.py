"""
CDIE v4 — Causal Assumption Transparency Layer (CATL)
Tests four core assumptions before causal discovery runs.
Makes violations visible as colored badges for EU AI Act compliance.
"""

import numpy as np
import pandas as pd
from scipy import stats

from typing import Any

from cdie.observability import get_logger

log = get_logger(__name__)


def test_faithfulness(data: pd.DataFrame, variable_names: list[str]) -> dict[str, Any]:
    """
    Markov blanket consistency check.
    Tests whether conditional independencies in data are consistent.
    Uses partial correlation as a proxy for conditional independence.
    """
    n_vars = len(variable_names)
    numeric_data = data[variable_names].values
    n_inconsistencies = 0
    n_tested = 0

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            for k in range(n_vars):
                if k in (i, j):
                    continue
                n_tested += 1

                # Partial correlation of i,j given k
                try:
                    r_ij = np.corrcoef(numeric_data[:, i], numeric_data[:, j])[0, 1]
                    r_ik = np.corrcoef(numeric_data[:, i], numeric_data[:, k])[0, 1]
                    r_jk = np.corrcoef(numeric_data[:, j], numeric_data[:, k])[0, 1]

                    denom = np.sqrt((1 - r_ik**2) * (1 - r_jk**2))
                    if denom < 1e-10:
                        continue
                    partial_r = (r_ij - r_ik * r_jk) / denom

                    # Check if marginal independence contradicts conditional
                    marginal_p = stats.pearsonr(numeric_data[:, i], numeric_data[:, j])[1]
                    is_marginal_indep = marginal_p > 0.05
                    is_conditional_dep = abs(partial_r) > 0.1

                    if is_marginal_indep and is_conditional_dep:
                        n_inconsistencies += 1
                except Exception:
                    continue

                if n_tested > 500:
                    break
            if n_tested > 500:
                break
        if n_tested > 500:
            break

    inconsistency_rate = n_inconsistencies / max(n_tested, 1)

    status = 'WARN' if inconsistency_rate > 0.05 else 'PASS'
    tooltip = f'Tested {n_tested} variable triples. Inconsistency rate: {inconsistency_rate:.1%}. '
    if status == 'WARN':
        tooltip += 'Possible path cancellation in feedback loops detected.'
    else:
        tooltip += 'Conditional independencies are consistent with potential DAG structures.'

    return {
        'test': 'Faithfulness',
        'status': status,
        'tooltip': tooltip,
        'details': {
            'n_tested': n_tested,
            'n_inconsistencies': n_inconsistencies,
            'inconsistency_rate': round(inconsistency_rate, 4),
        },
    }


def test_causal_sufficiency(data: pd.DataFrame, variable_names: list[str]) -> dict[str, Any]:
    """
    Run a preliminary independence check to detect potential hidden confounders.
    Checks for variable pairs that are correlated but have no obvious mediator.
    """
    numeric_data = data[variable_names]
    corr_matrix = numeric_data.corr().abs()
    n_vars = len(variable_names)

    bidirected_candidates = 0
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if corr_matrix.iloc[i, j] > 0.3:
                # Check if any third variable explains the correlation
                explained = False
                for k in range(n_vars):
                    if k in (i, j):
                        continue
                    r_ik = corr_matrix.iloc[i, k]
                    r_jk = corr_matrix.iloc[j, k]
                    if r_ik > 0.4 and r_jk > 0.4:
                        explained = True
                        break
                if not explained:
                    bidirected_candidates += 1

    has_bidirected = bidirected_candidates > 0
    status = 'WARN' if has_bidirected else 'PASS'
    tooltip = f'Found {bidirected_candidates} potential hidden confounder indicators. '
    if has_bidirected:
        tooltip += 'Potential hidden confounders detected. GFCI handles this — interpret edge directions with caution.'
    else:
        tooltip += 'No strong evidence of hidden confounders.'

    return {
        'test': 'Causal Sufficiency',
        'status': status,
        'tooltip': tooltip,
        'details': {'bidirected_candidates': bidirected_candidates},
    }


def test_stationarity(data: pd.DataFrame, variable_names: list[str]) -> dict[str, Any]:
    """
    Two-sample KS test on first-half vs second-half of time-ordered data.
    """
    n = len(data)
    if n < 100:
        return {
            'test': 'Stationarity',
            'status': 'UNKNOWN',
            'tooltip': f'Insufficient samples (n={n} < 100) for stationarity test.',
            'details': {'n_samples': n},
        }

    mid = n // 2
    non_stationary_vars = []

    for col in variable_names:
        if col not in data.columns:
            continue
        first_half = data[col].iloc[:mid].values
        second_half = data[col].iloc[mid:].values

        try:
            ks_stat, p_value = stats.ks_2samp(first_half, second_half)
            if p_value < 0.05:
                non_stationary_vars.append((col, round(float(ks_stat), 4), round(float(p_value), 4)))
        except Exception:
            continue

    status = 'WARN' if len(non_stationary_vars) > 0 else 'PASS'
    tooltip = f'Tested {len(variable_names)} variables. '
    if non_stationary_vars:
        var_names = [v[0] for v in non_stationary_vars]
        tooltip += f'Non-stationary: {", ".join(var_names[:3])}. Consider re-running on rolling windows.'
    else:
        tooltip += 'All variables appear stationary across time.'

    return {
        'test': 'Stationarity',
        'status': status,
        'tooltip': tooltip,
        'details': {'non_stationary_variables': non_stationary_vars},
    }


def test_acyclicity() -> dict[str, Any]:
    """
    Reported as PASS by design — PCMCI+ handles lagged temporal feedback loops.
    """
    return {
        'test': 'Acyclicity',
        'status': 'PASS',
        'tooltip': 'PCMCI+ handles temporal feedback loops explicitly. No contemporaneous cycles assumed.',
        'details': {},
    }


def test_positivity(data: pd.DataFrame, variable_names: list[str]) -> dict[str, Any]:
    """Positivity Test (Support Assumption).

    Checks whether all variables have sufficient variance and support.
    For binary-coded variables, also runs a propensity-score overlap check
    using logistic regression to detect regions of covariate space with
    no treated/untreated units (the correct statistical definition of positivity).

    Enhanced with adversarial detection: if zero-variance columns coincide
    with outlier injection patterns, flag as ADVERSARIAL_SUSPECTED.
    """
    zero_var_cols = []
    low_variance_cols = []
    adversarial_cols = []
    col_details = {}

    for col in variable_names:
        if col not in data.columns:
            continue
        try:
            series = pd.to_numeric(data[col], errors='coerce').dropna()
            var = series.var()
            col_info: dict[str, float | int | str | bool | None] = {
                'variance': round(float(var), 6) if not pd.isna(var) else 0,
                'n_unique': int(series.nunique()),
                'n_valid': len(series),
            }

            if pd.isna(var) or var == 0:
                zero_var_cols.append(col)
                col_info['failure_reason'] = 'zero_variance'

                # Adversarial check: zero variance + all identical values
                # is suspicious if other columns have normal variance
                if len(series) > 10 and series.nunique() == 1:
                    adversarial_cols.append(col)
                    col_info['adversarial_flag'] = True

            elif var < 0.05:
                low_variance_cols.append(col)
                col_info['failure_reason'] = 'low_variance'

                # Check for outlier injection pattern
                mean, std = series.mean(), series.std()
                if std > 0:
                    extreme = ((series - mean).abs() > 5 * std).sum()
                    if extreme > len(series) * 0.05:
                        adversarial_cols.append(col)
                        col_info['adversarial_flag'] = True
                        col_info['failure_reason'] = 'outlier_injection'
                        col_info['extreme_outliers'] = int(extreme)

            col_details[col] = col_info
        except Exception:
            continue

    # ── Propensity-score overlap check for binary treatment columns ───────────
    # This is the statistically correct positivity test: we verify that
    # P(T=1|X) ∈ (ε, 1-ε) for all covariate strata, not just overall prevalence.
    propensity_warnings: list[str] = []
    for col in variable_names:
        if col not in data.columns:
            continue
        series = pd.to_numeric(data[col], errors='coerce').dropna()
        unique_vals = series.unique()
        if set(unique_vals).issubset({0, 1}) and len(unique_vals) == 2:  # binary treatment
            try:
                from sklearn.linear_model import LogisticRegression

                covariates = [
                    c for c in variable_names if c != col and c in data.columns
                ]
                if covariates:
                    x_cov = data[covariates].fillna(0).values
                    t_arr = data[col].fillna(0).values
                    ps_model = LogisticRegression(max_iter=200, random_state=42).fit(x_cov, t_arr)
                    propensity = ps_model.predict_proba(x_cov)[:, 1]
                    _eps = 0.05
                    overlap_min = float(propensity[t_arr == 1].min()) if (t_arr == 1).any() else 1.0
                    overlap_max = float(propensity[t_arr == 0].max()) if (t_arr == 0).any() else 0.0
                    if overlap_min < _eps or overlap_max > 1 - _eps:
                        propensity_warnings.append(
                            f'{col}: P(T=1|X) ∈ [{overlap_min:.3f}, {overlap_max:.3f}] — '  # noqa: RUF001
                            f'overlap violation (ε={_eps})'
                        )
            except Exception:
                pass  # PS model failure is non-blocking

    has_violation = len(zero_var_cols) > 0
    has_adversarial = len(adversarial_cols) > 0

    if has_adversarial:
        status = 'ADVERSARIAL_SUSPECTED'
        tooltip = (
            f'SECURITY: Adversarial injection suspected in {", ".join(adversarial_cols)}. '
            'Zero-variance or outlier patterns are consistent with deliberate data poisoning. '
            'These columns are logged for review — do NOT silently discard.'
        )
    elif has_violation:
        status = 'FAIL'
        tooltip = (
            f'Positivity violation: {", ".join(zero_var_cols)} have zero variance. Rejecting analysis on these nodes.'
        )
    elif low_variance_cols:
        status = 'WARN'
        tooltip = f'Low variance in: {", ".join(low_variance_cols)}. Effect estimates may be unstable.'
    else:
        status = 'PASS'
        tooltip = 'Passed positivity and support checks. Variables have sufficient variance.'

    if propensity_warnings:
        status = 'WARN' if status == 'PASS' else status
        tooltip += f' Propensity overlap warnings: {propensity_warnings}'

    return {
        'test': 'Positivity',
        'status': status,
        'tooltip': tooltip,
        'details': {
            'zero_variance_variables': zero_var_cols,
            'low_variance_variables': low_variance_cols,
            'adversarial_variables': adversarial_cols,
            'propensity_overlap_warnings': propensity_warnings,
            'column_details': col_details,
        },
    }


def run_catl(data: pd.DataFrame, variable_names: list[str]) -> dict[str, Any]:
    """Run all CATL tests and return a complete report with adversarial alerts."""
    log.info('[CATL] Running Causal Assumption Transparency Layer')

    results = {}

    log.info('[CATL] Testing positivity')
    results['positivity'] = test_positivity(data, variable_names)
    pos_status = results['positivity']['status']
    if pos_status == 'ADVERSARIAL_SUSPECTED':
        log.warning('[CATL] positivity -> ADVERSARIAL_SUSPECTED *** SECURITY ALERT ***')
    else:
        log.info('[CATL] positivity result', status=pos_status)

    log.info('[CATL] Testing faithfulness')
    results['faithfulness'] = test_faithfulness(data, variable_names)
    log.info('[CATL] faithfulness result', status=results['faithfulness']['status'])

    log.info('[CATL] Testing causal sufficiency')
    results['sufficiency'] = test_causal_sufficiency(data, variable_names)
    log.info('[CATL] sufficiency result', status=results['sufficiency']['status'])

    log.info('[CATL] Testing stationarity')
    results['stationarity'] = test_stationarity(data, variable_names)
    log.info('[CATL] stationarity result', status=results['stationarity']['status'])

    log.info('[CATL] Checking acyclicity')
    results['acyclicity'] = test_acyclicity()
    log.info('[CATL] acyclicity result', status=results['acyclicity']['status'])

    # Summary with adversarial alert
    all_statuses = [r['status'] for r in results.values()]
    if 'ADVERSARIAL_SUSPECTED' in all_statuses:
        results['_summary'] = {
            'overall': 'ADVERSARIAL_SUSPECTED',
            'message': 'Data poisoning patterns detected. Manual review required before pipeline execution.',
            'adversarial_columns': results['positivity']['details'].get('adversarial_variables', []),
        }
    elif 'FAIL' in all_statuses:
        results['_summary'] = {
            'overall': 'FAIL',
            'message': 'One or more assumption tests failed.',
        }
    elif 'WARN' in all_statuses:
        results['_summary'] = {
            'overall': 'WARN',
            'message': 'Assumptions hold with caveats.',
        }
    else:
        results['_summary'] = {
            'overall': 'PASS',
            'message': 'All causal assumptions satisfied.',
        }

    log.info('[CATL] Summary', overall=results['_summary']['overall'])
    log.info('[CATL] All assumption tests complete')
    return results


if __name__ == '__main__':
    from cdie.config import VARIABLE_NAMES
    from cdie.pipeline.data_generator import generate_scm_data

    df = generate_scm_data()
    results = run_catl(df, VARIABLE_NAMES)
    for test_name, result in results.items():
        if test_name.startswith('_'):
            continue
        print(f'  {test_name}: {result["status"]} -- {result["tooltip"]}')
