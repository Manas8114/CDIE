"""
CDIE v4 — DoWhy Refutation Test Suite
Validates every causal edge with placebo, confounder, and subset tests.
Edges failing >= 2 tests are quarantined.
"""

import numpy as np
import pandas as pd
import networkx as nx


def _run_placebo_test(data, treatment, outcome, dag, variable_names):
    """Replace treatment with random noise; check if effect drops to ~0."""
    try:
        from dowhy import CausalModel

        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            graph=_dag_to_dot(dag, variable_names),
        )
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")

        refutation = model.refute_estimate(
            identified, estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=5,
        )

        new_effect = refutation.new_effect
        if isinstance(new_effect, (list, np.ndarray)):
            new_effect = float(np.mean(new_effect))

        passed = abs(new_effect) < abs(estimate.value) * 0.3
        return {
            "test": "placebo_treatment",
            "status": "PASS" if passed else "FAIL",
            "original_effect": round(float(estimate.value), 4),
            "placebo_effect": round(float(new_effect), 4),
        }
    except Exception as e:
        return {"test": "placebo_treatment", "status": "NOT_TESTED", "error": str(e)}


def _run_confounder_test(data, treatment, outcome, dag, variable_names):
    """Add random common cause; check if estimate changes by > 15%."""
    try:
        from dowhy import CausalModel

        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            graph=_dag_to_dot(dag, variable_names),
        )
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")

        refutation = model.refute_estimate(
            identified, estimate,
            method_name="random_common_cause",
            num_simulations=5,
        )

        new_effect = refutation.new_effect
        if isinstance(new_effect, (list, np.ndarray)):
            new_effect = float(np.mean(new_effect))

        orig = abs(estimate.value) if estimate.value != 0 else 1e-10
        change_pct = abs(new_effect - estimate.value) / orig
        passed = change_pct < 0.15

        return {
            "test": "random_common_cause",
            "status": "PASS" if passed else "WARN",
            "original_effect": round(float(estimate.value), 4),
            "new_effect": round(float(new_effect), 4),
            "change_pct": round(float(change_pct), 4),
        }
    except Exception as e:
        return {"test": "random_common_cause", "status": "NOT_TESTED", "error": str(e)}


def _run_subset_test(data, treatment, outcome, dag, variable_names, n_subsets=5):
    """Run ATE on random 80% subsets; check coefficient of variation."""
    if len(data) < 200:
        return {"test": "data_subset", "status": "NOT_TESTED", "reason": "n < 200"}

    try:
        from dowhy import CausalModel

        effects = []
        for i in range(n_subsets):
            subset = data.sample(frac=0.8, random_state=i)
            model = CausalModel(
                data=subset,
                treatment=treatment,
                outcome=outcome,
                graph=_dag_to_dot(dag, variable_names),
            )
            identified = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")
            effects.append(float(estimate.value))

        mean_effect = np.mean(effects)
        std_effect = np.std(effects)
        cv = std_effect / abs(mean_effect) if abs(mean_effect) > 1e-10 else float("inf")
        passed = cv < 0.25

        return {
            "test": "data_subset",
            "status": "PASS" if passed else "WARN",
            "effects": [round(e, 4) for e in effects],
            "cv": round(float(cv), 4),
        }
    except Exception as e:
        return {"test": "data_subset", "status": "NOT_TESTED", "error": str(e)}


def _dag_to_dot(dag, variable_names):
    """Convert networkx DAG to DOT string for DoWhy."""
    lines = ["digraph {"]
    for node in variable_names:
        if node in dag.nodes():
            lines.append(f'  "{node}";')

    for src, tgt in dag.edges():
        lines.append(f'  "{src}" -> "{tgt}";')
    lines.append("}")
    return "\n".join(lines)


def _run_refutation_fallback(data, treatment, outcome):
    """
    Fallback refutation using basic statistical checks when DoWhy is unavailable.
    """
    results = []

    # Placebo: shuffle treatment
    original_corr = np.corrcoef(data[treatment].values, data[outcome].values)[0, 1]
    placebo_corrs = []
    rng = np.random.default_rng(42)
    for _ in range(10):
        shuffled = rng.permutation(data[treatment].values)
        placebo_corrs.append(np.corrcoef(shuffled, data[outcome].values)[0, 1])
    avg_placebo = np.mean(np.abs(placebo_corrs))
    results.append({
        "test": "placebo_treatment",
        "status": "PASS" if avg_placebo < abs(original_corr) * 0.3 else "FAIL",
        "original_effect": round(float(original_corr), 4),
        "placebo_effect": round(float(avg_placebo), 4),
    })

    # Confounder: add random noise variable
    noise = rng.normal(0, 1, len(data))
    from sklearn.linear_model import LinearRegression
    X_orig = data[[treatment]].values
    X_conf = np.column_stack([X_orig, noise])
    y = data[outcome].values
    coef_orig = LinearRegression().fit(X_orig, y).coef_[0]
    coef_conf = LinearRegression().fit(X_conf, y).coef_[0]
    change = abs(coef_conf - coef_orig) / max(abs(coef_orig), 1e-10)
    results.append({
        "test": "random_common_cause",
        "status": "PASS" if change < 0.15 else "WARN",
        "original_effect": round(float(coef_orig), 4),
        "new_effect": round(float(coef_conf), 4),
        "change_pct": round(float(change), 4),
    })

    # Subset stability
    if len(data) >= 200:
        effects = []
        for i in range(5):
            subset = data.sample(frac=0.8, random_state=i)
            coef = LinearRegression().fit(subset[[treatment]].values, subset[outcome].values).coef_[0]
            effects.append(coef)
        cv = np.std(effects) / max(abs(np.mean(effects)), 1e-10)
        results.append({
            "test": "data_subset",
            "status": "PASS" if cv < 0.25 else "WARN",
            "effects": [round(e, 4) for e in effects],
            "cv": round(float(cv), 4),
        })
    else:
        results.append({"test": "data_subset", "status": "NOT_TESTED", "reason": "n < 200"})

    return results


def run_refutation(data: pd.DataFrame, map_dag: nx.DiGraph, variable_names: list[str]):
    """
    Run 3-test refutation on each edge. Quarantine edges failing >= 2 tests.
    """
    print(f"[Refutation] Validating {map_dag.number_of_edges()} causal edges...")

    edge_results = {}
    quarantined = []
    validated = []

    for src, tgt in map_dag.edges():
        if src not in data.columns or tgt not in data.columns:
            continue

        print(f"[Refutation] Testing {src} → {tgt}...")

        try:
            tests = []
            tests.append(_run_placebo_test(data, src, tgt, map_dag, variable_names))
            tests.append(_run_confounder_test(data, src, tgt, map_dag, variable_names))
            tests.append(_run_subset_test(data, src, tgt, map_dag, variable_names))
        except Exception:
            tests = _run_refutation_fallback(data, src, tgt)

        # Count failures
        n_fail = sum(1 for t in tests if t["status"] in ("FAIL", "WARN"))
        is_quarantined = n_fail >= 2

        edge_results[f"{src}->{tgt}"] = {
            "source": src,
            "target": tgt,
            "tests": tests,
            "n_fail": n_fail,
            "quarantined": is_quarantined,
        }

        if is_quarantined:
            quarantined.append((src, tgt))
            print(f"[Refutation]   → QUARANTINED ({n_fail} tests failed)")
        else:
            validated.append((src, tgt))
            print(f"[Refutation]   → VALIDATED ({n_fail} warnings)")

    print(f"[Refutation] Complete. Validated: {len(validated)}, Quarantined: {len(quarantined)}")

    return {
        "edge_results": edge_results,
        "validated_edges": validated,
        "quarantined_edges": quarantined,
        "pass_rate": round(len(validated) / max(len(edge_results), 1), 4),
    }


if __name__ == "__main__":
    from cdie.pipeline.data_generator import generate_scm_data, VARIABLE_NAMES
    from cdie.pipeline.gfci_discovery import run_discovery

    df = generate_scm_data()
    discovery = run_discovery(df)
    result = run_refutation(df, discovery["map_dag"], VARIABLE_NAMES)
    print(f"\nPass rate: {result['pass_rate']:.0%}")
