"""
CDIE v5 — Backtesting Engine
Replays historical data through the causal estimator to answer:
  "If we had done intervention X at time T, what would have happened?"
Compares predicted counterfactual with actual observed outcome.
"""

from typing import Any

import pandas as pd

from cdie.pipeline.data_generator import VARIABLE_NAMES
from cdie.pipeline.estimation import compute_ate_dml


class Backtester:
    """Counterfactual replay engine for causal intervention backtesting."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.n_rows = len(data)

    def backtest(
        self,
        source: str,
        target: str,
        magnitude: float,
        start_index: int = 0,
        end_index: int | None = None,
    ) -> dict[str, Any]:
        """
        Backtest a counterfactual intervention.

        Args:
            source: Treatment variable
            target: Outcome variable
            magnitude: Intervention magnitude (e.g., 0.20 for 20% increase)
            start_index: Start row index for the historical window
            end_index: End row index (defaults to midpoint)
        """
        if source not in self.data.columns or target not in self.data.columns:
            return {'error': f'Variables not found: {source}, {target}'}

        if end_index is None:
            end_index = self.n_rows // 2

        start_index = max(0, start_index)
        end_index = min(self.n_rows, end_index)

        if end_index - start_index < 30:
            return {'error': 'Insufficient data in window (need >= 30 rows)'}

        # Split data: training window and evaluation window
        train_data = self.data.iloc[start_index:end_index]
        eval_data = self.data.iloc[end_index:]

        if len(eval_data) < 10:
            eval_data = self.data.iloc[max(0, end_index - 50) : end_index]

        warnings: list[str] = []

        # Step 1: Estimate ATE from training window
        confounders = [
            c
            for c in VARIABLE_NAMES
            if c != source
            and c != target
            and c in train_data.columns
            and abs(train_data[c].corr(train_data[target])) > 0.1
        ][:5]

        ate_result = compute_ate_dml(train_data, source, target, confounders)
        ate = ate_result.get('ate', 0)

        # Step 2: Predict counterfactual outcome
        source_mean = float(train_data[source].mean())
        intervention_amount = source_mean * magnitude
        predicted_delta = ate * intervention_amount

        # Step 3: Compute actual observed change
        train_target_mean = float(train_data[target].mean())
        eval_target_mean = float(eval_data[target].mean())
        actual_delta = eval_target_mean - train_target_mean

        # Step 4: Accuracy scoring
        if actual_delta != 0:
            direction_match = (predicted_delta > 0) == (actual_delta > 0)
            magnitude_ratio = min(abs(predicted_delta), abs(actual_delta)) / max(
                abs(predicted_delta), abs(actual_delta)
            )
            accuracy_score = magnitude_ratio * (1.0 if direction_match else 0.3)
        elif predicted_delta == 0:
            accuracy_score = 1.0
            direction_match = True
            magnitude_ratio = 1.0
        else:
            accuracy_score = 0.0
            direction_match = False
            magnitude_ratio = 0.0

        # Step 5: Detect cases where model would have been wrong
        if not direction_match:
            warnings.append(
                f'DIRECTION_MISMATCH: Model predicted {"increase" if predicted_delta > 0 else "decrease"} '
                f'but actual was {"increase" if actual_delta > 0 else "decrease"}. '
                'Possible confounding or seasonal pattern not captured by GFCI.'
            )

        if accuracy_score < 0.3:
            warnings.append(
                'LOW_ACCURACY: Prediction accuracy below 30%. '
                'This intervention-outcome pair may have non-stationary dynamics. '
                'Consider re-running pipeline with more recent data.'
            )

        if abs(predicted_delta) > 3 * abs(actual_delta) and actual_delta != 0:
            warnings.append(
                f'OVERESTIMATION: Model predicted {abs(predicted_delta):.4f} but actual was {abs(actual_delta):.4f}. '
                'ATE may be inflated by confounders.'
            )

        return {
            'source': source,
            'target': target,
            'magnitude': magnitude,
            'window': {
                'start': start_index,
                'end': end_index,
                'n_train': len(train_data),
                'n_eval': len(eval_data),
            },
            'ate_estimate': {
                'ate': round(ate, 4),
                'method': ate_result.get('method', 'unknown'),
                'ci_lower': ate_result.get('ci_lower', 0),
                'ci_upper': ate_result.get('ci_upper', 0),
            },
            'predicted_delta': round(float(predicted_delta), 4),
            'actual_delta': round(float(actual_delta), 4),
            'accuracy': {
                'score': round(float(accuracy_score), 3),
                'direction_match': direction_match,
                'magnitude_ratio': round(float(magnitude_ratio), 3),
                'label': 'HIGH' if accuracy_score > 0.7 else 'MEDIUM' if accuracy_score > 0.4 else 'LOW',
            },
            'warnings': warnings,
        }

    def batch_backtest(
        self, source: str, targets: list[str] | None = None, magnitude: float = 0.2
    ) -> list[dict[str, Any]]:
        """Run backtest across multiple targets for one source intervention."""
        if targets is None:
            targets = [v for v in VARIABLE_NAMES if v != source and v in self.data.columns]

        results = []
        for target in targets:
            result = self.backtest(source, target, magnitude)
            if 'error' not in result:
                results.append(result)

        results.sort(key=lambda r: r['accuracy']['score'], reverse=True)
        return results
