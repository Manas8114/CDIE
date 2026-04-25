import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdie.pipeline.estimation import discover_heterogeneity


@pytest.fixture
def interaction_dataset():
    """
    Generate a dataset where 'DeviceTier' significantly modulates the effect
    Generate a dataset where 'device_tier' significantly modulates the effect
    of 'fraud_policy_strictness' on 'y'.
    """
    np.random.seed(42)
    n = 1000

    # Features
    device_tier = np.random.randint(1, 4, n)  # 1, 2, or 3
    subscriber_tenure_months = np.random.randint(1, 120, n)
    regional_risk_score = np.random.uniform(0, 1, n)

    # Treatment
    fraud_policy_strictness = np.random.uniform(0, 1, n)

    # Outcome with HTE:
    # High-tier devices and long-tenure subscribers have different sensitivity
    base_effect = 0.5
    interact_term = (device_tier == 3).astype(float) * 0.2 + (subscriber_tenure_months > 60).astype(float) * 0.1
    effect = base_effect + interact_term

    y = effect * fraud_policy_strictness + 0.05 * subscriber_tenure_months + np.random.normal(0, 0.01, n)

    df = pd.DataFrame(
        {
            'device_tier': device_tier,
            'subscriber_tenure_months': subscriber_tenure_months,
            'regional_risk_score': regional_risk_score,
            'fraud_policy_strictness': fraud_policy_strictness,
            'y': y,
        }
    )

    return df


def test_discover_heterogeneity_keys(interaction_dataset):
    """Verify that the discovery function returns all required metadata keys."""
    results = discover_heterogeneity(
        data=interaction_dataset,
        treatment='fraud_policy_strictness',
        outcome='y',
        context_cols=['device_tier', 'subscriber_tenure_months', 'regional_risk_score'],
    )

    assert 'method' in results
    assert 'feature_importance' in results
    assert 'individual_effects' in results
    assert 'high_effect_profile' in results
    assert 'low_effect_profile' in results
    assert results['method'] in ['CausalForestDML', 'ForestDRLearner']


def test_interaction_recovery(interaction_dataset):
    """
    Verify that DeviceTier is identified as the most important feature
    since it was injected as the primary moderator.
    """
    results = discover_heterogeneity(
        data=interaction_dataset,
        treatment='fraud_policy_strictness',
        outcome='y',
        context_cols=['device_tier', 'subscriber_tenure_months', 'regional_risk_score'],
    )

    importances = results['feature_importance']
    # device_tier should be at or near the top
    top_feature = max(importances, key=importances.get)
    assert top_feature == 'device_tier'
    assert importances['device_tier'] > importances['regional_risk_score']


def test_fallback_logic_small_data():
    """Verify that the function handles low-sample data gracefully."""
    df_small = pd.DataFrame(
        {
            'device_tier': [1, 2],
            'subscriber_tenure_months': [10, 20],
            'fraud_policy_strictness': [0.1, 0.2],
            'y': [0.5, 0.6],
        }
    )

    results = discover_heterogeneity(df_small, 'fraud_policy_strictness', 'y')

    assert results['method'] == 'unavailable'
    assert 'insufficient samples' in results['reason']


if __name__ == '__main__':
    pytest.main([__file__])
