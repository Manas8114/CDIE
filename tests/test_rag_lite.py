import pytest
from unittest.mock import MagicMock, patch

def test_sufficiency_gate_logic_lite():
    from cdie.api.rag.llm import check_sufficiency
    
    # All criteria met -> True
    effect_strong = {'point_estimate': 1.0, 'ci_lower': 0.9, 'ci_upper': 1.1}
    refutation_pass = {'placebo': 'PASS', 'confounder': 'PASS', 'subset': 'PASS'}
    analogies = [{'confidence': 'High', 'text': 'High confidence event'}]
    assert check_sufficiency(effect_strong, refutation_pass, analogies) is True

def test_template_explanation_generation_lite():
    from cdie.api.rag.llm import build_template_explanation
    effect = {'point_estimate': 0.15, 'ci_lower': 0.1, 'ci_upper': 0.2}
    explanation = build_template_explanation(
        "intervention", "SrcVar", "TgtVar", effect, None, None
    )
    assert "SrcVar" in explanation
    assert "0.15" in explanation
