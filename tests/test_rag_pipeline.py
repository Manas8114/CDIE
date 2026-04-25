import pytest
import os
from unittest.mock import MagicMock, patch
from cdie.api.rag.engine import ExplanationEngine

@pytest.fixture
def mock_redis():
    with patch('cdie.api.rag.cache.build_redis_client') as mock:
        client = MagicMock()
        mock.return_value = client
        yield client

@pytest.fixture
def mock_openai():
    with patch('openai.OpenAI') as mock:
        client = MagicMock()
        mock.return_value = client
        yield client

def test_engine_init_fallback(mock_redis):
    """Test that the engine falls back to TF-IDF when no OPEA endpoints are provided."""
    with patch.dict('os.environ', {}, clear=True):
        engine = ExplanationEngine()
        assert engine.embedding_provider == 'tfidf'
        assert engine.llm_provider == 'template'

def test_retrieve_analogies_tfidf(mock_redis):
    """Test TF-IDF based analogy retrieval."""
    with patch.dict('os.environ', {}, clear=True):
        engine = ExplanationEngine()
        # Mocking the event texts for predictable testing
        engine.event_texts = ["fraud in roaming", "sim box detection", "high revenue leakage"]
        
        analogies = engine.retrieve_analogies("fraud attempts", top_k=2)
        assert isinstance(analogies, list)
        assert len(analogies) <= 2
        # Check that it returns dictionaries with expected keys from events
        if analogies:
            assert 'text' in analogies[0]

def test_generate_explanation_template(mock_redis):
    """Test that the engine generates a valid template-based explanation."""
    with patch.dict('os.environ', {}, clear=True):
        engine = ExplanationEngine()
        effect = {'point_estimate': 0.15, 'ci_lower': 0.1, 'ci_upper': 0.2}
        explanation = engine.generate_explanation(
            query_type="intervention",
            source="SIMBoxFraudAttempts",
            target="RevenueLeakageVolume",
            effect=effect
        )
        assert isinstance(explanation, str)
        assert "SIMBoxFraudAttempts" in explanation
        assert "RevenueLeakageVolume" in explanation
        assert "0.15" in explanation

@patch('cdie.api.rag.engine.generate_llm_explanation')
def test_generate_explanation_llm_path(mock_gen_llm, mock_redis, mock_openai):
    """Test the orchestration path that calls the LLM."""
    # Set mock LLM return
    mock_gen_llm.return_value = "LLM Generated Explanation"
    
    with patch.dict('os.environ', {'OPEA_LLM_ENDPOINT': 'http://localhost:9000'}, clear=True):
        engine = ExplanationEngine()
        assert engine.llm_provider == 'openai_compat'
        
        effect = {'point_estimate': 0.15, 'ci_lower': 0.1, 'ci_upper': 0.2}
        # We need to make sure check_sufficiency returns False to trigger LLM
        with patch('cdie.api.rag.engine.check_sufficiency', return_value=False):
            explanation = engine.generate_explanation(
                query_type="intervention",
                source="SIMBoxFraudAttempts",
                target="RevenueLeakageVolume",
                effect=effect
            )
            assert explanation == "LLM Generated Explanation"
            mock_gen_llm.assert_called_once()

def test_explanation_caching(mock_redis):
    """Test that explanations are correctly retrieved from the Redis cache."""
    # Mock cache hit
    mock_redis.get.return_value = b"Cached Explanation"
    
    with patch.dict('os.environ', {}, clear=True):
        engine = ExplanationEngine()
        effect = {'point_estimate': 0.15}
        explanation = engine.generate_explanation("type", "src", "tgt", effect)
        
        assert explanation == "Cached Explanation"
        # Verify cache was checked
        assert mock_redis.get.called

def test_sufficiency_gate_logic():
    """Test the check_sufficiency logic directly."""
    from cdie.api.rag.llm import check_sufficiency
    
    # 1. Weak effect -> False
    assert check_sufficiency({'point_estimate': 0.0000001}, None, []) is False
    
    # 2. Wide CI -> False
    effect_wide = {'point_estimate': 1.0, 'ci_lower': 0.1, 'ci_upper': 1.9}
    assert check_sufficiency(effect_wide, None, []) is False
    
    # 3. Strong effect, tight CI, but failing refutation -> False
    effect_strong = {'point_estimate': 1.0, 'ci_lower': 0.9, 'ci_upper': 1.1}
    refutation_fail = {'placebo': 'FAIL'}
    assert check_sufficiency(effect_strong, refutation_fail, []) is False
    
    # 4. Strong effect, tight CI, passing refutation, but no high-conf analogies -> False
    refutation_pass = {'placebo': 'PASS', 'confounder': 'PASS', 'subset': 'PASS'}
    assert check_sufficiency(effect_strong, refutation_pass, [{'confidence': 'Low'}]) is False
    
    # 5. All criteria met -> True
    analogies = [{'confidence': 'High', 'text': 'High confidence event'}]
    assert check_sufficiency(effect_strong, refutation_pass, analogies) is True

if __name__ == "__main__":
    print("Running manual tests...")
    # Mocking redis for manual run
    with patch('cdie.api.rag.cache.build_redis_client') as mock_redis_client:
        mock_redis_client.return_value = MagicMock()
        with patch.dict('os.environ', {}, clear=True):
            engine = ExplanationEngine()
            print("Engine initialized.")
            analogies = engine.retrieve_analogies("fraud", top_k=1)
            print(f"Retrieved {len(analogies)} analogies.")
            explanation = engine.generate_explanation("type", "src", "tgt", {'point_estimate': 0.1})
            print("Explanation generated.")
            print(explanation[:100] + "...")

