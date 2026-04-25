import pytest
import pandas as pd
import numpy as np
import networkx as nx
from cdie.pipeline.estimation import add_mapie_intervals
from cdie.pipeline.gfci_discovery import _run_pc_fallback
from cdie.pipeline.catl import test_positivity
from cdie.pipeline.data_generator import generate_scm_data
from cdie.pipeline.run_pipeline import run_pipeline

def test_add_mapie_intervals_logic():
    """Test that MAPIE intervals are correctly calculated and labeled."""
    df = pd.DataFrame({
        'X': np.linspace(0, 10, 100),
        'Y': np.linspace(0, 10, 100) + np.random.normal(0, 0.1, 100)
    })
    ate_result = {'ate': 1.0}
    
    # Mocking mapie as it might be slow or missing in some environments
    # But here we want to test the actual integration if possible
    try:
        result = add_mapie_intervals(df, 'X', 'Y', ate_result)
        assert 'mapie_point' in result
        assert 'mapie_lower' in result
        assert 'mapie_upper' in result
        assert 'confidence_label' in result
        assert result['mapie_method'] == 'conformal_plus'
    except ImportError:
        pytest.skip("MAPIE not installed")

def test_pc_fallback_execution():
    """Test that PC algorithm fallback executes without error."""
    df = pd.DataFrame(np.random.normal(0, 1, (100, 3)), columns=['A', 'B', 'C'])
    variable_names = ['A', 'B', 'C']
    
    cg, method = _run_pc_fallback(df.values, variable_names)
    assert method == 'PC'
    assert cg is not None
    # Check if it has the expected graph structure from causal-learn
    assert hasattr(cg, 'G')

def test_positivity_detection():
    """Test positivity check for zero variance and overlap violations."""
    # 1. Zero variance case
    df_zero = pd.DataFrame({
        'A': [1] * 100,
        'B': np.random.normal(0, 1, 100)
    })
    res_zero = test_positivity(df_zero, ['A', 'B'])
    assert res_zero['status'] in ('FAIL', 'ADVERSARIAL_SUSPECTED')
    assert 'A' in res_zero['details']['zero_variance_variables']

    # 2. Perfect prediction (Overlap violation)
    df_overlap = pd.DataFrame({
        'T': [0] * 50 + [1] * 50,
        'X': [0] * 50 + [1] * 50, # X perfectly predicts T
        'Y': np.random.normal(0, 1, 100)
    })
    res_overlap = test_positivity(df_overlap, ['T', 'X', 'Y'])
    # Should have a warning about propensity overlap
    assert any('overlap violation' in w for w in res_overlap['details']['propensity_overlap_warnings'])
    assert res_overlap['status'] == 'WARN'

@pytest.mark.slow
def test_full_pipeline_e2e_lite():
    """End-to-end test of the pipeline with a small dataset."""
    # We use a very small N to keep it fast
    with pytest.MonkeyPatch.context() as m:
        # Override data size for speed
        import cdie.pipeline.data_generator
        m.setattr(cdie.pipeline.data_generator, 'SCM_ROWS', 200)
        
        # Run pipeline in a way that doesn't save to production data/
        # (Though run_pipeline usually saves to data/, we trust the environment is clean)
        try:
            # We use a temporary runtime dir to avoid polluting
            m.setenv("CDIE_RUNTIME_DIR", "./test_runtime")
            
            # Execute pipeline
            # Note: this might still be slow due to GFCI
            # We skip GFCI timeout by setting a very short timeout if possible
            results = run_pipeline(timeout_seconds=5)
            
            assert 'catl' in results
            assert 'discovery' in results
            assert 'estimation' in results
            assert 'safety_map' in results
            
            # Check if safety map was generated
            assert results['safety_map']['n_scenarios'] > 0
        except Exception as e:
            pytest.fail(f"Pipeline E2E failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
