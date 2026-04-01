"""
CDIE v4 — Prior Extractor Verification Script
Tests the PriorExtractor against synthetic_prior_test.txt
"""
import sys
import json
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from cdie.pipeline.prior_extractor import PriorExtractor
from cdie.pipeline.data_generator import VARIABLE_NAMES

DATA_DIR = Path(__file__).parent / "data"

EXPECTED_EDGES = [
    ("CallDataRecordVolume", "NetworkLoad"),
    ("SIMBoxFraudAttempts", "RevenueLeakageVolume"),
    ("FraudPolicyStrictness", "SIMFraudDetectionRate"),
    ("RevenueLeakageVolume", "CashFlowRisk"),
    ("SIMFraudDetectionRate", "ARPUImpact"),
    ("RegulatorySignal", "ITURegulatoryPressure"),
    ("NetworkLoad", "NetworkOpExCost"),
    ("ARPUImpact", "SubscriberRetentionScore"),
]

@pytest.fixture
def text():
    test_file = DATA_DIR / "synthetic_prior_test.txt"
    assert test_file.exists(), f"Test file not found: {test_file}"
    text_content = test_file.read_text(encoding="utf-8")
    assert len(text_content) > 100, "Test file too short"
    assert "CallDataRecordVolume" in text_content, "Expected variable not found in text"
    return text_content

@pytest.fixture
def extractor():
    return PriorExtractor()

def test_text_extraction(text):
    """Test that text can be read from the synthetic file."""
    assert text is not None

def test_extractor_init(extractor):
    """Test PriorExtractor initializes without crash."""
    assert extractor is not None

def test_extraction_with_llm(extractor, text):
    """Test the full LLM extraction (requires OPEA or OpenAI endpoint)."""
    if not extractor.client:
        print("[SKIP] No LLM client configured — skipping live extraction test")
        return

    priors = extractor.extract_from_text(text)
    
    for p in priors:
        assert "source" in p and "target" in p and "confidence" in p, f"Malformed prior: {p}"
        assert p["source"] in VARIABLE_NAMES, f"Invalid source: {p['source']}"
        assert p["target"] in VARIABLE_NAMES, f"Invalid target: {p['target']}"
        assert 0.0 <= p["confidence"] <= 1.0, f"Confidence out of range: {p['confidence']}"

    extracted_pairs = {(p["source"], p["target"]) for p in priors}
    hits = sum(1 for e in EXPECTED_EDGES if e in extracted_pairs)
    print(f"Extraction hits: {hits}/{len(EXPECTED_EDGES)}")
    output_path = DATA_DIR / "extracted_priors.json"
    with open(output_path, "w") as f:
        json.dump(priors, f, indent=2)

def test_pipeline_integration():
    """Verify the pipeline can load extracted_priors.json."""
    priors_path = DATA_DIR / "extracted_priors.json"
    if not priors_path.exists():
        print("[SKIP] No extracted_priors.json found — skipping pipeline integration")
        return

    with open(priors_path) as f:
        priors = json.load(f)

    assert isinstance(priors, list), "Priors file should contain a JSON array"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
