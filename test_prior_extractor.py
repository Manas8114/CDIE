"""
CDIE v4 — Prior Extractor Verification Script
Tests the PriorExtractor against synthetic_prior_test.txt
"""
import sys
import json
from pathlib import Path

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


def test_text_extraction():
    """Test that text can be read from the synthetic file."""
    test_file = DATA_DIR / "synthetic_prior_test.txt"
    assert test_file.exists(), f"Test file not found: {test_file}"

    text = test_file.read_text(encoding="utf-8")
    assert len(text) > 100, "Test file too short"
    assert "CallDataRecordVolume" in text, "Expected variable not found in text"
    print(f"[PASS] Text extraction: {len(text)} chars loaded")
    return text


def test_extractor_init():
    """Test PriorExtractor initializes without crash."""
    extractor = PriorExtractor()
    print(f"[PASS] PriorExtractor initialized (LLM client: {'connected' if extractor.client else 'offline'})")
    return extractor


def test_extraction_with_llm(extractor, text):
    """Test the full LLM extraction (requires OPEA or OpenAI endpoint)."""
    if not extractor.client:
        print("[SKIP] No LLM client configured — skipping live extraction test")
        print("       Set OPEA_LLM_ENDPOINT or OPENAI_API_KEY to enable")
        return None

    priors = extractor.extract_from_text(text)
    print(f"[INFO] Extracted {len(priors)} priors from LLM")

    # Validate structure
    for p in priors:
        assert "source" in p and "target" in p and "confidence" in p, f"Malformed prior: {p}"
        assert p["source"] in VARIABLE_NAMES, f"Invalid source: {p['source']}"
        assert p["target"] in VARIABLE_NAMES, f"Invalid target: {p['target']}"
        assert 0.0 <= p["confidence"] <= 1.0, f"Confidence out of range: {p['confidence']}"

    print(f"[PASS] All {len(priors)} priors have valid structure")

    # Check coverage of expected edges
    extracted_pairs = {(p["source"], p["target"]) for p in priors}
    hits = sum(1 for e in EXPECTED_EDGES if e in extracted_pairs)
    print(f"[INFO] Coverage: {hits}/{len(EXPECTED_EDGES)} expected edges found")

    # Save for pipeline consumption
    output_path = DATA_DIR / "extracted_priors.json"
    with open(output_path, "w") as f:
        json.dump(priors, f, indent=2)
    print(f"[PASS] Priors saved to {output_path}")

    return priors


def test_pipeline_integration():
    """Verify the pipeline can load extracted_priors.json."""
    priors_path = DATA_DIR / "extracted_priors.json"
    if not priors_path.exists():
        print("[SKIP] No extracted_priors.json found — skipping pipeline integration")
        return

    with open(priors_path) as f:
        priors = json.load(f)

    assert isinstance(priors, list), "Priors file should contain a JSON array"
    high_conf = [p for p in priors if p.get("confidence", 0) > 0.70]
    print(f"[PASS] Pipeline integration: {len(high_conf)}/{len(priors)} priors above 0.70 threshold")


if __name__ == "__main__":
    print("=" * 60)
    print("  CDIE v4 — Prior Extractor Verification")
    print("=" * 60)

    text = test_text_extraction()
    extractor = test_extractor_init()
    priors = test_extraction_with_llm(extractor, text)
    test_pipeline_integration()

    print("\n" + "=" * 60)
    if priors is not None:
        print(f"  VERIFICATION COMPLETE: {len(priors)} priors extracted")
    else:
        print("  VERIFICATION COMPLETE: Structural tests passed (LLM offline)")
    print("=" * 60)
