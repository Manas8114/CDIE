"""
CDIE v4 — Intent Parser
Rule-based query classifier with OPEA LLM entity extraction.
Classifies queries into: intervention, counterfactual, root_cause, temporal.
"""

import re
import os
import json
import requests  # type: ignore[import-untyped]
from cdie.pipeline.data_generator import VARIABLE_NAMES

# Simplified variable names for matching
VARIABLE_ALIASES = {
    # Telecom mapping
    "sim box": "SIMBoxFraudAttempts",
    "sim box fraud": "SIMBoxFraudAttempts",
    "cdr": "CallDataRecordVolume",
    "call data": "CallDataRecordVolume",
    "arpu": "ARPUImpact",
    "loss": "ARPUImpact",
    # Generic & FinServ mapping
    "transaction volume": "TransactionVolume",
    "transaction amount": "TransactionVolume",
    "transection amount": "TransactionVolume",
    "amount": "TransactionVolume",
    "fraud attempts": "SIMBoxFraudAttempts",
    "fraud prob": "SIMBoxFraudAttempts",
    "fraud": "SIMBoxFraudAttempts",
    "detection policy": "FraudPolicyStrictness",
    "policy strictness": "FraudPolicyStrictness",
    "strictness": "FraudPolicyStrictness",
    "detection rate": "SIMFraudDetectionRate",
    "fraud detection": "SIMFraudDetectionRate",
    "chargeback volume": "RevenueLeakageVolume",
    "chargebacks": "RevenueLeakageVolume",
    "chargeback": "RevenueLeakageVolume",
    "customer trust score": "SubscriberRetentionScore",
    "trust score": "SubscriberRetentionScore",
    "trust": "SubscriberRetentionScore",
    "account age": "SubscriberRetentionScore",
    "revenue impact": "RevenueLeakageVolume",
    "revenue leakage": "RevenueLeakageVolume",
    "leakage": "RevenueLeakageVolume",
    "revenue": "RevenueLeakageVolume",
    "operational cost": "OperationalCost",
    "cost": "OperationalCost",
    "opex": "OperationalCost",
    "liquidity risk": "LiquidityRisk",
    "liquidity": "LiquidityRisk",
    "system load": "SystemLoad",
    "network load": "SystemLoad",
    "load": "SystemLoad",
    "external news signal": "ExternalNewsSignal",
    "news signal": "ExternalNewsSignal",
    "news": "ExternalNewsSignal",
    "regulatory pressure": "RegulatoryPressure",
    "regulation": "RegulatoryPressure",
    "regulatory": "RegulatoryPressure",
    "itu": "RegulatoryPressure",
}

# Add exact variable names to aliases
for v in VARIABLE_NAMES:
    VARIABLE_ALIASES[v.lower()] = v


INTERVENTION_PATTERNS = [
    r"what\s+(?:happens|if|would happen)\s+(?:if|when)",
    r"(?:increase|decrease|raise|lower|change|reduce|boost|double|triple)\s+\w+",
    r"(?:impact|effect)\s+of\s+(?:increasing|decreasing|changing|reducing)",
    r"how\s+(?:does|would|will)\s+(?:increasing|decreasing|changing)",
    r"simulate",
    r"\d+%\s+(?:increase|decrease|change|rise|drop|reduction)",
]

COUNTERFACTUAL_PATTERNS = [
    r"what\s+would\s+have\s+happened",
    r"if\s+we\s+had\s+(?:not|never)",
    r"had\s+we\s+(?:not|instead)",
    r"counterfactual",
    r"what\s+if\s+.*\s+(?:had|were|was)\s+(?:not|different|lower|higher)",
]

ROOT_CAUSE_PATTERNS = [
    r"why\s+(?:did|does|is|has|was)",
    r"what\s+(?:causes|caused|drives|drove)",
    r"root\s+cause",
    r"reason\s+(?:for|behind|why)",
    r"explain\s+(?:the|why|how)",
    r"what\s+led\s+to",
]

TEMPORAL_PATTERNS = [
    r"when\s+does\s+\w+\s+affect",
    r"(?:time|temporal)\s+(?:lag|delay|effect)",
    r"how\s+(?:long|soon|quickly)",
    r"lagged?\s+effect",
    r"over\s+time",
    r"delayed?\s+impact",
]


def extract_entities_llm(query: str) -> dict:
    """Extract entities using OPEA LLM endpoint."""
    llm_url = os.environ.get("OPEA_LLM_ENDPOINT", "http://tgi-service:80")

    prompt = f"""
    You are a causal inference entity extractor.
    Extract the source (intervention), target (outcome), and magnitude (percentage change) from the following query.
    Available variables: {", ".join(VARIABLE_NAMES)}
    
    Query: "{query}"
    
    Return ONLY a JSON object with keys: "source", "target", "magnitude".
    If magnitude is not specified, use 20.0. If a variable is not in the list, return null.
    """

    try:
        response = requests.post(
            f"{llm_url}/generate",
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 128, "temperature": 0.01},
            },
            timeout=5,
        )
        if response.status_code == 200:
            text = response.json().get("generated_text", "")
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
    except Exception:
        pass
    return {}


def extract_variables(query: str) -> tuple:
    """
    Extract source and target variables from a query string
    using alias matching and LLM-based DAG traversal.
    """
    llm_url = os.environ.get("OPEA_LLM_ENDPOINT", "http://tgi-service:80")

    # Try LLM-based DAG traversal for missing target
    q_lower = query.lower()
    found_vars = []
    for alias, var_name in sorted(VARIABLE_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in q_lower:
            if var_name not in found_vars:
                found_vars.append(var_name)
            if len(found_vars) >= 2:
                break

    source = found_vars[0] if len(found_vars) > 0 else None
    target = found_vars[1] if len(found_vars) > 1 else None

    # If we have a source but no target, try LLM suggestion
    if source and not target:
        try:
            prompt = (
                f"Given these variables: {', '.join(VARIABLE_NAMES)}, "
                f"what is the most logical downstream outcome of '{source}' "
                f"in a telecom fraud SCM? Return only the variable name."
            )
            msg = {"inputs": prompt, "parameters": {"max_new_tokens": 32}}
            res = requests.post(f"{llm_url}/generate", json=msg, timeout=3)
            if res.status_code == 200:
                suggested = (
                    res.json().get("generated_text", "").strip().split("\n")[0].strip()
                )
                if suggested in VARIABLE_NAMES:
                    target = suggested
        except Exception:
            pass

    # Fallback default target
    if source and not target:
        target = "ARPUImpact"

    return source, target


def extract_magnitude(query: str) -> float:
    """Extract magnitude percentage from query."""
    patterns = [
        r"(\d+)\s*%\s*(?:increase|rise|growth|up|higher|boost)",
        r"(?:increase|raise|boost|grow)\s+(?:by\s+)?(\d+)\s*%",
        r"(\d+)\s*%\s*(?:decrease|decline|drop|down|lower|reduction|reduce)",
        r"(?:decrease|reduce|lower|cut|drop)\s+(?:by\s+)?(\d+)\s*%",
        r"(\d+)\s*%",
    ]

    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            pct = float(match.group(1))
            if any(
                w in query.lower()
                for w in ["decrease", "reduce", "lower", "cut", "drop", "decline"]
            ):
                return -pct
            return pct

    return 20.0


def classify_query(query: str) -> dict:
    """
    Classify a user query into one of four types.
    Returns classification with extracted variables and magnitude.
    """
    q_lower = query.lower().strip()

    if not q_lower:
        return {"type": "error", "message": "Query cannot be empty."}

    scores = {
        "intervention": 0,
        "counterfactual": 0,
        "root_cause": 0,
        "temporal": 0,
    }

    for pattern in INTERVENTION_PATTERNS:
        if re.search(pattern, q_lower):
            scores["intervention"] += 1

    for pattern in COUNTERFACTUAL_PATTERNS:
        if re.search(pattern, q_lower):
            scores["counterfactual"] += 1

    for pattern in ROOT_CAUSE_PATTERNS:
        if re.search(pattern, q_lower):
            scores["root_cause"] += 1

    for pattern in TEMPORAL_PATTERNS:
        if re.search(pattern, q_lower):
            scores["temporal"] += 1

    best_type = max(scores, key=lambda k: scores[k])
    if scores[best_type] == 0:
        best_type = "intervention"

    # Try LLM extraction first, fall back to regex
    llm_entities = extract_entities_llm(query)

    if llm_entities:
        source = llm_entities.get("source")
        target = llm_entities.get("target")
        magnitude = llm_entities.get("magnitude", 20.0)
    else:
        source, target = extract_variables(query)
        magnitude = extract_magnitude(query)

    is_ambiguous = (
        scores[best_type] <= 1 and sum(1 for s in scores.values() if s > 0) > 1
    )

    return {
        "type": best_type,
        "source": source,
        "target": target,
        "magnitude": magnitude,
        "confidence": "high"
        if scores[best_type] >= 2
        else "medium"
        if scores[best_type] == 1
        else "low",
        "ambiguous": is_ambiguous,
        "all_scores": scores,
    }


# Pre-defined query presets for demo — Telecom Domain
DEMO_QUERIES = {
    "What happens if fraud attempts increase by 30%?": {
        "source": "SIMBoxFraudAttempts",
        "target": "RevenueLeakageVolume",
        "value": 0.3,
    },
    "What if we increase detection policy strictness by 20%?": {
        "source": "FraudPolicyStrictness",
        "target": "SIMFraudDetectionRate",
        "value": 0.2,
    },
    "Why did chargeback volume increase?": {
        "source": "SIMBoxFraudAttempts",
        "target": "RevenueLeakageVolume",
        "value": 0.1,
    },
    "Show me the temporal impact of fraud on trust.": {
        "source": "SIMBoxFraudAttempts",
        "target": "SubscriberRetentionScore",
        "value": -0.05,
    },
}
