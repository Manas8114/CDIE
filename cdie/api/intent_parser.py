"""
CDIE v4 — Intent Parser
Rule-based query classifier with OPEA LLM entity extraction.
Classifies queries into: intervention, counterfactual, root_cause, temporal.
"""

import re
import os
import json
import difflib
import requests  # type: ignore[import-untyped]
from cdie.pipeline.data_generator import VARIABLE_NAMES, GROUND_TRUTH_EDGES

# Simplified variable names for matching
VARIABLE_ALIASES = {
    # Telecom mapping
    "sim box": "SIMBoxFraudAttempts",
    "sim box fraud": "SIMBoxFraudAttempts",
    "sim fraud": "SIMBoxFraudAttempts",
    "cdr": "CallDataRecordVolume",
    "call data": "CallDataRecordVolume",
    "call records": "CallDataRecordVolume",
    "arpu": "ARPUImpact",
    "arpu impact": "ARPUImpact",
    "retention": "SubscriberRetentionScore",
    "subscriber retention": "SubscriberRetentionScore",
    "subscriber score": "SubscriberRetentionScore",
    "network opex": "NetworkOpExCost",
    "opex": "NetworkOpExCost",
    "operating cost": "NetworkOpExCost",
    "cash flow": "CashFlowRisk",
    "cashflow": "CashFlowRisk",
    "regulatory signal": "RegulatorySignal",
    "itu pressure": "ITURegulatoryPressure",
    "fraud attempts": "SIMBoxFraudAttempts",
    "fraud prob": "SIMBoxFraudAttempts",
    "fraud": "SIMBoxFraudAttempts",
    "fraud policy": "FraudPolicyStrictness",
    "detection policy": "FraudPolicyStrictness",
    "policy strictness": "FraudPolicyStrictness",
    "policy increase": "FraudPolicyStrictness",
    "increase policy": "FraudPolicyStrictness",
    "strictness": "FraudPolicyStrictness",
    "detection rate": "SIMFraudDetectionRate",
    "fraud detection": "SIMFraudDetectionRate",
    "detection": "SIMFraudDetectionRate",
    "revenue impact": "ARPUImpact",
    "revenue leakage": "RevenueLeakageVolume",
    "leakage": "RevenueLeakageVolume",
    "revenue": "RevenueLeakageVolume",
    "cost": "NetworkOpExCost",
    "network load": "NetworkLoad",
    "load": "NetworkLoad",
    "regulatory pressure": "ITURegulatoryPressure",
    "regulation": "RegulatorySignal",
    "regulatory": "RegulatorySignal",
    "itu": "ITURegulatoryPressure",
}

# Add exact variable names to aliases
for v in VARIABLE_NAMES:
    VARIABLE_ALIASES[v.lower()] = v

DEFAULT_TARGETS = {
    "CallDataRecordVolume": "NetworkLoad",
    "SIMBoxFraudAttempts": "RevenueLeakageVolume",
    "FraudPolicyStrictness": "SIMFraudDetectionRate",
    "SIMFraudDetectionRate": "RevenueLeakageVolume",
    "RevenueLeakageVolume": "ARPUImpact",
    "SubscriberRetentionScore": "ARPUImpact",
    "NetworkLoad": "NetworkOpExCost",
    "NetworkOpExCost": "CashFlowRisk",
    "CashFlowRisk": "ARPUImpact",
    "RegulatorySignal": "FraudPolicyStrictness",
    "ITURegulatoryPressure": "FraudPolicyStrictness",
}

VARIABLE_DESCRIPTIONS = {
    "CallDataRecordVolume": "Volume of call detail records processed across the network.",
    "SIMBoxFraudAttempts": "Observed or inferred SIM-box fraud attempt pressure.",
    "FraudPolicyStrictness": "How aggressively fraud controls and blocking policies are enforced.",
    "SIMFraudDetectionRate": "Rate at which SIM-box or fraud activity is detected.",
    "RevenueLeakageVolume": "Estimated revenue lost due to bypass or fraud leakage.",
    "SubscriberRetentionScore": "Retention health of subscribers after fraud-control actions.",
    "ARPUImpact": "Impact on average revenue per user.",
    "NetworkOpExCost": "Operational cost burden caused by network and fraud-control actions.",
    "CashFlowRisk": "Cash-flow exposure caused by fraud or cost escalation.",
    "NetworkLoad": "Load or pressure on the network infrastructure.",
    "RegulatorySignal": "External regulatory activity or warning signal.",
    "ITURegulatoryPressure": "ITU-related regulatory pressure affecting fraud policy decisions.",
}

DOWNSTREAM_GRAPH: dict[str, list[str]] = {}
for src, tgt in GROUND_TRUTH_EDGES:
    DOWNSTREAM_GRAPH.setdefault(src, []).append(tgt)


def _get_llm_url() -> str | None:
    llm_url = os.environ.get("OPEA_LLM_ENDPOINT")
    return llm_url if llm_url else None


def _aliases_for_variable(variable_name: str) -> list[str]:
    aliases = [
        alias
        for alias, resolved_name in VARIABLE_ALIASES.items()
        if resolved_name == variable_name and alias != variable_name.lower()
    ]
    aliases.sort(key=len)
    return aliases


def suggest_variables(query: str, limit: int = 5) -> list[str]:
    """Suggest the closest valid variables for a free-form query."""
    q_lower = query.lower().strip()
    if not q_lower:
        return VARIABLE_NAMES[:limit]

    candidates: list[tuple[float, str]] = []
    for variable_name in VARIABLE_NAMES:
        aliases = [variable_name.lower(), *_aliases_for_variable(variable_name)]
        best_score = 0.0
        for alias in aliases:
            score = difflib.SequenceMatcher(None, q_lower, alias).ratio()
            if alias in q_lower:
                score += 0.35
            if any(token and token in alias for token in q_lower.split()):
                score += 0.1
            best_score = max(best_score, score)
        candidates.append((best_score, variable_name))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [name for score, name in candidates if score > 0.2][:limit]


def build_query_suggestions(source: str | None = None, target: str | None = None) -> list[str]:
    """Return example questions tailored to the parsed variables."""
    if source and target and source != target:
        return [
            f"What happens if {source} increases by 20%?",
            f"What happens if {source} decreases by 20%?",
            f"How does {source} affect {target}?",
        ]

    if source:
        downstream = DEFAULT_TARGETS.get(source, "ARPUImpact")
        if downstream == source:
            return [
                f"What happens if {source} increases by 20%?",
                f"What happens if {source} decreases by 20%?",
                f"Why did {source} change?",
            ]
        return [
            f"What happens if {source} increases by 20%?",
            f"How does {source} affect {downstream}?",
            f"Why did {downstream} change?",
        ]

    return list(DEMO_QUERIES.keys())[:3]


def get_variable_catalog() -> list[dict[str, object]]:
    """Return UI-friendly metadata for supported variables."""
    catalog: list[dict[str, object]] = []
    for variable_name in VARIABLE_NAMES:
        aliases = _aliases_for_variable(variable_name)
        examples = build_query_suggestions(variable_name, DEFAULT_TARGETS.get(variable_name))
        catalog.append(
            {
                "name": variable_name,
                "label": variable_name,
                "description": VARIABLE_DESCRIPTIONS.get(variable_name, variable_name),
                "aliases": aliases[:5],
                "examples": examples,
            }
        )
    return catalog


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
    llm_url = _get_llm_url()
    if not llm_url:
        return {}

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

    if source and not target and source in DEFAULT_TARGETS:
        target = DEFAULT_TARGETS[source]

    if source and not target:
        downstream = DOWNSTREAM_GRAPH.get(source, [])
        if downstream:
            target = downstream[0]

    # If we still have a source but no target, try LLM suggestion
    if source and not target:
        llm_url = _get_llm_url()
        try:
            if llm_url:
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

    if any(w in query.lower() for w in ["decrease", "reduce", "lower", "cut", "drop", "decline"]):
        return -20.0
    if any(w in query.lower() for w in ["increase", "raise", "boost", "grow", "tighten"]):
        return 20.0

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
        if source not in VARIABLE_NAMES:
            source = None
        if target not in VARIABLE_NAMES:
            target = None
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
        "value": 30.0,
    },
    "What if we increase detection policy strictness by 20%?": {
        "source": "FraudPolicyStrictness",
        "target": "SIMFraudDetectionRate",
        "value": 20.0,
    },
    "Why did chargeback volume increase?": {
        "source": "SIMBoxFraudAttempts",
        "target": "RevenueLeakageVolume",
        "value": 10.0,
    },
    "Show me the temporal impact of fraud on trust.": {
        "source": "SIMBoxFraudAttempts",
        "target": "SubscriberRetentionScore",
        "value": -5.0,
    },
}
