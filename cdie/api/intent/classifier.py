"""
CDIE v5 — Intent Parser Classifier
"""

import re

from cdie.api.intent.constants import (
    COUNTERFACTUAL_PATTERNS,
    INTERVENTION_PATTERNS,
    ROOT_CAUSE_PATTERNS,
    TEMPORAL_PATTERNS,
)
from cdie.api.intent.extractor import (
    extract_entities_llm,
    extract_magnitude,
    extract_variables,
)
from cdie.pipeline.data_generator import VARIABLE_NAMES


from typing import Any

def classify_query(query: str) -> dict[str, Any]:
    """
    Classify a user query into one of four types.
    Returns classification with extracted variables and magnitude.
    """
    q_lower = query.lower().strip()

    if not q_lower:
        return {'type': 'error', 'message': 'Query cannot be empty.'}

    scores = {
        'intervention': 0,
        'counterfactual': 0,
        'root_cause': 0,
        'temporal': 0,
    }

    for pattern in INTERVENTION_PATTERNS:
        if re.search(pattern, q_lower):
            scores['intervention'] += 1

    for pattern in COUNTERFACTUAL_PATTERNS:
        if re.search(pattern, q_lower):
            scores['counterfactual'] += 1

    for pattern in ROOT_CAUSE_PATTERNS:
        if re.search(pattern, q_lower):
            scores['root_cause'] += 1

    for pattern in TEMPORAL_PATTERNS:
        if re.search(pattern, q_lower):
            scores['temporal'] += 1

    best_type = max(scores, key=lambda k: scores[k])
    if scores[best_type] == 0:
        best_type = 'intervention'

    # Try LLM extraction first, fall back to regex
    llm_entities = extract_entities_llm(query)

    if llm_entities:
        source = llm_entities.get('source')
        target = llm_entities.get('target')
        magnitude = llm_entities.get('magnitude', 20.0)
        if source not in VARIABLE_NAMES:
            source = None
        if target not in VARIABLE_NAMES:
            target = None
    else:
        source, target = extract_variables(query)
        magnitude = extract_magnitude(query)

    is_ambiguous = scores[best_type] <= 1 and sum(1 for s in scores.values() if s > 0) > 1

    return {
        'type': best_type,
        'source': source,
        'target': target,
        'magnitude': magnitude,
        'confidence': 'high' if scores[best_type] >= 2 else 'medium' if scores[best_type] == 1 else 'low',
        'ambiguous': is_ambiguous,
        'all_scores': scores,
    }
