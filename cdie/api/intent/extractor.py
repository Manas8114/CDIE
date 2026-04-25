"""
CDIE v5 — Intent Parser Extractor
"""

import contextlib
import difflib
import json
import os
import re

import requests  # type: ignore[import-untyped]
from typing import Any, cast

from cdie.api.intent.constants import (
    DEFAULT_TARGETS,
    DOWNSTREAM_GRAPH,
    VARIABLE_ALIASES,
)
from cdie.pipeline.data_generator import VARIABLE_NAMES


def _get_llm_url() -> str | None:
    llm_url = os.environ.get('OPEA_LLM_ENDPOINT')
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


def extract_entities_llm(query: str) -> dict[str, Any]:
    """Extract entities using OPEA LLM endpoint."""
    llm_url = _get_llm_url()
    if not llm_url:
        return {}

    prompt = f"""
    You are a causal inference entity extractor.
    Extract the source (intervention), target (outcome), and magnitude (percentage change) from the following query.
    Available variables: {', '.join(VARIABLE_NAMES)}

    Query: "{query}"

    Return ONLY a JSON object with keys: "source", "target", "magnitude".
    If magnitude is not specified, use 20.0. If a variable is not in the list, return null.
    """

    with contextlib.suppress(Exception):
        response = requests.post(
            f'{llm_url}/generate',
            json={
                'inputs': prompt,
                'parameters': {'max_new_tokens': 128, 'temperature': 0.01},
            },
            timeout=5,
        )
        if response.status_code == 200:
            text = response.json().get('generated_text', '')
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return cast(dict[str, Any], json.loads(match.group(0)))
    return {}


def extract_variables(query: str) -> tuple[str | None, str | None]:
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
        with contextlib.suppress(Exception):
            if llm_url:
                prompt = (
                    f'Given these variables: {", ".join(VARIABLE_NAMES)}, '
                    f"what is the most logical downstream outcome of '{source}' "
                    f'in a telecom fraud SCM? Return only the variable name.'
                )
                msg = {'inputs': prompt, 'parameters': {'max_new_tokens': 32}}
                res = requests.post(f'{llm_url}/generate', json=msg, timeout=3)
                if res.status_code == 200:
                    suggested = res.json().get('generated_text', '').strip().split('\n')[0].strip()
                    if suggested in VARIABLE_NAMES:
                        target = suggested

    # Fallback default target
    if source and not target:
        target = 'ARPUImpact'

    return source, target


def extract_magnitude(query: str) -> float:
    """Extract magnitude percentage from query."""
    patterns = [
        r'(\d+)\s*%\s*(?:increase|rise|growth|up|higher|boost)s?',
        r'(?:increase|raise|boost|grow|tighten)s?\s+(?:by\s+)?(\d+)\s*%',
        r'(\d+)\s*%\s*(?:decrease|decline|drop|down|lower|reduction|reduce)s?',
        r'(?:decrease|reduce|lower|cut|drop|decline)s?\s+(?:by\s+)?(\d+)\s*%',
        r'(\d+)\s*%',
    ]

    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            pct = float(match.group(1))
            if any(w in query.lower() for w in ['decrease', 'reduce', 'lower', 'cut', 'drop', 'decline']):
                return -pct
            return pct

    if any(w in query.lower() for w in ['decrease', 'reduce', 'lower', 'cut', 'drop', 'decline']):
        return -20.0
    if any(w in query.lower() for w in ['increase', 'raise', 'boost', 'grow', 'tighten']):
        return 20.0

    return 20.0
