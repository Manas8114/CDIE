"""
CDIE v5 — Intent Parser Package
"""

from cdie.api.intent.classifier import classify_query
from cdie.api.intent.constants import (
    DEFAULT_TARGETS,
    DEMO_QUERIES,
    VARIABLE_ALIASES,
    VARIABLE_DESCRIPTIONS,
)
from cdie.api.intent.extractor import (
    extract_magnitude,
    extract_variables,
    suggest_variables,
)

__all__ = [
    'VARIABLE_ALIASES',
    'VARIABLE_DESCRIPTIONS',
    'DEFAULT_TARGETS',
    'DEMO_QUERIES',
    'suggest_variables',
    'extract_variables',
    'extract_magnitude',
    'classify_query',
    'get_variable_catalog',
    'build_query_suggestions',
]

# Additional helpers from intent_parser.py
from cdie.pipeline.data_generator import VARIABLE_NAMES


def _aliases_for_variable(variable_name: str) -> list[str]:
    from cdie.api.intent.extractor import _aliases_for_variable as _aliases
    return _aliases(variable_name)

def build_query_suggestions(source: str | None = None, target: str | None = None) -> list[str]:
    """Return example questions tailored to the parsed variables."""
    if source and target and source != target:
        return [
            f'What happens if {source} increases by 20%?',
            f'What happens if {source} decreases by 20%?',
            f'How does {source} affect {target}?',
        ]

    if source:
        downstream = DEFAULT_TARGETS.get(source, 'ARPUImpact')
        if downstream == source:
            return [
                f'What happens if {source} increases by 20%?',
                f'What happens if {source} decreases by 20%?',
                f'Why did {source} change?',
            ]
        return [
            f'What happens if {source} increases by 20%?',
            f'How does {source} affect {downstream}?',
            f'Why did {downstream} change?',
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
                'name': variable_name,
                'label': variable_name,
                'description': VARIABLE_DESCRIPTIONS.get(variable_name, variable_name),
                'aliases': aliases[:5],
                'examples': examples,
            }
        )
    return catalog
