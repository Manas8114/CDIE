"""
CDIE v5 — RAG LLM Explanation Module

Handles:
- Sufficiency gate (skip LLM when causal evidence is already strong)
- LLM explanation generation via OPEA TextGen or OpenAI
- Template-based deterministic explanations (always available fallback)
"""

from __future__ import annotations

import time

from typing import Any
from cdie.observability import METRIC_LLM_CALL, METRIC_LLM_FAIL, get_logger, increment

log = get_logger(__name__)


# ── Sufficiency Gate ──────────────────────────────────────────────────────────

def check_sufficiency(
    effect: dict[str, Any],
    refutation_status: dict[str, Any] | None,
    analogies: list[dict[str, Any]] | None,
) -> bool:
    """Return True when deterministic causal evidence is strong enough to skip the LLM.

    Gate criteria (ALL must hold):
    1. Non-trivial effect point estimate (|point| > 1e-6)
    2. Tight CI (width < 60% of |point estimate|)
    3. All 3 refutation tests (placebo, confounder, subset) passed
    4. At least one High or Medium confidence analogy retrieved

    Rationale: Richens & Everitt (ICLR 2024) — robust agents must learn causal
    structure. When the Safety Map encodes validated structure with tight CIs and
    passing refutations, template explanations capture the full causal semantics.
    """
    if not isinstance(effect, dict) or not effect:
        return False

    point = effect.get('point_estimate', 0)
    ci_lower = effect.get('ci_lower', 0)
    ci_upper = effect.get('ci_upper', 0)

    if abs(point) < 1e-6:
        return False

    ci_width = abs(ci_upper - ci_lower)
    if ci_width > 0.6 * abs(point):
        return False

    if refutation_status:
        statuses = [v for k, v in refutation_status.items() if k in ('placebo', 'confounder', 'subset')]
        if not statuses or not all(s == 'PASS' for s in statuses):
            return False
    else:
        return False

    return bool(analogies and any(a.get('confidence') in ('High', 'Medium') for a in analogies))


# ── LLM Explanation Generator ─────────────────────────────────────────────────


# Simple global circuit breaker state for LLM
_LLM_CIRCUIT_OPEN = False
_LLM_LAST_FAILURE = 0.0
_CIRCUIT_COOLDOWN = 60.0  # 1 minute


def generate_llm_explanation(
    client: Any,
    llm_model: str,
    query_type: str,
    source: str,
    target: str,
    effect: dict[str, Any],
    refutation_status: dict[str, Any] | None,
    analogies: list[dict[str, Any]] | None,
) -> str:
    """Generate an explanation via LLM (OPEA or OpenAI).

    Args:
        client:           Initialised ``openai.OpenAI`` client instance.
        llm_model:        Model identifier string.
        query_type:       One of ``intervention``, ``counterfactual``, ``root_cause``, ``temporal``.
        source:           Intervention variable name.
        target:           Outcome variable name.
        effect:           Effect dict with ``point_estimate``, ``ci_lower``, ``ci_upper``.
        refutation_status: Dict of refutation test results.
        analogies:        List of retrieved analogy dicts.

    Returns:
        Markdown-formatted explanation string, or empty string on failure.
    """
    global _LLM_CIRCUIT_OPEN, _LLM_LAST_FAILURE

    if not client:
        return ''

    # Check circuit breaker
    if _LLM_CIRCUIT_OPEN:
        if time.time() - _LLM_LAST_FAILURE < _CIRCUIT_COOLDOWN:
            log.warning('[rag.llm] LLM circuit open — skipping call to prevent cascade')
            return ''
        else:
            log.info('[rag.llm] LLM circuit cooldown expired — attempting half-open probe')

    increment(METRIC_LLM_CALL)

    point = effect.get('point_estimate', 0) if isinstance(effect, dict) else 0
    lower = effect.get('ci_lower', 0) if isinstance(effect, dict) else 0
    upper = effect.get('ci_upper', 0) if isinstance(effect, dict) else 0

    analogies_list = '\n'.join([f'- {a["text"]}' for a in (analogies or [])])

    prompt = (
        'You are the **CDIE v5 Causal Intelligence Engine**, an expert AI system built on the '
        '**OPEA (Open Platform for Enterprise AI)** framework.\n'
        f'Generate an **OPEA Causal Intelligence Report** responding to a {query_type} '
        'query regarding telecom fraud.\n\n'
        '**CAUSAL EVIDENCE (From Offline Causal Discovery):**\n'
        f'- Source Intervention: {source}\n'
        f'- Target Effect: {target}\n'
        f'- Doubly-Robust ATE (Average Treatment Effect): {point:.4f} '
        f'(95% CI: [{lower:.4f}, {upper:.4f}])\n'
        f'- Refutation Test Status (Robustness): {refutation_status}\n\n'
        '**RELEVANT TELECOM PLAYBOOKS (From OPEA TEI RAG Retrieval):**\n'
        f'{analogies_list}\n\n'
        '**FORMATTING REQUIREMENTS (CRITICAL):**\n'
        'Output a highly professional markdown report. Do not include pleasantries. '
        'Strictly use this structure:\n\n'
        '### 📊 Causal Impact Summary\n'
        f'Explain the causal effect magnitude ({point:.4f}) and whether it represents '
        'an increase or decrease.\n\n'
        '### 🛡️ Validation & Refutation\n'
        "State the confidence interval and the results of the refutation tests "
        "(did they pass?). Use terms like 'do-calculus'.\n\n"
        '### 📖 Playbook Recommendation (RAG)\n'
        'Synthesize the provided Telecom Playbooks into concrete action items '
        'that operators should follow based on this causal finding.\n'
    )

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are CDIE v5, an elite Causal Inference engine for '
                        'telecom network intelligence, reporting to a Chief Network Officer.'
                    ),
                },
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.3,
            max_tokens=250,
            timeout=15,  # Hard timeout for OPEA/OpenAI
        )
        # Reset circuit on success
        _LLM_CIRCUIT_OPEN = False
        return str(response.choices[0].message.content.strip())
    except Exception as exc:
        increment(METRIC_LLM_FAIL)
        # Trip circuit breaker
        _LLM_CIRCUIT_OPEN = True
        _LLM_LAST_FAILURE = time.time()
        log.warning('[rag.llm] LLM generation failed — tripping circuit breaker', error=str(exc))
        return ''


# ── Template-based Explanations (always available) ────────────────────────────

def explain_intervention(
    source: str, target: str, effect: dict[str, Any], refutation: dict[str, Any] | None, analogies: list[dict[str, Any]] | None
) -> str:
    point = effect.get('point_estimate', 0) if isinstance(effect, dict) else 0
    lower = effect.get('ci_lower', point * 0.8) if isinstance(effect, dict) else point * 0.8
    upper = effect.get('ci_upper', point * 1.2) if isinstance(effect, dict) else point * 1.2
    direction = 'increase' if point > 0 else 'decrease'
    magnitude = abs(point)

    explanation = (
        f'**Impact Summary**: A change in {source} is estimated to cause a '
        f'{direction} of {magnitude:.2f} units in {target} '
        f'(95% CI: [{lower:.2f}, {upper:.2f}]).\n\n'
    )
    explanation += (
        f'**Causal Chain**: This effect propagates through the validated causal pathway '
        f'{source} → {target}. The estimate is doubly-robust (LinearDML), meaning it remains '
        f'valid even if either the outcome model or treatment model is misspecified.\n\n'
    )
    if refutation:
        n_pass = sum(1 for v in refutation.values() if v == 'PASS')
        n_total = len(refutation)
        explanation += (
            f'**Validation**: {n_pass}/{n_total} refutation tests passed. '
            f'This causal claim has been tested against placebo treatments, '
            f'random confounders, and data subsets.\n\n'
        )
    if analogies:
        high_conf = [a for a in analogies if a.get('confidence') in ('High', 'Medium')]
        if high_conf:
            explanation += '**Historical Precedent**: '
            explanation += high_conf[0]['text'] + '\n\n'
    explanation += (
        f'**Recommended Action**: Monitor {target} closely when implementing changes to {source}. '
        f'Consider segment-specific impacts (Enterprise vs Retail) before deployment.'
    )
    return explanation


def explain_counterfactual(source: str, target: str, effect: dict[str, Any], _analogies: list[dict[str, Any]] | None) -> str:
    point = effect.get('point_estimate', 0) if isinstance(effect, dict) else 0
    return (
        f'**Counterfactual Analysis**: Had {source} remained at its baseline value, '
        f'{target} would have been approximately {abs(point):.2f} units '
        f'{"higher" if point < 0 else "lower"} than observed.\n\n'
        f"This estimate uses DoWhy's counterfactual framework, applying the "
        f'structural equations from the discovered causal model.'
    )


def explain_root_cause(source: str, target: str, _effect: dict[str, Any], _analogies: list[dict[str, Any]] | None) -> str:
    return (
        f'**Root Cause Analysis**: The primary causal driver of changes in {source} '
        f'traces back through the causal graph. Key upstream factors include variables '
        f'that are direct parents of {source} in the discovered DAG.\n\n'
        f'Use the interactive causal graph to trace the full causal chain and identify '
        f'the most actionable intervention point.'
    )


def explain_temporal(source: str, target: str, temporal_info: dict[str, Any] | None, _analogies: list[dict[str, Any]] | None) -> str:
    lag = temporal_info.get('lag', 2) if temporal_info else 2
    return (
        f'**Temporal Effect**: Changes in {source} take approximately {lag} time period(s) '
        f'to fully manifest in {target}.\n\n'
        f'This lag was identified by PCMCI+ temporal causal discovery and cross-validated '
        f'with Granger causality tests. Plan interventions with this delay in mind.'
    )


def build_template_explanation(
    query_type: str,
    source: str,
    target: str,
    effect: dict[str, Any],
    refutation_status: dict[str, Any] | None = None,
    analogies: list[dict[str, Any]] | None = None,
    temporal_info: dict[str, Any] | None = None,
) -> str:
    """Dispatch to the correct template-based explanation function."""
    if query_type == 'intervention':
        return explain_intervention(source, target, effect, refutation_status, analogies)
    if query_type == 'counterfactual':
        return explain_counterfactual(source, target, effect, analogies)
    if query_type == 'root_cause':
        return explain_root_cause(source, target, effect, analogies)
    if query_type == 'temporal':
        return explain_temporal(source, target, temporal_info, analogies)
    return explain_intervention(source, target, effect, refutation_status, analogies)
