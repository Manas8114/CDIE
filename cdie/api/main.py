"""
CDIE v4 — FastAPI Backend (Online Phase)
Serves validated causal intervention results via pre-computed Safety Map.
"""

import contextlib
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast, Optional, Union

import pandas as pd
import psutil
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydantic import BaseModel

from cdie.api.drift import DriftAnalyzer
from cdie.api.intent import (
    DEFAULT_TARGETS,
    DEMO_QUERIES,
    VARIABLE_ALIASES,
    build_query_suggestions,
    classify_query,
    get_variable_catalog,
    suggest_variables,
)
from cdie.api.lookup import SafetyMapLookup
from cdie.api.models import (
    BatchPrescribeRequest,
    BatchQueryRequest,
    BenchmarkResponse,
    CATESegment,
    CATLResponse,
    EffectResult,
    ExpertCorrectionRequest,
    ExpertCorrectionResponse,
    GraphResponse,
    HealthResponse,
    PrescribeRequest,
    PrescribeResponse,
    QueryRequest,
    QueryResponse,
    RefutationStatus,
    VariableCatalogResponse,
)
from cdie.api.rag import ExplanationEngine
from cdie.config import (
    ALLOWED_ORIGINS,
    APP_TITLE,
    DATA_DIR,
    CATE_CONFIG,
    HEURISTICS_CONFIG,
    RATE_LIMIT_STRING,
    SECURE_HEADERS,
    REDIS_URL,
    VERSION,
)
from cdie.observability import (
    METRICS_ENABLED,
    get_logger,
    get_metrics,
)
from cdie.pipeline.data_generator import VARIABLE_NAMES
from cdie.pipeline.datastore import DataStoreManager
from cdie.runtime import get_runtime_paths

log = get_logger(__name__)

# Initialize Store Manager
store_manager = DataStoreManager()


# Initialize Rate Limiter (Redis-backed for multi-container; in-memory for dev)
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=REDIS_URL if REDIS_URL else 'memory://',
)

# Global instances
RUNTIME_PATHS = get_runtime_paths(DATA_DIR)
safety_map_lookup = SafetyMapLookup()
explanation_engine = ExplanationEngine()


def _normalize_magnitude(magnitude: float) -> float:
    """Accept both 20 and 0.2 style inputs and normalize to percentage units."""
    if magnitude != 0 and abs(magnitude) <= 1:
        return magnitude * 100
    return magnitude


def _normalize_relative_fraction(magnitude: float) -> float:
    """Accept both 20 and 0.2 style inputs and normalize to fractional units."""
    if magnitude != 0 and abs(magnitude) > 1:
        return magnitude / 100
    return magnitude


def _get_available_llm_endpoints() -> tuple[str | None, str | None]:
    """Only use LLM endpoints when explicitly configured."""
    tgi_url = os.environ.get('TGI_ENDPOINT')
    llm_url = os.environ.get('OPEA_LLM_ENDPOINT')
    return (tgi_url or None, llm_url or None)


def _build_trust_message(match_type: str, confidence_label: str) -> str:
    if match_type == 'fallback':
        return (
            'No validated scenario matched this exact query. The result is a heuristic '
            'fallback estimate and should be treated as directional guidance only.'
        )
    if match_type == 'nearest':
        return (
            'This answer uses the nearest validated scenario, not an exact precomputed '
            'match. Review the effect size with caution before acting on it.'
        )
    if confidence_label == 'UNPROVEN':
        return (
            'A scenario was found, but validation checks or drift warnings reduce trust '
            'in this estimate. Treat it as unproven until reviewed.'
        )
    return 'This answer comes from a validated precomputed scenario in the Safety Map.'


def _build_evidence_tier(match_type: str, confidence_label: str) -> str:
    if match_type == 'fallback':
        return 'heuristic'
    if confidence_label == 'UNPROVEN':
        return 'unproven'
    if match_type == 'nearest':
        return 'validated-nearest'
    return 'validated'


def _resolve_variable_name(raw_name: str | None) -> str | None:
    """Resolve free-form or aliased variable names to the canonical SCM name."""
    if not raw_name:
        return None

    candidate = raw_name.strip()
    if not candidate:
        return None

    if candidate in VARIABLE_NAMES:
        return candidate

    lowered = candidate.lower()
    if lowered in VARIABLE_ALIASES:
        return VARIABLE_ALIASES[lowered]

    for alias, variable_name in sorted(VARIABLE_ALIASES.items(), key=lambda item: -len(item[0])):
        if alias in lowered:
            return variable_name

    return None


# Config removed to cdie.config


def _load_safety_map() -> bool:
    """Load the most reliable Safety Map representation available."""
    db_path = DATA_DIR / 'safety_map.db'
    json_path = DATA_DIR / 'safety_map.json'
    runtime_db_path = RUNTIME_PATHS['runtime_db']

    loaded = False
    if db_path.exists():
        loaded = safety_map_lookup.load(str(db_path))

    if not loaded and runtime_db_path.exists():
        loaded = safety_map_lookup.load(str(runtime_db_path))

    if not loaded and json_path.exists():
        loaded = safety_map_lookup.load(str(json_path))

    if loaded:
        log.info('[API] Safety Map loaded', path=str(safety_map_lookup.db_path))
    else:
        log.warning(
            '[API] Safety Map not found — run the offline pipeline first',
            db_path=str(db_path),
            json_path=str(json_path),
            hint='python -m cdie.pipeline.run_pipeline',
        )
    return loaded


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    # Load Safety Map on startup
    _load_safety_map()
    yield


app = FastAPI(
    title=APP_TITLE,
    version=VERSION,
    description='Pre-computed causal intervention lookup with validation and uncertainty quantification.',
    lifespan=lifespan,
)

# Register Limiter
app.state.limiter = limiter
async def _custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Any:
    from slowapi import _rate_limit_exceeded_handler
    return _rate_limit_exceeded_handler(request, exc)

app.add_exception_handler(RateLimitExceeded, _custom_rate_limit_exceeded_handler) # type: ignore

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'PUT', 'DELETE'],
    allow_headers=['*'],
)


@app.middleware('http')
async def add_secure_headers(request: Request, call_next: Any) -> Any:
    response = await call_next(request)
    for header, value in SECURE_HEADERS.items():
        response.headers[header] = value
    return response


@app.get('/metrics')
async def get_metrics_endpoint() -> dict[str, Any]:
    """Return in-process request and service counters.

    Requires ``CDIE_ENABLE_METRICS=1`` (default: enabled).
    Returns HTTP 404 when metrics are disabled.
    """
    if not METRICS_ENABLED:
        raise HTTPException(404, 'Metrics are disabled. Set CDIE_ENABLE_METRICS=1 to enable.')
    return {'metrics': get_metrics(), 'enabled': True}


@app.get('/health', response_model=HealthResponse)
@limiter.limit(RATE_LIMIT_STRING)
async def health(request: Request) -> HealthResponse:
    """System health check."""
    metadata = safety_map_lookup.get_metadata()
    memory = psutil.Process().memory_info().rss / 1024 / 1024 if psutil else 0

    # Check OPEA connectivity
    opea_health = {}
    import requests
    for service, url in [
        ('llm', os.environ.get('OPEA_LLM_ENDPOINT')),
        ('embedding', os.environ.get('OPEA_EMBEDDING_ENDPOINT')),
        ('reranking', os.environ.get('OPEA_RERANKING_ENDPOINT')),
    ]:
        if url:
            # Try both /v1/health_check and /health
            ok = False
            for path in ['/v1/health_check', '/health', '/v1/health']:
                with contextlib.suppress(Exception):
                    res = requests.get(f'{url}{path}', timeout=1)
                    if res.status_code == 200:
                        ok = True
                        break
            opea_health[service] = 'UP' if ok else 'DOWN'
        else:
            opea_health[service] = 'NOT_CONFIGURED'

    return HealthResponse(
        status='healthy' if safety_map_lookup.is_loaded() else 'degraded',
        safety_map_hash=metadata.get('sha256_hash', 'not_loaded'),
        last_computed=metadata.get('created_at', 'unknown'),
        ks_status='OK',
        memory_mb=round(memory, 1),
        n_scenarios=metadata.get('n_scenarios', 0),
        storage_backend=metadata.get('storage_backend', 'unloaded'),
        opea_status=opea_health,
    )


@app.post('/query', response_model=QueryResponse)
@limiter.limit(RATE_LIMIT_STRING)
async def query(request: Request, query_data: QueryRequest) -> QueryResponse:
    """Process a causal query against the Safety Map."""
    if not safety_map_lookup.is_loaded():
        raise HTTPException(503, 'Safety Map not loaded. Run the offline pipeline first.')

    query_text = query_data.query.strip()
    if not query_text:
        raise HTTPException(400, 'Query cannot be empty.')

    query_id: str = str(uuid.uuid4())
    query_id = query_id[:8]

    classification = classify_query(query_text)

    # Extraction logic
    if query_data.scenario:
        # Validate scenario ID: source__target__magnitude_key
        parts = query_data.scenario.split('__')
        if len(parts) == 3:
            source, target, magnitude_key = parts
            from cdie.config import MAGNITUDE_LEVELS
            if magnitude_key in MAGNITUDE_LEVELS:
                magnitude = _normalize_magnitude(MAGNITUDE_LEVELS[magnitude_key] * 100)
            else:
                log.warning('[API] Invalid magnitude key in scenario ID', key=magnitude_key)
                magnitude = _normalize_magnitude(query_data.magnitude or 20.0)
        else:
            raise HTTPException(400, "Invalid scenario ID format. Expected 'source__target__magnitude_key'")
    elif query_text in DEMO_QUERIES:
        preset = DEMO_QUERIES[query_text]
        source = str(preset.get('source') or '')
        target = str(preset.get('target') or '')
        magnitude = _normalize_magnitude(float(cast(float, preset.get('value') or 10.0)))
    else:
        source = str(classification['source'] or '')
        target = str(classification['target'] or '')
        magnitude = _normalize_magnitude(float(cast(float, classification.get('magnitude', 20))))

    # Magnitude override from request body
    if query_data.magnitude is not None:
        magnitude = _normalize_magnitude(query_data.magnitude)

    variable_suggestions = suggest_variables(query_text)
    query_suggestions = build_query_suggestions(source or None, target or None)

    if not source:
        raise HTTPException(
            422,
            'Could not identify a valid variable in your query. Try one of: '
            + ', '.join(variable_suggestions or VARIABLE_NAMES[:5]),
        )

    # Lookup scenario
    scenario, is_exact = safety_map_lookup.find_best_scenario(source, target, magnitude)
    match_type = 'exact' if scenario and is_exact else 'nearest' if scenario else 'fallback'

    # Universal fallback: if still no scenario, generate a realistic one from the query variables
    if not scenario and source:
        h = HEURISTICS_CONFIG.get('fallback_heuristics', {})
        multiplier = h.get('magnitude_to_ate_multiplier', 0.45)
        default_pe = h.get('default_point_estimate', 0.05)
        ci_l_m = h.get('ci_lower_multiplier', 0.65)
        ci_u_m = h.get('ci_upper_multiplier', 1.35)
        def_ratio = h.get('default_intervention_ratio', 0.1)

        fallback_pe = round(magnitude / 100 * multiplier, 4) if magnitude else default_pe

        # Build CATE segments from config
        cate_by_segment = {}
        for seg_name, seg_conf in CATE_CONFIG.get('segments', {}).items():
            s_ate = round(fallback_pe * seg_conf.get('ate_multiplier', 1.0), 4)
            cate_by_segment[seg_name] = {
                'ate': s_ate,
                'ci_lower': round(fallback_pe * seg_conf.get('ci_lower_multiplier', 0.8), 4),
                'ci_upper': round(fallback_pe * seg_conf.get('ci_upper_multiplier', 1.2), 4),
                'n_samples': seg_conf.get('n_samples_default', 1000),
                'risk_level': seg_conf.get('risk_level', 'unknown'),
            }

        scenario = {
            'id': f'fallback_{source}__{target}',
            'source': source,
            'target': target or 'ARPUImpact',
            'magnitude_value': magnitude / 100 if magnitude else def_ratio,
            'effect': {
                'point_estimate': fallback_pe,
                'ci_lower': round(fallback_pe * ci_l_m, 4),
                'ci_upper': round(fallback_pe * ci_u_m, 4),
                'confidence_level': 0.95,
                'ate_used': fallback_pe,
                'intervention_amount': magnitude / 100 if magnitude else def_ratio,
            },
            'causal_path': f'{source} → {target or "ARPUImpact"}',
            'refutation_status': 'ESTIMATED',
            'cate_by_segment': cate_by_segment,
        }
        is_exact = False
        match_type = 'fallback'

    if not scenario:
        # Try with default magnitude
        scenario = safety_map_lookup.find_scenario(source, target)
        if scenario:
            match_type = 'nearest'

    # KS staleness check
    training_dist = safety_map_lookup._get_store_val('training_distributions', {})
    ks_result = {'warning': False, 'ks_statistic': 0.0}
    if source in training_dist:
        mean_val = training_dist[source].get('mean', 0)
        ks_result = safety_map_lookup.check_staleness(source, mean_val)

    # Build effect result
    effect = None
    refutation = RefutationStatus()
    causal_path = ''
    cate_segments = []
    confidence_label = 'ESTIMATED'

    if scenario:
        eff = scenario.get('effect', {})
        effect = EffectResult(
            point_estimate=eff.get('point_estimate', 0),
            ci_lower=eff.get('ci_lower', 0),
            ci_upper=eff.get('ci_upper', 0),
            confidence_level=eff.get('confidence_level', 0.95),
            ate_used=eff.get('ate_used', 0),
            intervention_amount=eff.get('intervention_amount', 0),
        )
        causal_path = scenario.get('causal_path', f'{source} → {target}')
        confidence_label = scenario.get('refutation_status', 'VALIDATED')

        # Extract refutation from graph edges
        graph = safety_map_lookup.get_graph()
        n_fail = 0
        for edge in graph.get('edges', []):
            if edge.get('from') == source and edge.get('to') == target:
                tests = edge.get('tests', [])
                if len(tests) >= 3:
                    refutation = RefutationStatus(
                        placebo=tests[0].get('status', 'NOT_TESTED'),
                        confounder=tests[1].get('status', 'NOT_TESTED'),
                        subset=tests[2].get('status', 'NOT_TESTED'),
                    )
                    n_fail = sum(1 for t in tests if t.get('status') == 'FAIL')
                break

        if n_fail > 0 or ks_result.get('warning', False) or confidence_label == 'UNKNOWN':
            confidence_label = 'UNPROVEN'

        # CATE segments
        for seg_name, seg_data in scenario.get('cate_by_segment', {}).items():
            if isinstance(seg_data, dict) and seg_data.get('ate') is not None:
                ate_val = seg_data.get('ate', 0)
                risk = 'Critical' if abs(ate_val) > 0.5 else 'High' if abs(ate_val) > 0.2 else 'Low'
                cate_segments.append(
                    CATESegment(
                        segment=seg_name,
                        ate=ate_val,
                        ci_lower=seg_data.get('ci_lower'),
                        ci_upper=seg_data.get('ci_upper'),
                        n_samples=seg_data.get('n_samples', 0),
                        risk_level=risk,
                    )
                )

    # Generate explanation
    analogies = explanation_engine.retrieve_analogies(query_text)
    explanation = explanation_engine.generate_explanation(
        query_type=classification['type'],
        source=source,
        target=target,
        effect=scenario.get('effect', {}) if scenario else {},
        refutation_status={
            'placebo': refutation.placebo,
            'confounder': refutation.confounder,
            'subset': refutation.subset,
        },
        analogies=analogies,
    )

    extrapolation_note = ''
    if match_type == 'nearest':
        extrapolation_note = ' (Nearest validated scenario)'
    elif match_type == 'fallback':
        extrapolation_note = ' (Heuristic fallback estimate)'

    evidence_tier = _build_evidence_tier(match_type, confidence_label)
    trust_message = _build_trust_message(match_type, confidence_label)

    return QueryResponse(
        query_type=classification['type'],
        query_id=query_id,
        scenario_id=scenario.get('id') if scenario else None,
        source=source,
        target=target,
        magnitude=f'{magnitude:+.0f}%',
        effect=effect,
        causal_path=causal_path + extrapolation_note,
        refutation_status=refutation,
        ks_warning=bool(ks_result.get('warning', False)),
        ks_statistic=ks_result.get('ks_statistic', 0),
        kl_divergence=ks_result.get('kl_divergence', 0),
        drift_detected=ks_result.get('drift_detected', False),
        explanation=explanation,
        historical_analogies=[a['text'] for a in analogies[:3]],
        cate_segments=cate_segments,
        feature_importance=scenario.get('feature_importance', {}) if scenario else {},
        confidence_label=confidence_label,
        match_type=match_type,
        evidence_tier=evidence_tier,
        trust_message=trust_message,
        used_fallback=match_type == 'fallback',
        suggested_queries=query_suggestions,
        available_variables=variable_suggestions or VARIABLE_NAMES[:5],
    )


@app.post('/api/query/batch', response_model=list[QueryResponse])
@limiter.limit(RATE_LIMIT_STRING)
async def query_batch(request: Request, batch_data: BatchQueryRequest) -> list[QueryResponse]:
    """Process multiple causal queries in bulk."""
    results = []
    for q_req in batch_data.queries:
        try:
            res = await query(request, q_req)
            results.append(res)
        except Exception as exc:
            log.warning('[API] Batch query item failed', query=q_req.query, error=str(exc))
            continue
    return results


@app.get('/graph', response_model=GraphResponse)
@limiter.limit(RATE_LIMIT_STRING)
async def get_graph(request: Request) -> GraphResponse:
    """Return the causal graph."""
    if not safety_map_lookup.is_loaded():
        raise HTTPException(503, 'Safety Map not loaded.')
    return GraphResponse(**safety_map_lookup.get_graph())


@app.get('/benchmark', response_model=BenchmarkResponse)
@limiter.limit(RATE_LIMIT_STRING)
async def get_benchmarks(request: Request) -> BenchmarkResponse:
    """Return benchmark results."""
    if not safety_map_lookup.is_loaded():
        raise HTTPException(503, 'Safety Map not loaded.')
    return BenchmarkResponse(**safety_map_lookup.get_benchmarks())

@app.get('/catl', response_model=CATLResponse)
@limiter.limit(RATE_LIMIT_STRING)
async def get_catl(request: Request) -> CATLResponse:
    """Return CATL assumption test results."""
    if not safety_map_lookup.is_loaded():
        raise HTTPException(503, 'Safety Map not loaded.')
    return CATLResponse(**safety_map_lookup.get_catl())


@app.get('/xgboost')
@limiter.limit(RATE_LIMIT_STRING)
async def get_xgboost(request: Request) -> dict[str, Any]:
    """Return XGBoost comparison data."""
    if not safety_map_lookup.is_loaded():
        raise HTTPException(503, 'Safety Map not loaded.')
    return safety_map_lookup.get_xgboost_comparison()


@app.get('/temporal')
@limiter.limit(RATE_LIMIT_STRING)
async def get_temporal(request: Request) -> dict[str, Any]:
    """Return temporal causal results."""
    if not safety_map_lookup.is_loaded():
        raise HTTPException(503, 'Safety Map not loaded.')
    return safety_map_lookup.get_temporal()


@app.get('/metadata')
@limiter.limit(RATE_LIMIT_STRING)
async def get_metadata(request: Request) -> dict[str, Any]:
    """Return Safety Map metadata."""
    return safety_map_lookup.get_metadata()


@app.get('/demo-queries')
@limiter.limit(RATE_LIMIT_STRING)
async def get_demo_queries(request: Request) -> dict[str, dict[str, Any]]:
    """Return pre-defined demo queries."""
    return DEMO_QUERIES


@app.get('/variables', response_model=VariableCatalogResponse)
@limiter.limit(RATE_LIMIT_STRING)
async def get_variables(request: Request) -> VariableCatalogResponse:
    """Return variable metadata and example prompts for the UI."""
    catalog = get_variable_catalog()
    from cdie.api.models import VariableInfo
    vars_list = [VariableInfo(**cast(dict[str, Any], v)) for v in catalog]
    return VariableCatalogResponse(variables=vars_list)


@app.post('/prescribe', response_model=PrescribeResponse)
@limiter.limit(RATE_LIMIT_STRING)
async def prescribe(request: Request, prescribe_data: PrescribeRequest) -> PrescribeResponse:
    """Find the top interventions to maximize or minimize a target using LLM target resolution."""
    if not safety_map_lookup.is_loaded():
        raise HTTPException(503, 'Safety Map not loaded.')

    # Use LLM to resolve target from natural language if needed
    tgi_url, llm_url = _get_available_llm_endpoints()
    raw_target = prescribe_data.target.strip()
    resolved_target = _resolve_variable_name(raw_target) or raw_target

    prompt = f"""
    Map the following target query to one of these valid causal variables: {', '.join(VARIABLE_NAMES)}
    Query: "{raw_target}"
    Return ONLY the exact variable name. If no match, return "ARPUImpact".
    """

    try:
        import requests

        # Attempt direct TGI first for fast /generate
        suggested = ''
        response = None

        if tgi_url:
            response = requests.post(
                f'{tgi_url}/generate',
                json={'inputs': prompt, 'parameters': {'max_new_tokens': 32}},
                timeout=3,
            )

        if response is None or response.status_code != 200:
            # Try OPEA OpenAI-compatible endpoint as fallback
            if llm_url:
                payload = {
                    'model': os.environ.get('LLM_MODEL_ID', 'Intel/neural-chat-7b-v3-3'),
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 32,
                }
                response = requests.post(f'{llm_url}/v1/chat/completions', json=payload, timeout=5)
                if response.status_code == 200:
                    suggested = response.json().get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        elif response is not None:
            suggested = response.json().get('generated_text', '').strip()

        if suggested:
            # Clean up response (some LLMs add preamble)
            suggested = suggested.split('\n')[-1].split(':')[-1].strip().strip('"').strip("'")
            normalized_suggested = _resolve_variable_name(suggested)
            if normalized_suggested:
                resolved_target = normalized_suggested
            else:
                normalized_raw = _resolve_variable_name(raw_target)
                if normalized_raw:
                    resolved_target = normalized_raw
    except Exception as e:
        print(f'[API] LLM Target Resolution failed: {e}. Falling back to fuzzy matching.')
        normalized_raw = _resolve_variable_name(raw_target)
        if normalized_raw:
            resolved_target = normalized_raw

    prescriptions = safety_map_lookup.find_prescriptions(
        target=resolved_target, limit=prescribe_data.limit, maximize=prescribe_data.maximize
    )

    return PrescribeResponse(
        target=resolved_target,
        prescriptions=prescriptions,
        message=(
            f"LLM resolved '{raw_target}' to '{resolved_target}'. "
            f"Found {len(prescriptions)} recommendations to "
            f"{'maximize' if prescribe_data.maximize else 'minimize'} {resolved_target}."
        ),
    )


@app.post('/api/batch/prescribe', response_model=list[PrescribeResponse])
@limiter.limit(RATE_LIMIT_STRING)
async def batch_prescribe(request: Request, batch_data: BatchPrescribeRequest) -> list[PrescribeResponse]:
    """Process multiple prescription requests in bulk."""
    results = []
    for p_req in batch_data.requests:
        try:
            res = await prescribe(request, p_req)
            results.append(res)
        except Exception as exc:
            log.warning('[API] Batch prescription item failed', target=p_req.target, error=str(exc))
            continue
    return results



@app.post('/expert/correct', response_model=ExpertCorrectionResponse)
@limiter.limit(RATE_LIMIT_STRING)
async def expert_correct(request: Request, correction_data: ExpertCorrectionRequest) -> ExpertCorrectionResponse:
    """Apply expert correction to causal graph."""
    corrections_path = DATA_DIR / 'prior_corrections.json'

    corrections = []
    if corrections_path.exists():
        with open(corrections_path) as f:
            corrections = json.load(f)

    corrections.append(
        {
            'from_node': correction_data.from_node,
            'to_node': correction_data.to_node,
            'action': correction_data.action,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        }
    )

    with open(corrections_path, 'w') as f:
        json.dump(corrections, f, indent=2)

    return ExpertCorrectionResponse(
        success=True,
        message=(
            f"Correction recorded: {correction_data.action} edge "
            f"{correction_data.from_node} → {correction_data.to_node}. "
            "Will be applied in next pipeline run."
        ),
    )


@app.get('/benchmark/latency')
@limiter.limit(RATE_LIMIT_STRING)
async def benchmark_latency(request: Request) -> dict[str, Any]:
    """Run latency benchmark against the Safety Map lookup."""
    if not safety_map_lookup.is_loaded():
        raise HTTPException(503, 'Safety Map not loaded.')

    import statistics

    test_queries = [
        ('SIMBoxFraudAttempts', 'ARPUImpact', 30),
        ('FraudPolicyStrictness', 'SIMFraudDetectionRate', 20),
        ('RevenueLeakageVolume', 'CashFlowRisk', 15),
        ('CallDataRecordVolume', 'NetworkLoad', 25),
        ('RegulatorySignal', 'ITURegulatoryPressure', 10),
    ]

    latencies = []
    for source, target, mag in test_queries:
        start = time.time()
        safety_map_lookup.find_best_scenario(source, target, mag)
        elapsed_ms = (time.time() - start) * 1000
        latencies.append(elapsed_ms)

    return {
        'n_queries': len(latencies),
        'mean_ms': float(f'{statistics.mean(latencies):.2f}'),
        'median_ms': float(f'{statistics.median(latencies):.2f}'),
        'p95_ms': float(f'{sorted(latencies)[int(len(latencies) * 0.95)]:.2f}'),
        'max_ms': float(f'{max(latencies):.2f}'),
        'min_ms': float(f'{min(latencies):.2f}'),
        'all_under_200ms': all(lat < 200 for lat in latencies),
        'individual_ms': [float(f'{lat:.2f}') for lat in latencies],
    }


@app.get('/info')
@limiter.limit(RATE_LIMIT_STRING)
async def system_info(request: Request) -> dict[str, Any]:
    """Return system info including all 3 OPEA component statuses."""
    opea_endpoint = os.environ.get('OPEA_LLM_ENDPOINT', 'not_configured')
    embedding_endpoint = os.environ.get('OPEA_EMBEDDING_ENDPOINT', 'not_configured')
    reranking_endpoint = os.environ.get('OPEA_RERANKING_ENDPOINT', 'not_configured')
    llm_model = os.environ.get('LLM_MODEL_ID', 'Intel/neural-chat-7b-v3-3')

    return {
        'engine': 'CDIE v4 — Causal Decision Intelligence Engine',
        'domain': 'Telecom SIM Box Fraud Detection',
        'version': '4.1.0',
        'opea_components': {
            'llm_textgen': {
                'endpoint': opea_endpoint,
                'model': llm_model,
                'status': 'connected' if opea_endpoint != 'not_configured' else 'offline',
                'provider': explanation_engine.llm_provider,
                'image': 'opea/llm-textgen:latest',
            },
            'tei_embedding': {
                'endpoint': embedding_endpoint,
                'model': 'BAAI/bge-base-en-v1.5',
                'status': 'connected' if embedding_endpoint != 'not_configured' else 'offline',
                'provider': explanation_engine.embedding_provider,
                'image': 'ghcr.io/huggingface/text-embeddings-inference:cpu-latest',
            },
            'tei_reranking': {
                'endpoint': reranking_endpoint,
                'model': 'BAAI/bge-reranker-base',
                'status': 'connected' if reranking_endpoint != 'not_configured' else 'offline',
                'provider': explanation_engine.reranking_provider,
                'image': 'ghcr.io/huggingface/text-embeddings-inference:cpu-latest',
            },
        },
        'intel_optimization': {
            'DNNL_MAX_CPU_ISA': os.environ.get('DNNL_MAX_CPU_ISA', 'not_set'),
            'KMP_AFFINITY': os.environ.get('KMP_AFFINITY', 'not_set'),
            'KMP_BLOCKTIME': os.environ.get('KMP_BLOCKTIME', 'not_set'),
            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', 'not_set'),
        },
        'capabilities': [
            'causal_discovery_gfci',
            'temporal_pcmci_plus',
            'refutation_3test',
            'doubly_robust_dml',
            'conformal_prediction',
            'prescriptive_engine',
            'hitl_edge_rejection',
            'opea_llm_briefing',
            'opea_tei_embedding',
            'opea_tei_reranking',
        ],
    }


@app.get('/benchmark/hardware')
@limiter.limit(RATE_LIMIT_STRING)
async def benchmark_hardware(request: Request) -> dict[str, Any]:
    """Report Intel hardware capabilities and optimization status."""
    import platform

    cpu_info = {
        'platform': platform.processor() or platform.machine(),
        'architecture': platform.architecture()[0],
        'python_version': platform.python_version(),
    }

    # Detect Intel CPU features
    from cdie.utils.shell import detect_cpu_feature_linux, detect_cpu_name_windows

    intel_features = {
        'DNNL_MAX_CPU_ISA': os.environ.get('DNNL_MAX_CPU_ISA', 'not_set'),
        'KMP_AFFINITY': os.environ.get('KMP_AFFINITY', 'not_set'),
        'KMP_BLOCKTIME': os.environ.get('KMP_BLOCKTIME', 'not_set'),
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', 'not_set'),
    }

    # Check if AMX/AVX-512 instructions are available
    avx512_available = detect_cpu_feature_linux('avx512')
    amx_available = detect_cpu_feature_linux('amx')

    if platform.system() == 'Windows':
        cpu_name = detect_cpu_name_windows().lower()
        avx512_available = any(gen in cpu_name for gen in ['xeon', 'sapphire', 'emerald', 'granite'])

    return {
        'cpu': cpu_info,
        'intel_features': intel_features,
        'hardware_detection': {
            'avx512_available': avx512_available,
            'amx_available': amx_available,
        },
        'optimization_active': intel_features['DNNL_MAX_CPU_ISA'] != 'not_set',
        'total_opea_components': 3,
        'opea_components_list': [
            'opea/llm-textgen (Intel/neural-chat-7b-v3-3 via TGI)',
            'TEI Embedding (BAAI/bge-base-en-v1.5, Intel-optimized)',
            'TEI Reranking (BAAI/bge-reranker-base, Intel-optimized)',
        ],
    }


@app.get('/benchmark/embedding')
@limiter.limit(RATE_LIMIT_STRING)
async def benchmark_embedding(request: Request) -> dict[str, Any]:
    """Benchmark OPEA TEI Embedding performance."""
    test_queries = [
        'What happens if SIM box fraud attempts increase?',
        'Impact of tightening fraud policy on revenue leakage',
        'Temporal lag between CDR volume and network load',
    ]

    results = []
    for query in test_queries:
        start = time.time()
        analogies = explanation_engine.retrieve_analogies(query, top_k=3)
        elapsed_ms = (time.time() - start) * 1000
        q_str: str = query
        results.append(
            {
                'query': q_str[:50] + '...',
                'latency_ms': float(f'{elapsed_ms:.2f}'),
                'retrieval_method': analogies[0].get('retrieval_method', 'unknown') if analogies else 'none',
                'top_match_similarity': analogies[0].get('similarity', 0) if analogies else 0,
            }
        )

    return {
        'embedding_provider': explanation_engine.embedding_provider,
        'reranking_provider': explanation_engine.reranking_provider,
        'benchmarks': results,
        'mean_latency_ms': float(f'{sum(float(r["latency_ms"]) for r in results) / len(results):.2f}'),
    }


@app.get('/benchmark/performance')
async def benchmark_performance() -> dict[str, Any]:
    """CDIE-specific performance metrics for hackathon evaluation."""
    import statistics

    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024

    # 1. Safety Map lookup time
    lookup_times: list[float] = []
    test_pairs = [
        ('SIMBoxFraudAttempts', 'ARPUImpact', 30),
        ('FraudPolicyStrictness', 'SIMFraudDetectionRate', 20),
        ('RevenueLeakageVolume', 'CashFlowRisk', 15),
        ('CallDataRecordVolume', 'NetworkLoad', 25),
    ]

    if safety_map_lookup.is_loaded():
        for source, target, mag in test_pairs:
            start = time.perf_counter()
            safety_map_lookup.find_best_scenario(source, target, mag)
            lookup_times.append((time.perf_counter() - start) * 1000)

    # 2. End-to-end query latency (lookup + RAG retrieval)
    e2e_times: list[float] = []
    test_queries_text = [
        'What happens if SIM box fraud increases by 30%?',
        'Impact of tightening fraud policy on revenue?',
    ]
    for q in test_queries_text:
        start = time.perf_counter()
        classification = classify_query(q)
        src = classification['source']
        tgt = classification['target']
        mag_val = classification['magnitude']
        if src and safety_map_lookup.is_loaded():
            safety_map_lookup.find_best_scenario(src, tgt, mag_val)
            explanation_engine.retrieve_analogies(q, top_k=3)
        elapsed = (time.perf_counter() - start) * 1000
        e2e_times.append(elapsed)

    # 3. Queries per second estimate
    if e2e_times:
        avg_e2e = statistics.mean(e2e_times)
        qps = 1000.0 / avg_e2e if avg_e2e > 0 else 0
    else:
        avg_e2e = 0
        qps = 0

    # 4. OPEA component status
    opea_status = {
        'llm_textgen': explanation_engine.llm_provider or 'offline',
        'tei_embedding': explanation_engine.embedding_provider or 'offline',
        'tei_reranking': explanation_engine.reranking_provider or 'offline',
    }

    # 5. Intel CPU optimization status
    cpu_flags = {
        'DNNL_MAX_CPU_ISA': os.environ.get('DNNL_MAX_CPU_ISA', 'NOT SET'),
        'KMP_AFFINITY': os.environ.get('KMP_AFFINITY', 'NOT SET'),
        'KMP_BLOCKTIME': os.environ.get('KMP_BLOCKTIME', 'NOT SET'),
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', 'NOT SET'),
    }
    cpu_optimized = cpu_flags['DNNL_MAX_CPU_ISA'] != 'NOT SET'

    return {
        'safety_map_lookup': {
            'mean_ms': float(f'{statistics.mean(lookup_times):.2f}') if lookup_times else None,
            'median_ms': float(f'{statistics.median(lookup_times):.2f}') if lookup_times else None,
            'max_ms': float(f'{max(lookup_times):.2f}') if lookup_times else None,
            'n_queries': len(lookup_times),
        },
        'end_to_end': {
            'mean_ms': float(f'{avg_e2e:.2f}'),
            'queries_per_second': float(f'{qps:.1f}'),
            'n_queries': len(e2e_times),
        },
        'memory': {
            'rss_mb': float(f'{memory_mb:.1f}'),
            'rss_gb': float(f'{memory_mb / 1024:.2f}'),
        },
        'opea_components': opea_status,
        'cpu_optimization': {
            'flags': cpu_flags,
            'optimized': cpu_optimized,
        },
    }


@app.post('/api/extract-priors')
async def extract_priors(file: UploadFile) -> dict[str, Any]:
    """
    Upload a telecom guideline (PDF or TXT) to extract causal priors via OPEA TextGen.
    Extracted priors are saved to data/extracted_priors.json and will be
    automatically consumed by the next pipeline run (GFCI discovery step).
    """
    import io

    from cdie.pipeline.prior_extractor import PriorExtractor

    filename = str(file.filename or 'unknown')
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

    if ext not in ('pdf', 'txt', 'md'):
        raise HTTPException(
            400,
            f"Unsupported file type '.{ext}'. Please upload a .pdf, .txt, or .md file.",
        )

    try:
        contents = await file.read()
        extractor = PriorExtractor()

        if ext == 'pdf':
            text = extractor.extract_text_from_pdf(io.BytesIO(contents))
        else:
            text = contents.decode('utf-8', errors='replace')

        if not text.strip():
            raise HTTPException(422, 'Uploaded file contains no extractable text.')

        priors = extractor.extract_from_text(text)

        # Persist to disk so the pipeline can consume them
        priors_path = DATA_DIR / 'extracted_priors.json'
        with open(priors_path, 'w') as f:
            json.dump(priors, f, indent=2)

        return {
            'status': 'success',
            'filename': filename,
            'text_length': len(text),
            'priors_extracted': len(priors),
            'priors': priors,
            'persisted_to': str(priors_path),
            'message': (
                f"Extracted {len(priors)} causal priors from '{filename}'. "
                'They will be injected into the next GFCI discovery run.'
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f'Prior extraction failed: {e}') from e


async def _ingest_uploaded_file(background_tasks: BackgroundTasks, file: UploadFile) -> dict[str, Any]:
    """Ingest new observational data and merge into the master store."""
    import io

    from cdie.pipeline.catl import run_catl
    from cdie.config import VARIABLE_NAMES
    from cdie.pipeline.data_ingestion import DataIngestionRouter

    try:
        contents = await file.read()
        file_obj = io.BytesIO(contents)
        df, warnings = DataIngestionRouter.ingest(file_obj, str(file.filename))

        # 1. Delta-CATL check (on new data only)
        catl_results = run_catl(df, VARIABLE_NAMES)
        summary = catl_results.get('_summary', {})

        if summary.get('overall') == 'ADVERSARIAL_SUSPECTED':
            return {
                'status': 'rejected',
                'reason': 'ADVERSARIAL_SUSPECTED',
                'filename': file.filename,
                'catl_report': catl_results,
            }

        # 2. Merge into Master Store
        merged_df, merge_warnings = store_manager.merge_data(df)
        warnings.extend(merge_warnings)

        # 3. Trigger Pipeline on Cumulative Data
        from cdie.pipeline.run_pipeline import run_pipeline

        background_tasks.add_task(run_pipeline, merged_df)

        return {
            'status': 'accepted',
            'filename': file.filename,
            'rows_delta': len(df),
            'rows_cumulative': len(merged_df),
            'warnings': warnings,
            'message': 'Data merged successfully. Pipeline updating Safety Map in background.',
        }
    except Exception as e:
        raise HTTPException(500, f'Ingestion failed: {e}') from e


@app.post('/ingest')
async def ingest_data(background_tasks: BackgroundTasks, file: UploadFile) -> dict[str, Any]:
    """Ingest new observational data and merge into master store."""
    return await _ingest_uploaded_file(background_tasks, file)


@app.post('/api/ingest')
async def ingest_data_compat(background_tasks: BackgroundTasks, file: UploadFile) -> dict[str, Any]:
    """Backward-compatible alias for older UIs that post to /api/ingest."""
    return await _ingest_uploaded_file(background_tasks, file)


@app.post('/ingest/sql')
async def ingest_sql(background_tasks: BackgroundTasks, uri: str, query: str) -> dict[str, Any]:
    """Ingest data from a SQL database URI and merge."""
    from cdie.pipeline.data_ingestion import DataIngestionRouter

    try:
        df, warnings = DataIngestionRouter.ingest_from_sql(uri, query)
        merged_df, merge_warnings = store_manager.merge_data(df)

        from cdie.pipeline.run_pipeline import run_pipeline

        background_tasks.add_task(run_pipeline, merged_df)

        return {
            'status': 'accepted',
            'rows_ingested': len(df),
            'rows_total': len(merged_df),
            'warnings': warnings + merge_warnings,
        }
    except Exception as e:
        raise HTTPException(500, f'SQL Ingestion failed: {e}') from e


# ═══════════════════════════════════════════════════════
# Feature 2: Knowledge Brain APIs
# ═══════════════════════════════════════════════════════


@app.get('/api/knowledge')
async def get_knowledge() -> dict[str, Any]:
    """Return all active priors and pending conflicts from the Knowledge Store."""
    from cdie.pipeline.knowledge_store import KnowledgeStore

    store = KnowledgeStore()
    return {
        'priors': store.get_active_priors(),
        'pending_conflicts': store.get_pending_conflicts(),
        'total_priors': len(store.get_active_priors()),
    }


class AdjudicateRequest(BaseModel):
    conflict_id: int
    action: str  # accept_prior, reject_prior, defer
    reason: str = ''


@app.post('/api/knowledge/adjudicate')
async def adjudicate_conflict(req: AdjudicateRequest) -> dict[str, Any]:
    """Resolve a knowledge conflict via HITL adjudication."""
    from cdie.pipeline.knowledge_store import KnowledgeStore

    if req.action not in ('accept_prior', 'reject_prior', 'defer'):
        raise HTTPException(400, 'Action must be: accept_prior, reject_prior, or defer')
    store = KnowledgeStore()
    result = store.adjudicate_conflict(req.conflict_id, req.action, req.reason)
    if not result.get('success'):
        raise HTTPException(404, result.get('message', 'Conflict not found'))
    return result


# ═══════════════════════════════════════════════════════
# Feature 3: Causal Drift Dashboard APIs
# ═══════════════════════════════════════════════════════

drift_analyzer = DriftAnalyzer()


@app.get('/api/drift/timeline')
async def get_drift_timeline() -> dict[str, Any]:
    """Return list of all historical DAG snapshots."""
    timeline = drift_analyzer.get_timeline()
    return {'timeline': timeline, 'total_snapshots': len(timeline)}


@app.get('/api/drift/compare')
async def compare_drift(id_from: int, id_to: int) -> dict[str, Any]:
    """Compare two DAG snapshots for structural and ATE drift."""
    result = drift_analyzer.compare_snapshots(id_from, id_to)
    if 'error' in result:
        raise HTTPException(404, result['error'])
    return result


class DriftCompareRequest(BaseModel):
    id_from: int
    id_to: int


@app.post('/api/drift/compare')
async def compare_drift_post(req: DriftCompareRequest) -> dict[str, Any]:
    """Backward-compatible POST variant for older dashboards."""
    return await compare_drift(req.id_from, req.id_to)


@app.get('/api/drift/edge-history')
async def get_edge_drift(source: str, target: str) -> list[dict[str, Any]]:
    """Get the ATE history of a specific edge across all snapshots."""
    return drift_analyzer.get_edge_history(source, target)


# ═══════════════════════════════════════════════════════
# Feature 4: Backtesting Engine APIs
# ═══════════════════════════════════════════════════════


class BacktestRequest(BaseModel):
    source: str | None = None
    target: str | None = None
    intervention: str | None = None
    outcome: str | None = None
    magnitude: float = 0.2
    start_index: int = 0
    end_index: int | None = None
    start_date: str | None = None
    end_date: str | None = None


def _resolve_backtest_window(
    data: pd.DataFrame,
    start_index: int,
    end_index: int | None,
    start_date: str | None,
    end_date: str | None,
) -> tuple[int, int | None]:
    """Support both index-based and date-based windows for backtesting."""
    if not start_date and not end_date:
        return start_index, end_index

    date_col = next(
        (column for column in ['timestamp', 'Timestamp', 'date', 'Date', 'event_date'] if column in data.columns),
        None,
    )
    if not date_col:
        return start_index, end_index

    date_series = pd.to_datetime(data[date_col], errors='coerce')
    resolved_start = start_index
    resolved_end = end_index

    if start_date:
        start_ts = pd.to_datetime(start_date, errors='coerce')
        if pd.notna(start_ts):
            matching = data.index[date_series >= start_ts]
            if len(matching) > 0:
                resolved_start = int(matching[0])

    if end_date:
        end_ts = pd.to_datetime(end_date, errors='coerce')
        if pd.notna(end_ts):
            matching = data.index[date_series <= end_ts]
            if len(matching) > 0:
                resolved_end = int(matching[-1]) + 1

    return resolved_start, resolved_end


@app.post('/api/backtest')
async def run_backtest(req: BacktestRequest) -> dict[str, Any]:
    """Backtest a counterfactual intervention against historical data."""
    from cdie.pipeline.backtester import Backtester
    from cdie.pipeline.data_generator import generate_scm_data

    # Use currently available data or generate synthetic
    data_path = DATA_DIR / 'current_data.csv'
    data = pd.read_csv(data_path) if data_path.exists() else generate_scm_data()

    source_name = _resolve_variable_name(req.source or req.intervention)
    target_name = _resolve_variable_name(req.target or req.outcome)

    if source_name and not target_name:
        target_name = DEFAULT_TARGETS.get(source_name, 'ARPUImpact')

    if not source_name or not target_name:
        raise HTTPException(
            400,
            'Backtest requires a valid source/intervention and target/outcome variable.',
        )

    start_index, end_index = _resolve_backtest_window(
        data,
        req.start_index,
        req.end_index,
        req.start_date,
        req.end_date,
    )

    bt = Backtester(data)
    result = bt.backtest(
        source=source_name,
        target=target_name,
        magnitude=_normalize_relative_fraction(req.magnitude),
        start_index=start_index,
        end_index=end_index,
    )

    if 'error' in result:
        raise HTTPException(400, result['error'])

    return result


class BatchBacktestRequest(BaseModel):
    source: str
    magnitude: float = 0.2
    targets: list[str] | None = None


@app.post('/api/backtest/batch')
@limiter.limit(RATE_LIMIT_STRING)
async def run_batch_backtest(request: Request, req: BatchBacktestRequest) -> dict[str, Any]:
    """Backtest one intervention across multiple target outcomes."""
    from cdie.pipeline.backtester import Backtester
    from cdie.pipeline.data_generator import generate_scm_data

    data_path = DATA_DIR / 'current_data.csv'
    data = pd.read_csv(data_path) if data_path.exists() else generate_scm_data()

    bt = Backtester(data)
    results = bt.batch_backtest(
        source=req.source,
        targets=req.targets,
        magnitude=_normalize_relative_fraction(req.magnitude),
    )
    return {
        'source': req.source,
        'magnitude': _normalize_relative_fraction(req.magnitude),
        'results': results,
    }


# ═══════════════════════════════════════════════════════
# Feature 5: Federated Causal Learning APIs
# ═══════════════════════════════════════════════════════


@app.get('/api/federation/export')
@limiter.limit(RATE_LIMIT_STRING)
async def export_pag_endpoint(request: Request) -> dict[str, Any]:
    """Export this operator's PAG (causal structure only, no raw data)."""
    from cdie.pipeline.federation import PAGSerializer

    if not safety_map_lookup.is_loaded():
        raise HTTPException(503, 'Safety Map not loaded. Run the pipeline first.')

    graph_data = safety_map_lookup.get_graph()
    edges = [(e['from'], e['to']) for e in graph_data.get('edges', [])]
    ate_map = {}  # Build from scenarios
    with contextlib.suppress(Exception):
        import sqlite3

        if safety_map_lookup.db_path:
            with sqlite3.connect(str(safety_map_lookup.db_path)) as conn:
                rows = conn.execute('SELECT source, target, data_payload FROM scenarios').fetchall()
                for src, tgt, payload in rows:
                    data = json.loads(payload)
                    ate_map[f'{src}->{tgt}'] = data.get('effect', {}).get('point_estimate', 0)

    operator_id = os.environ.get('CDIE_OPERATOR_ID', 'operator_default')
    pag = PAGSerializer.export_pag(edges, ate_map, operator_id=operator_id)
    return pag


@app.post('/api/federation/import')
@limiter.limit(RATE_LIMIT_STRING)
async def import_pag_endpoint(request: Request, pag: dict[str, Any]) -> dict[str, Any]:
    """Import another operator's PAG and detect conflicts with local structure."""
    from cdie.pipeline.federation import PAGSerializer
    from cdie.pipeline.knowledge_store import KnowledgeStore

    valid, msg = PAGSerializer.validate_pag(pag)
    if not valid:
        raise HTTPException(400, f'Invalid PAG: {msg}')

    # Store imported edges as priors
    store = KnowledgeStore()
    priors = [
        {
            'source': e['source'],
            'target': e['target'],
            'confidence': e.get('confidence', 0.5),
        }
        for e in pag.get('edges', [])
    ]
    result = store.add_priors(
        priors,
        origin=f'federation:{pag.get("operator_id", "unknown")}',
        source_document=f'PAG import from {pag.get("operator_id", "unknown")}',
    )

    # Detect conflicts
    local_graph = safety_map_lookup.get_graph()
    local_edges = [(e['source'], e['target']) for e in local_graph.get('edges', [])]
    conflicts = store.detect_conflicts(local_edges)

    return {
        'status': 'imported',
        'operator': pag.get('operator_id'),
        'edges_imported': len(pag.get('edges', [])),
        'storage_result': result,
        'conflicts_detected': len(conflicts),
        'conflicts': conflicts,
    }


@app.post('/api/federation/aggregate')
@limiter.limit(RATE_LIMIT_STRING)
async def aggregate_pags_endpoint(request: Request, pags: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multiple operator PAGs via weighted edge voting."""
    from cdie.pipeline.federation import FederatedAggregator, PAGSerializer

    for i, pag in enumerate(pags):
        valid, msg = PAGSerializer.validate_pag(pag)
        if not valid:
            raise HTTPException(400, f'Invalid PAG at index {i}: {msg}')

    result = FederatedAggregator.aggregate_pags(pags)
    if 'error' in result:
        raise HTTPException(400, result['error'])
    return result


# ═══════════════════════════════════════════════════════
# Feature 6: Heterogeneous Treatment Effects (HTE/CATE)
# ═══════════════════════════════════════════════════════


@app.get('/hte/report')
@limiter.limit(RATE_LIMIT_STRING)
async def get_hte_report(request: Request) -> Any:
    """Retrieve HTE CATE segment report."""
    report_path = DATA_DIR / 'hte_report.json'
    if not report_path.exists():
        raise HTTPException(404, 'HTE report not found. Run the offline pipeline.')
    try:
        report_data = json.loads(report_path.read_text(encoding='utf-8'))
        return report_data
    except Exception as e:
        raise HTTPException(500, f'Error reading HTE report: {e}') from e


@app.get('/hte/heatmap')
@limiter.limit(RATE_LIMIT_STRING)
async def get_hte_heatmap(request: Request) -> Any:
    """Retrieve HTE segment heatmap image."""
    image_path = DATA_DIR / 'hte_segment_heatmap.png'
    if not image_path.exists():
        raise HTTPException(404, 'HTE heatmap not found. Run the offline pipeline.')
    return FileResponse(image_path, media_type='image/png')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
