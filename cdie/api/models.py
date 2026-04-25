"""
CDIE v4 — Pydantic Request/Response Models
API contracts per SRS §3.4.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description='User query about enterprise data')
    scenario: str | None = Field(None, description='Optional specific scenario ID to override discovery')
    magnitude: float | None = Field(None, ge=-100, le=100, description='Optional magnitude override (percentage)')

    @field_validator('scenario')
    @classmethod
    def validate_scenario_id(cls, v: str | None) -> str | None:
        if v is None:
            return v
        parts = v.split('__')
        if len(parts) != 3:
            raise ValueError("Scenario ID must be in format 'source__target__magnitude_key'")

        from cdie.config import MAGNITUDE_LEVELS, VARIABLE_NAMES
        source, target, mag_key = parts
        if source not in VARIABLE_NAMES:
            raise ValueError(f"Invalid source variable: {source}")
        if target not in VARIABLE_NAMES:
            raise ValueError(f"Invalid target variable: {target}")
        if mag_key not in MAGNITUDE_LEVELS:
            raise ValueError(f"Invalid magnitude key: {mag_key}")
        return v


class BatchQueryRequest(BaseModel):
    queries: list[QueryRequest] = Field(..., min_length=1, description='List of causal queries')


class EffectResult(BaseModel):
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95
    ate_used: float = 0.0
    intervention_amount: float = 0.0


class RefutationStatus(BaseModel):
    placebo: str = 'NOT_TESTED'
    confounder: str = 'NOT_TESTED'
    subset: str = 'NOT_TESTED'


class CATESegment(BaseModel):
    segment: str
    ate: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    n_samples: int = 0
    risk_level: str = 'unknown'


class QueryResponse(BaseModel):
    query_type: str
    query_id: str
    scenario_id: str | None = None
    source: str = ''
    target: str = ''
    magnitude: str = ''
    effect: EffectResult | None = None
    causal_path: str = ''
    refutation_status: RefutationStatus | None = None
    ks_warning: bool = False
    ks_statistic: float = 0.0
    kl_divergence: float = 0.0
    drift_detected: bool = False
    explanation: str = ''
    historical_analogies: list[str] = Field(default_factory=list)
    cate_segments: list[CATESegment] = Field(default_factory=list)
    feature_importance: dict[str, float] = Field(default_factory=dict)
    confidence_label: str = 'ESTIMATED'
    match_type: str = 'unknown'
    evidence_tier: str = 'unknown'
    trust_message: str = ''
    used_fallback: bool = False
    suggested_queries: list[str] = Field(default_factory=list)
    available_variables: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    safety_map_hash: str = ''
    last_computed: str = ''
    ks_status: str = 'OK'
    memory_mb: float = 0.0
    n_scenarios: int = 0
    storage_backend: str = 'unloaded'
    opea_status: dict[str, str] = Field(default_factory=dict)


class GraphNode(BaseModel):
    id: str
    label: str
    type: str = 'variable'


class GraphEdge(BaseModel):
    source: str = Field(..., alias='from')
    target: str = Field(..., alias='to')
    edge_type: str = 'directed'
    weight: float = 0.0
    refutation_status: str = 'UNKNOWN'

    model_config = ConfigDict(populate_by_name=True)


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class BenchmarkMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    shd: int
    status: str = 'COMPLETE'


class BenchmarkResponse(BaseModel):
    sachs: BenchmarkMetrics
    alarm: BenchmarkMetrics
    own_scm: BenchmarkMetrics | None = None


class CATLBadge(BaseModel):
    test: str
    status: str
    tooltip: str
    details: dict[str, Any] = Field(default_factory=dict)


class CATLResponse(BaseModel):
    faithfulness: CATLBadge
    sufficiency: CATLBadge
    stationarity: CATLBadge
    acyclicity: CATLBadge


class ExpertCorrectionRequest(BaseModel):
    from_node: str
    to_node: str
    action: str = Field(..., pattern='^(add|remove|reverse)$')

    @field_validator('from_node', 'to_node')
    @classmethod
    def validate_nodes(cls, v: str) -> str:
        from cdie.config import VARIABLE_NAMES
        if v not in VARIABLE_NAMES:
            raise ValueError(f"Invalid variable name: {v}")
        return v



class ExpertCorrectionResponse(BaseModel):
    success: bool
    message: str


class PrescribeRequest(BaseModel):
    target: str = Field(..., description='The variable to optimize (maximize/minimize)')
    maximize: bool = True
    limit: int = 3

    @field_validator('target')
    @classmethod
    def validate_target(cls, v: str) -> str:
        from cdie.config import VARIABLE_NAMES
        if v not in VARIABLE_NAMES:
            # Try to resolve via aliases if needed, but strict for now
            raise ValueError(f"Invalid target variable: {v}. Must be one of {VARIABLE_NAMES}")
        return v



class PrescribeResponse(BaseModel):
    target: str
    goal: str
    recommendations: list[QueryResponse]
    metamodel_score: float = 0.0


class BatchPrescribeRequest(BaseModel):
    requests: list[PrescribeRequest] = Field(..., min_length=1, description='List of prescription requests')


class VariableInfo(BaseModel):
    name: str
    label: str
    description: str
    aliases: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class VariableCatalogResponse(BaseModel):
    variables: list[VariableInfo] = Field(default_factory=list)
