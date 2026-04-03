"""
CDIE v4 — Pydantic Request/Response Models
API contracts per SRS §3.4.
"""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, description="User query about enterprise data"
    )


class EffectResult(BaseModel):
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95
    ate_used: float = 0.0
    intervention_amount: float = 0.0


class RefutationStatus(BaseModel):
    placebo: str = "NOT_TESTED"
    confounder: str = "NOT_TESTED"
    subset: str = "NOT_TESTED"


class CATESegment(BaseModel):
    segment: str
    ate: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    n_samples: int = 0
    risk_level: str = "unknown"


class QueryResponse(BaseModel):
    query_type: str
    query_id: str
    scenario_id: Optional[str] = None
    source: str = ""
    target: str = ""
    magnitude: str = ""
    effect: Optional[EffectResult] = None
    causal_path: str = ""
    refutation_status: Optional[RefutationStatus] = None
    ks_warning: bool = False
    ks_statistic: float = 0.0
    explanation: str = ""
    historical_analogies: list[str] = Field(default_factory=list)
    cate_segments: list[CATESegment] = Field(default_factory=list)
    confidence_label: str = "ESTIMATED"
    match_type: str = "unknown"
    evidence_tier: str = "unknown"
    trust_message: str = ""
    used_fallback: bool = False
    suggested_queries: list[str] = Field(default_factory=list)
    available_variables: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    safety_map_hash: str = ""
    last_computed: str = ""
    ks_status: str = "OK"
    memory_mb: float = 0.0
    n_scenarios: int = 0
    storage_backend: str = "unloaded"


class GraphNode(BaseModel):
    id: str
    label: str
    type: str = "variable"


class GraphEdge(BaseModel):
    source: str = Field(..., alias="from")
    target: str = Field(..., alias="to")
    edge_type: str = "directed"
    weight: float = 0.0
    refutation_status: str = "UNKNOWN"

    model_config = ConfigDict(populate_by_name=True)


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class BenchmarkMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    shd: int
    status: str = "COMPLETE"


class BenchmarkResponse(BaseModel):
    sachs: BenchmarkMetrics
    alarm: BenchmarkMetrics
    own_scm: Optional[BenchmarkMetrics] = None


class CATLBadge(BaseModel):
    test: str
    status: str
    tooltip: str
    details: dict = Field(default_factory=dict)


class CATLResponse(BaseModel):
    faithfulness: CATLBadge
    sufficiency: CATLBadge
    stationarity: CATLBadge
    acyclicity: CATLBadge


class ExpertCorrectionRequest(BaseModel):
    from_node: str
    to_node: str
    action: str = Field(..., pattern="^(add|remove|reverse)$")


class ExpertCorrectionResponse(BaseModel):
    success: bool
    message: str


class PrescribeRequest(BaseModel):
    target: str
    maximize: bool = True
    limit: int = 3


class PrescribeResponse(BaseModel):
    target: str
    prescriptions: list[dict[str, Any]]
    message: str


class VariableInfo(BaseModel):
    name: str
    label: str
    description: str
    aliases: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class VariableCatalogResponse(BaseModel):
    variables: list[VariableInfo] = Field(default_factory=list)
