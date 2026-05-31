from pydantic import BaseModel, Field
from typing import Literal


class RunRequest(BaseModel):
    input: str = Field(..., description="Task text, code diff, or financial query")
    domain: Literal["code_review", "finance", "general"] = "code_review"
    pipeline: list[str] | None = Field(
        None, description="Override agent pipeline order. Default: analyst→critic→synthesizer"
    )
    mode: Literal["standard", "adversarial"] = Field(
        "standard",
        description=(
            "'standard' uses the cooperative analyst→critic→synthesizer pipeline. "
            "'adversarial' uses the 6-agent planner→actor→critic→validator→refiner→judge loop."
        ),
    )
    max_rounds: int | None = Field(
        None,
        description="Adversarial mode only. Max iteration rounds before forced acceptance (default: 2).",
    )
    session_id: str | None = Field(
        None,
        description="Multi-user mode. Chat session to attach this run to; created if omitted.",
    )


class AgentMessageOut(BaseModel):
    role: str
    content: str
    metadata: dict = {}
    timestamp: str
    message_id: str


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    weight: float
    position: dict[str, float]
    runCount: int = 0
    metadata: dict = {}


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float
    type: str


class GraphStats(BaseModel):
    totalRuns: int
    totalNodes: int
    totalEdges: int
    clusters: int
    concepts: int


class GraphState(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    stats: GraphStats


class AdversarialMeta(BaseModel):
    rounds_completed: int
    max_rounds: int
    judge_verdict: str
    judge_score: float
    judge_rationale: str
    unresolved_issues: list[str]


class RunResponse(BaseModel):
    task_id: str
    domain: str
    pipeline: list[str]
    messages: list[AgentMessageOut]
    scores: dict[str, float] = {}
    model_used: str = ""
    knowledge_graph: GraphState
    adversarial_meta: AdversarialMeta | None = None


class IngestRequest(BaseModel):
    text: str
    source: str = "manual"


class IngestResponse(BaseModel):
    chunks_added: int
    source: str


# --- Multi-user platform schemas ---------------------------------------

class ProviderAddRequest(BaseModel):
    provider: Literal["openrouter", "anthropic", "openai", "google"]
    api_key: str = Field(..., description="The provider API key (stored encrypted, never returned)")
    label: str | None = None
    base_url: str | None = None


class ProviderConnectionOut(BaseModel):
    id: str
    provider: str
    conn_type: str
    label: str | None = None
    last4: str | None = None
    status: str
    last_validated_at: str | None = None
    created_at: str | None = None


class ProviderTestOut(BaseModel):
    ok: bool
    detail: str


class OAuthStartOut(BaseModel):
    auth_url: str
    state: str


class SessionCreateRequest(BaseModel):
    title: str | None = None
    domain: str = "general"


class SessionOut(BaseModel):
    id: str
    title: str | None = None
    domain: str
    created_at: str | None = None
    updated_at: str | None = None


class ChatMessageOut(BaseModel):
    id: str
    role: str
    content: str
    model_used: str | None = None
    scores: dict | None = None
    adversarial_run_id: str | None = None
    created_at: str | None = None


class MemoryHitOut(BaseModel):
    id: str
    source_type: str
    session_id: str | None = None
    domain: str | None = None
    content: str
    similarity: float
    created_at: str | None = None
