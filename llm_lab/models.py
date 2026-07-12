from typing import Any, Literal

from pydantic import BaseModel, Field

VerdictLabel = Literal["pass", "fail", "partial"]


class IntentRequest(BaseModel):
    goal: str = Field(..., description="Free-text task description")
    preferred_model: str | None = None
    preferred_model_2: str | None = None


class Verdict(BaseModel):
    label: VerdictLabel
    reason: str
    score: float | None = None


class CustomMetrics(BaseModel):
    min_output_length: int | None = None
    max_output_length: int | None = None
    min_tokens: int | None = None
    max_tokens: int | None = None
    must_contain: list[str] | None = None
    must_not_contain: list[str] | None = None


class Step(BaseModel):
    action: str
    model: str | None = None
    prompt: str
    output: str | None = None
    verdict: Verdict | None = None


class Plan(BaseModel):
    template_id: str | None
    steps: list[Step]
    metrics: CustomMetrics | None = None


class RunSummary(BaseModel):
    run_id: str
    intent_id: str
    model: str
    steps: int
    total_tokens: int
    total_cost_usd: float
    all_passed: bool


class EventLogEntry(BaseModel):
    id: int
    intent_id: str
    seq: int
    timestamp: str
    action: str
    model: str | None
    detail: str | None
    input_hash: str | None
    output_hash: str | None
    token_usage: str | None
    cost_usd: float | None


class BatchRequest(BaseModel):
    goal: str
    models: list[str] = Field(..., min_length=1, max_length=20)


class TemplateDef(BaseModel):
    template_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    intent_keywords: list[str]
    steps: list[str]
    verifier: str = "structural"
    artifact_schema: str = ""


class PlannerTemplate(BaseModel):
    template_id: str
    intent_keywords: list[str]
    steps: list[str]
    verifier: str
    artifact_schema: str


# ── Response schema models ──────────────────────────────────────────────


class StepResult(BaseModel):
    action: str
    prompt: str
    output: str | None = None
    model: str | None = None
    verdict: dict[str, Any] | None = None
    tokens: int = 0
    cost: float = 0.0


class ModelResult(BaseModel):
    model: str
    plan_template: str | None = None
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    all_passed: bool = False
    steps: list[StepResult] = []


class RunResult(BaseModel):
    run_id: str
    intent_id: str
    goal: str
    model: str
    verifier: str
    steps: int
    plan_template: str | None = None
    steps_detail: list[StepResult]
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    all_passed: bool = False


class CompareResult(BaseModel):
    intent_id: str
    goal: str
    model_a: ModelResult
    model_b: ModelResult
    summary: dict[str, Any]


class BatchResult(BaseModel):
    intent_id: str
    goal: str
    models: list[ModelResult]
    count: int
