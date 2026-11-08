from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "fde-servicing-agent"
    milestone: int = 2


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2_000)


class SourceCitation(BaseModel):
    source_id: str
    document_version: str
    title: str
    document_type: str
    effective_date: str
    snippet: str
    citation_label: str


class RetrievalDebug(BaseModel):
    requested_mode: str
    effective_mode: str
    latency_ms: float
    rankings: dict[str, list[str]] = Field(default_factory=dict)
    selected_document_ids: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None
    embedding_cache_status: str | None = None


OperationState = Literal["PENDING", "COMPLETED", "FAILED", "UNKNOWN"]
HandoffState = Literal["CREATED", "ACCEPTED", "RESOLVED"]


class NotePreview(BaseModel):
    ticket_id: str
    subject: str
    status: str
    owner: str
    note: str
    expected_ticket_version: int


class NotePrepareRequest(BaseModel):
    ticket_id: str = Field(pattern=r"^CASE-\d+$")
    note: str = Field(min_length=1, max_length=2_000)
    operation_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )


class NoteConfirmationRequest(BaseModel):
    customer_confirmed: bool = False
    simulate_timeout_after_success: bool = False


class OperationResponse(BaseModel):
    operation_id: str
    operation_type: str
    state: OperationState
    ticket_id: str
    message: str
    preview: NotePreview | None = None
    note_id: str | None = None
    error_code: str | None = None
    created_at: str
    updated_at: str


class HandoffCreateRequest(BaseModel):
    ticket_id: str = Field(pattern=r"^CASE-\d+$")
    reason: str = Field(min_length=1, max_length=1_000)
    troubleshooting_history: list[str] = Field(
        default_factory=list, max_length=20
    )
    operation_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )


class HandoffTransitionRequest(BaseModel):
    operation_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )


class HandoffResponse(BaseModel):
    handoff_id: str
    operation_id: str | None
    ticket_id: str
    trigger_type: Literal["USER", "SLA"]
    state: HandoffState
    queue: str
    reason: str
    case_context: dict[str, str]
    source_references: list[str]
    troubleshooting_history: list[str]
    created_by: str
    assigned_to: str | None = None
    sla_deadline: str | None = None
    backup_routed_at: str | None = None
    created_at: str
    accepted_at: str | None = None
    resolved_at: str | None = None
    updated_at: str


class SLAStatusResponse(BaseModel):
    ticket_id: str
    deadline: str | None
    overdue: bool
    action: Literal[
        "not_due", "closed", "created", "already_escalated", "no_deadline"
    ]
    handoff_id: str | None = None
    handoff_state: HandoffState | None = None
    queue: str | None = None


class SLASweepResponse(BaseModel):
    checked: int
    created: int
    already_escalated: int
    backup_routed: int
    as_of: str


class ChatResponse(BaseModel):
    reply: str
    intent: Literal["ticket_status", "complex_request"]
    execution_id: str
    steps: list[str]
    citations: list[SourceCitation] = Field(default_factory=list)
    generation_mode: Literal["not_applicable", "deterministic", "llm"] = (
        "not_applicable"
    )
    evidence_status: Literal[
        "not_applicable", "sufficient", "insufficient", "conflicting"
    ] = "not_applicable"
    warnings: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    retrieval_debug: RetrievalDebug | None = None
    sla: SLAStatusResponse | None = None

