from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .auth import DemoUserContext
from .generation import GroundedAnswerGenerator
from .logging_config import log_event
from .retrieval import RetrievalResult
from .schemas import RetrievalDebug, SourceCitation
from .tools import (
    CRMAuthorizationError,
    EntitlementTool,
    EvidenceRetrievalTool,
    MockCRMTool,
    TicketNotFoundError,
)

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict, total=False):
    message: str
    execution_id: str
    user_context: DemoUserContext
    case_id: str | None
    topic: str
    ticket: dict[str, Any] | None
    entitlements: list[dict[str, Any]]
    retrieval: RetrievalResult
    error: str | None
    reply: str
    citations: list[SourceCitation]
    generation_mode: str
    evidence_status: str
    warnings: list[str]
    retrieval_debug: RetrievalDebug | None
    steps: Annotated[list[str], operator.add]
    trace: Annotated[list[str], operator.add]


def build_complex_workflow(
    crm_tool: MockCRMTool,
    entitlement_tool: EntitlementTool,
    evidence_tool: EvidenceRetrievalTool,
    answer_generator: GroundedAnswerGenerator,
):
    def triage(state: WorkflowState) -> dict[str, Any]:
        topic = "case_follow_up" if state.get("case_id") else "general_servicing"
        log_event(
            logger,
            "workflow.triage",
            execution_id=state["execution_id"],
            topic=topic,
        )
        return {
            "topic": topic,
            "steps": ["workflow:triage"],
            "trace": [f"workflow.triage:{topic}"],
        }

    def load_case_context(state: WorkflowState) -> dict[str, Any]:
        case_id = state.get("case_id")
        if case_id is None:
            return {
                "ticket": None,
                "error": None,
                "steps": ["mock_crm:no_case_reference"],
                "trace": ["crm.case_context:none"],
            }
        try:
            ticket = crm_tool.get_ticket(case_id, state["user_context"])
        except CRMAuthorizationError:
            result = {"ticket": None, "error": "unauthorized"}
        except TicketNotFoundError:
            result = {"ticket": None, "error": "not_found"}
        else:
            result = {"ticket": ticket, "error": None}

        outcome = result["error"] or "authorized"
        log_event(
            logger,
            "workflow.crm_lookup",
            execution_id=state["execution_id"],
            ticket_id=case_id,
            outcome=outcome,
        )
        return {
            **result,
            "steps": ["mock_crm:get_ticket"],
            "trace": [f"crm.get_ticket:{outcome}"],
        }

    def gather_grounded_evidence(state: WorkflowState) -> dict[str, Any]:
        if state.get("error"):
            return {
                "entitlements": [],
                "retrieval": RetrievalResult((), "insufficient", ()),
                "steps": ["mock_knowledge:search", "mock_policy:guidance"],
                "trace": ["evidence.skipped:case_context_error"],
            }

        ticket = state.get("ticket")
        try:
            entitlements = (
                entitlement_tool.get_active_entitlements(
                    ticket["customer_id"], state["user_context"]
                )
                if ticket
                else []
            )
            retrieval = evidence_tool.search(
                state["message"],
                user_context=state["user_context"],
                ticket=ticket,
                entitlements=entitlements,
            )
        except CRMAuthorizationError:
            return {
                "entitlements": [],
                "retrieval": RetrievalResult((), "insufficient", ()),
                "error": "unauthorized",
                "steps": ["mock_knowledge:search", "mock_policy:guidance"],
                "trace": ["evidence.authorization:denied"],
            }

        source_ids = [item.document.source_id for item in retrieval.evidence]
        diagnostics = retrieval.diagnostics
        log_event(
            logger,
            "workflow.evidence_retrieved",
            execution_id=state["execution_id"],
            evidence_status=retrieval.status,
            source_ids=source_ids,
            entitlement_count=len(entitlements),
            retrieval_mode=diagnostics.effective_mode,
            requested_retrieval_mode=diagnostics.requested_mode,
            retrieval_latency_ms=diagnostics.latency_ms,
            rankings=diagnostics.rankings,
            selected_document_ids=diagnostics.selected_document_ids,
            fallback_reason=diagnostics.fallback_reason,
        )
        return {
            "entitlements": entitlements,
            "retrieval": retrieval,
            # Retained as legacy step values for Milestone 1 API compatibility.
            "steps": ["mock_knowledge:search", "mock_policy:guidance"],
            "trace": [
                f"entitlements.read:{len(entitlements)}",
                (
                    f"retrieval.{diagnostics.requested_mode}"
                    f"->{diagnostics.effective_mode}:{retrieval.status}:"
                    f"{len(retrieval.evidence)}:{diagnostics.latency_ms:.3f}ms"
                ),
                *(
                    [f"retrieval.fallback:{diagnostics.fallback_reason}"]
                    if diagnostics.fallback_reason
                    else []
                ),
            ],
        }

    def compose_response(state: WorkflowState) -> dict[str, Any]:
        case_id = state.get("case_id")
        if state.get("error") == "unauthorized":
            result: dict[str, Any] = {
                "reply": (
                    f"You are not authorized to view {case_id} with the selected "
                    "demo-user context."
                ),
                "citations": [],
                "generation_mode": "deterministic",
                "evidence_status": "insufficient",
                "warnings": ["Protected evidence was not retrieved."],
                "trace": ["generation.skipped:unauthorized"],
                "retrieval_debug": None,
            }
        elif state.get("error") == "not_found":
            result = {
                "reply": f"I could not find {case_id} in the local demo CRM.",
                "citations": [],
                "generation_mode": "deterministic",
                "evidence_status": "insufficient",
                "warnings": ["No authoritative case record was found."],
                "trace": ["generation.skipped:case_not_found"],
                "retrieval_debug": None,
            }
        else:
            generated = answer_generator.generate(
                question=state["message"],
                ticket=state.get("ticket"),
                entitlements=state.get("entitlements", []),
                retrieval=state["retrieval"],
            )
            diagnostics = state["retrieval"].diagnostics
            result = {
                "reply": generated.reply,
                "citations": list(generated.citations),
                "generation_mode": generated.mode,
                "evidence_status": state["retrieval"].status,
                "warnings": list(generated.warnings),
                "trace": [f"generation.{generated.mode}:citation_validated"],
                "retrieval_debug": RetrievalDebug(
                    requested_mode=diagnostics.requested_mode,
                    effective_mode=diagnostics.effective_mode,
                    latency_ms=diagnostics.latency_ms,
                    rankings={
                        name: list(document_ids)
                        for name, document_ids in diagnostics.rankings.items()
                    },
                    selected_document_ids=list(
                        diagnostics.selected_document_ids
                    ),
                    fallback_reason=diagnostics.fallback_reason,
                    embedding_cache_status=diagnostics.embedding_cache_status,
                ),
            }
        return {**result, "steps": ["workflow:compose_response"]}

    graph = StateGraph(WorkflowState)
    graph.add_node("triage", triage)
    graph.add_node("load_case_context", load_case_context)
    graph.add_node("gather_grounded_evidence", gather_grounded_evidence)
    graph.add_node("compose_response", compose_response)
    graph.add_edge(START, "triage")
    graph.add_edge("triage", "load_case_context")
    graph.add_edge("load_case_context", "gather_grounded_evidence")
    graph.add_edge("gather_grounded_evidence", "compose_response")
    graph.add_edge("compose_response", END)
    return graph.compile()
