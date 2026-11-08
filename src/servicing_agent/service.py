from __future__ import annotations

import logging
from datetime import date, timedelta
from uuid import uuid4

from .auth import DemoUserContext
from .database import Database
from .generation import (
    DeterministicGroundedGenerator,
    GroundedAnswerGenerator,
    OpenAICompatibleConfig,
    OpenAICompatibleGroundedGenerator,
)
from .intent_router import extract_ticket_id, route_intent
from .history import (
    HistoryAdapter,
    MockHistoryAdapter,
    SnowflakeHistoryAdapter,
    SnowflakeHistoryConfig,
)
from .logging_config import log_event
from .retrieval import HybridRetrievalService
from .schemas import ChatResponse
from .settings import Settings
from .sla import SLAService
from .tools import (
    CRMAuthorizationError,
    EntitlementTool,
    EvidenceRetrievalTool,
    HistoryTool,
    MockCRMTool,
    TicketNotFoundError,
)
from .workflow import build_complex_workflow

logger = logging.getLogger(__name__)


def build_answer_generator(settings: Settings) -> GroundedAnswerGenerator:
    if settings.llm_provider == "openai_compatible" and settings.llm_api_key:
        return OpenAICompatibleGroundedGenerator(
            OpenAICompatibleConfig(
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                api_base=settings.llm_api_base,
                timeout_seconds=settings.llm_timeout_seconds,
            )
        )
    if settings.llm_provider not in {"none", "openai_compatible"}:
        log_event(
            logger,
            "llm.disabled",
            reason="unsupported_provider",
            provider=settings.llm_provider,
        )
    elif settings.llm_provider == "openai_compatible" and not settings.llm_api_key:
        log_event(logger, "llm.disabled", reason="missing_api_key")
    return DeterministicGroundedGenerator()


def build_history_adapter(settings: Settings) -> HistoryAdapter:
    if settings.history_backend == "mock":
        return MockHistoryAdapter()
    if settings.history_backend == "snowflake":
        return SnowflakeHistoryAdapter(SnowflakeHistoryConfig.from_environment())
    raise ValueError(
        "HISTORY_BACKEND must be either 'mock' or 'snowflake'."
    )


class ChatService:
    def __init__(
        self,
        database: Database,
        *,
        settings: Settings | None = None,
        as_of: date | None = None,
        answer_generator: GroundedAnswerGenerator | None = None,
        retriever: HybridRetrievalService | None = None,
        history_adapter: HistoryAdapter | None = None,
        sla_service: SLAService | None = None,
    ):
        settings = settings or Settings.from_environment()
        as_of = as_of or date.today()
        self.crm_tool = MockCRMTool(database)
        self.entitlement_tool = EntitlementTool(database, as_of)
        self.retrieval_service = retriever or HybridRetrievalService(
            mode=settings.retrieval_mode,
            embedding_model=settings.retrieval_model,
            embedding_cache_dir=settings.retrieval_cache_dir,
            rrf_k=settings.retrieval_rrf_k,
            semantic_min_score=settings.retrieval_semantic_min_score,
            semantic_min_top_score=settings.retrieval_semantic_min_top_score,
        )
        self.evidence_tool = EvidenceRetrievalTool(self.retrieval_service, as_of)
        self.history_tool = HistoryTool(
            history_adapter or build_history_adapter(settings)
        )
        self.sla_service = sla_service or SLAService(
            database,
            self.crm_tool,
            backup_after=timedelta(minutes=settings.sla_backup_after_minutes),
        )
        self.answer_generator = answer_generator or build_answer_generator(settings)
        self.complex_workflow = build_complex_workflow(
            self.crm_tool,
            self.entitlement_tool,
            self.evidence_tool,
            self.history_tool,
            self.answer_generator,
        )

    def chat(self, message: str, user_context: DemoUserContext) -> ChatResponse:
        execution_id = str(uuid4())
        intent = route_intent(message)
        case_id = extract_ticket_id(message)
        log_event(
            logger,
            "chat.received",
            execution_id=execution_id,
            user_id=user_context.user_id,
            intent=intent,
            ticket_id=case_id,
        )

        if intent == "ticket_status":
            response = self._handle_ticket_status(
                execution_id, case_id, user_context
            )
        else:
            result = self.complex_workflow.invoke(
                {
                    "message": message,
                    "execution_id": execution_id,
                    "user_context": user_context,
                    "case_id": case_id,
                    "steps": [],
                    "trace": [],
                }
            )
            response = ChatResponse(
                reply=result["reply"],
                intent="complex_request",
                execution_id=execution_id,
                steps=result["steps"],
                citations=result["citations"],
                generation_mode=result["generation_mode"],
                evidence_status=result["evidence_status"],
                warnings=result["warnings"],
                trace=result["trace"],
                retrieval_debug=result.get("retrieval_debug"),
                history_debug=result.get("history_debug"),
            )

        log_event(
            logger,
            "chat.completed",
            execution_id=execution_id,
            intent=response.intent,
            step_count=len(response.steps),
            evidence_status=response.evidence_status,
            generation_mode=response.generation_mode,
            citation_count=len(response.citations),
            history_backend=(
                response.history_debug.backend if response.history_debug else None
            ),
            history_status=(
                response.history_debug.status if response.history_debug else None
            ),
        )
        return response

    def _handle_ticket_status(
        self,
        execution_id: str,
        case_id: str | None,
        user_context: DemoUserContext,
    ) -> ChatResponse:
        steps = ["intent_router:ticket_status"]
        if case_id is None:
            return ChatResponse(
                reply="Please include a case ID such as CASE-4821.",
                intent="ticket_status",
                execution_id=execution_id,
                steps=steps,
                trace=["intent_router:ticket_status", "crm.skipped:no_case_id"],
            )

        steps.append("mock_crm:get_ticket")
        try:
            ticket = self.crm_tool.get_ticket(case_id, user_context)
        except CRMAuthorizationError:
            outcome = "unauthorized"
            reply = (
                f"You are not authorized to view {case_id} with the selected "
                "demo-user context."
            )
        except TicketNotFoundError:
            outcome = "not_found"
            reply = f"I could not find {case_id} in the local demo CRM."
        else:
            outcome = "authorized"
            sla = self.sla_service.check_ticket(
                ticket["ticket_id"], user_context
            )
            reply = (
                f"{ticket['ticket_id']} is {ticket['status']}. "
                f"Priority: {ticket['priority']}. Owner: {ticket['owner']}. "
                f"Last updated: {ticket['updated_at']}."
            )
            if sla.overdue and sla.handoff_id:
                reply += (
                    f" Its SLA deadline is overdue; deterministic escalation "
                    f"{sla.handoff_id} is {sla.handoff_state} in {sla.queue}."
                )

        log_event(
            logger,
            "chat.crm_lookup",
            execution_id=execution_id,
            ticket_id=case_id,
            outcome=outcome,
        )
        return ChatResponse(
            reply=reply,
            intent="ticket_status",
            execution_id=execution_id,
            steps=steps,
            trace=["intent_router:ticket_status", f"crm.get_ticket:{outcome}"],
            sla=sla if outcome == "authorized" else None,
        )
