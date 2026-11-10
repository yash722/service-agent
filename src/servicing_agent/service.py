from __future__ import annotations

import logging
import time
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
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                reasoning_effort=settings.llm_reasoning_effort,
                history_record_limit=settings.history_llm_record_limit,
                fallback_history_record_limit=(
                    settings.history_answer_record_limit
                ),
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
    return DeterministicGroundedGenerator(
        settings.history_answer_record_limit
    )


def build_history_adapter(settings: Settings) -> HistoryAdapter:
    if settings.history_backend == "mock":
        return MockHistoryAdapter()
    if settings.history_backend == "snowflake":
        return SnowflakeHistoryAdapter(SnowflakeHistoryConfig.from_settings(settings))
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
            top_k=settings.retrieval_top_k,
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
        started = time.perf_counter()
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
                generation_debug=result.get("generation_debug"),
            )

        response.total_latency_ms = round(
            (time.perf_counter() - started) * 1000, 3
        )
        log_event(
            logger,
            "chat.completed",
            execution_id=execution_id,
            intent=response.intent,
            step_count=len(response.steps),
            evidence_status=response.evidence_status,
            generation_mode=response.generation_mode,
            total_latency_ms=response.total_latency_ms,
            retrieval_latency_ms=(
                response.retrieval_debug.latency_ms
                if response.retrieval_debug
                else None
            ),
            history_latency_ms=(
                response.history_debug.latency_ms
                if response.history_debug
                else None
            ),
            generation_latency_ms=(
                response.generation_debug.latency_ms
                if response.generation_debug
                else None
            ),
            generation_fallback_reason=(
                response.generation_debug.fallback_reason
                if response.generation_debug
                else None
            ),
            generation_finish_reason=(
                response.generation_debug.finish_reason
                if response.generation_debug
                else None
            ),
            generation_output_tokens=(
                response.generation_debug.output_tokens
                if response.generation_debug
                else None
            ),
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
