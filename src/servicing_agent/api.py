from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status

from .actions import (
    ControlledNoteService,
    IdempotencyConflictError,
    OperationAuthorizationError,
    OperationNotFoundError,
    StaleConfirmationError,
)
from .auth import DemoUserContext, resolve_demo_user
from .database import Database
from .handoffs import CaseHandoffService, InvalidHandoffTransitionError
from .logging_config import configure_logging
from .schemas import (
    ChatRequest,
    ChatResponse,
    HandoffCreateRequest,
    HandoffResponse,
    HandoffTransitionRequest,
    HealthResponse,
    NoteConfirmationRequest,
    NotePrepareRequest,
    OperationResponse,
    SLAStatusResponse,
)
from .service import ChatService
from .settings import Settings
from .tools import CRMAuthorizationError, TicketNotFoundError


def create_app(
    database_path: str | Path | None = None,
    *,
    configure_logs: bool = True,
) -> FastAPI:
    settings = Settings.from_environment()
    resolved_database_path = Path(database_path or settings.database_path)
    if configure_logs:
        configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        database = Database(resolved_database_path)
        database.initialize()
        chat_service = ChatService(database, settings=settings)
        app.state.chat_service = chat_service
        app.state.note_service = ControlledNoteService(
            database, chat_service.crm_tool
        )
        app.state.handoff_service = CaseHandoffService(
            database,
            chat_service.crm_tool,
            chat_service.entitlement_tool,
            chat_service.evidence_tool,
        )
        app.state.sla_service = chat_service.sla_service
        yield

    app = FastAPI(
        title="FDE Servicing Agent",
        version="0.4.0",
        description="Local-only Milestone 4 evaluation-driven servicing proof of concept",
        lifespan=lifespan,
    )

    async def get_user_context(
        x_demo_user: str = Header(default="demo-agent-alex"),
    ) -> DemoUserContext:
        user_context = resolve_demo_user(x_demo_user)
        if user_context is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unknown local demo user.",
            )
        return user_context

    async def get_chat_service(request: Request) -> ChatService:
        return request.app.state.chat_service

    async def get_note_service(request: Request) -> ControlledNoteService:
        return request.app.state.note_service

    async def get_handoff_service(request: Request) -> CaseHandoffService:
        return request.app.state.handoff_service

    def raise_action_http(exc: Exception) -> None:
        if isinstance(
            exc,
            (CRMAuthorizationError, OperationAuthorizationError),
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="The selected demo user is not authorized for this operation.",
            ) from exc
        if isinstance(exc, (TicketNotFoundError, OperationNotFoundError)):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The requested local demo record was not found.",
            ) from exc
        if isinstance(exc, StaleConfirmationError):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=exc.response.model_dump(),
            ) from exc
        if isinstance(
            exc,
            (IdempotencyConflictError, InvalidHandoffTransitionError),
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        if isinstance(exc, ValueError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        raise exc

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(
        request_body: ChatRequest,
        user_context: DemoUserContext = Depends(get_user_context),
        chat_service: ChatService = Depends(get_chat_service),
    ) -> ChatResponse:
        return chat_service.chat(request_body.message, user_context)

    @app.post(
        "/actions/notes/prepare",
        response_model=OperationResponse,
        tags=["controlled-actions"],
    )
    async def prepare_note(
        body: NotePrepareRequest,
        user_context: DemoUserContext = Depends(get_user_context),
        note_service: ControlledNoteService = Depends(get_note_service),
    ) -> OperationResponse:
        try:
            return note_service.prepare_note(
                body.ticket_id,
                body.note,
                user_context,
                operation_id=body.operation_id,
            )
        except Exception as exc:
            raise_action_http(exc)
            raise

    @app.post(
        "/actions/notes/{operation_id}/confirm",
        response_model=OperationResponse,
        tags=["controlled-actions"],
    )
    async def confirm_note(
        operation_id: str,
        body: NoteConfirmationRequest,
        user_context: DemoUserContext = Depends(get_user_context),
        note_service: ControlledNoteService = Depends(get_note_service),
    ) -> OperationResponse:
        try:
            return note_service.confirm_note(
                operation_id,
                user_context,
                customer_confirmed=body.customer_confirmed,
                simulate_timeout_after_success=body.simulate_timeout_after_success,
            )
        except Exception as exc:
            raise_action_http(exc)
            raise

    @app.post(
        "/actions/notes/{operation_id}/cancel",
        response_model=OperationResponse,
        tags=["controlled-actions"],
    )
    async def cancel_note(
        operation_id: str,
        user_context: DemoUserContext = Depends(get_user_context),
        note_service: ControlledNoteService = Depends(get_note_service),
    ) -> OperationResponse:
        try:
            return note_service.cancel_note(operation_id, user_context)
        except Exception as exc:
            raise_action_http(exc)
            raise

    @app.get(
        "/actions/operations/{operation_id}",
        response_model=OperationResponse,
        tags=["controlled-actions"],
    )
    async def get_operation(
        operation_id: str,
        user_context: DemoUserContext = Depends(get_user_context),
        note_service: ControlledNoteService = Depends(get_note_service),
    ) -> OperationResponse:
        try:
            return note_service.get_operation(operation_id, user_context)
        except Exception as exc:
            raise_action_http(exc)
            raise

    @app.post(
        "/actions/operations/{operation_id}/reconcile",
        response_model=OperationResponse,
        tags=["controlled-actions"],
    )
    async def reconcile_operation(
        operation_id: str,
        user_context: DemoUserContext = Depends(get_user_context),
        note_service: ControlledNoteService = Depends(get_note_service),
    ) -> OperationResponse:
        try:
            return note_service.reconcile(operation_id, user_context)
        except Exception as exc:
            raise_action_http(exc)
            raise

    @app.post(
        "/handoffs",
        response_model=HandoffResponse,
        tags=["human-escalation"],
    )
    async def create_handoff(
        body: HandoffCreateRequest,
        user_context: DemoUserContext = Depends(get_user_context),
        handoff_service: CaseHandoffService = Depends(get_handoff_service),
    ) -> HandoffResponse:
        try:
            return handoff_service.create(
                body.ticket_id,
                body.reason,
                body.troubleshooting_history,
                user_context,
                operation_id=body.operation_id,
            )
        except Exception as exc:
            raise_action_http(exc)
            raise

    @app.get(
        "/handoffs/{handoff_id}",
        response_model=HandoffResponse,
        tags=["human-escalation"],
    )
    async def get_handoff(
        handoff_id: str,
        user_context: DemoUserContext = Depends(get_user_context),
        handoff_service: CaseHandoffService = Depends(get_handoff_service),
    ) -> HandoffResponse:
        try:
            return handoff_service.get(handoff_id, user_context)
        except Exception as exc:
            raise_action_http(exc)
            raise

    @app.post(
        "/handoffs/{handoff_id}/accept",
        response_model=HandoffResponse,
        tags=["human-escalation"],
    )
    async def accept_handoff(
        handoff_id: str,
        body: HandoffTransitionRequest,
        user_context: DemoUserContext = Depends(get_user_context),
        handoff_service: CaseHandoffService = Depends(get_handoff_service),
    ) -> HandoffResponse:
        try:
            return handoff_service.accept(
                handoff_id, user_context, operation_id=body.operation_id
            )
        except Exception as exc:
            raise_action_http(exc)
            raise

    @app.post(
        "/handoffs/{handoff_id}/resolve",
        response_model=HandoffResponse,
        tags=["human-escalation"],
    )
    async def resolve_handoff(
        handoff_id: str,
        body: HandoffTransitionRequest,
        user_context: DemoUserContext = Depends(get_user_context),
        handoff_service: CaseHandoffService = Depends(get_handoff_service),
    ) -> HandoffResponse:
        try:
            return handoff_service.resolve(
                handoff_id, user_context, operation_id=body.operation_id
            )
        except Exception as exc:
            raise_action_http(exc)
            raise

    @app.get(
        "/tickets/{ticket_id}/handoffs",
        response_model=list[HandoffResponse],
        tags=["human-escalation"],
    )
    async def list_handoffs(
        ticket_id: str,
        user_context: DemoUserContext = Depends(get_user_context),
        handoff_service: CaseHandoffService = Depends(get_handoff_service),
    ) -> list[HandoffResponse]:
        try:
            return handoff_service.list_for_ticket(ticket_id, user_context)
        except Exception as exc:
            raise_action_http(exc)
            raise

    @app.post(
        "/sla/{ticket_id}/check",
        response_model=SLAStatusResponse,
        tags=["sla"],
    )
    async def check_sla(
        ticket_id: str,
        request: Request,
        user_context: DemoUserContext = Depends(get_user_context),
    ) -> SLAStatusResponse:
        try:
            return request.app.state.sla_service.check_ticket(
                ticket_id, user_context
            )
        except Exception as exc:
            raise_action_http(exc)
            raise

    return app


app = create_app()

