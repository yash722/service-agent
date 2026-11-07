from __future__ import annotations

import logging
from uuid import uuid4

from .auth import DemoUserContext
from .database import Database
from .intent_router import extract_ticket_id, route_intent
from .logging_config import log_event
from .schemas import ChatResponse
from .tools import (
    CRMAuthorizationError,
    MockCRMTool,
    MockKnowledgeTool,
    MockPolicyTool,
    TicketNotFoundError,
)
from .workflow import build_complex_workflow

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, database: Database):
        self.crm_tool = MockCRMTool(database)
        self.complex_workflow = build_complex_workflow(
            self.crm_tool,
            MockKnowledgeTool(),
            MockPolicyTool(),
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
                }
            )
            response = ChatResponse(
                reply=result["reply"],
                intent="complex_request",
                execution_id=execution_id,
                steps=result["steps"],
            )

        log_event(
            logger,
            "chat.completed",
            execution_id=execution_id,
            intent=response.intent,
            step_count=len(response.steps),
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
            reply = (
                f"{ticket['ticket_id']} is {ticket['status']}. "
                f"Priority: {ticket['priority']}. Owner: {ticket['owner']}. "
                f"Last updated: {ticket['updated_at']}."
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
        )

