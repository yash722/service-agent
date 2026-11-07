from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .auth import DemoUserContext
from .database import Database


class TicketNotFoundError(LookupError):
    pass


class CRMAuthorizationError(PermissionError):
    pass


class MockCRMTool:
    """Local CRM adapter; the interface can be replaced in a later milestone."""

    def __init__(self, database: Database):
        self.database = database

    def get_ticket(
        self, ticket_id: str, user_context: DemoUserContext
    ) -> dict[str, Any]:
        ticket = self.database.fetch_ticket(ticket_id)
        if ticket is None:
            raise TicketNotFoundError(ticket_id)
        if not user_context.can_access_customer(ticket["customer_id"]):
            raise CRMAuthorizationError(
                f"{user_context.user_id} cannot access {ticket_id.upper()}"
            )
        return ticket


@dataclass(frozen=True)
class MockKnowledgeTool:
    def search(self, message: str) -> str:
        del message
        return "Review the case timeline and confirm the next owner action."


@dataclass(frozen=True)
class MockPolicyTool:
    def guidance(self, message: str) -> str:
        del message
        return "No production action is allowed from this Milestone 1 demo."

