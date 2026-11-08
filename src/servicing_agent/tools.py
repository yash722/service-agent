from __future__ import annotations

from datetime import date
from typing import Any

from .auth import DemoUserContext
from .database import Database
from .retrieval import HybridRetrievalService, RetrievalResult


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


class EntitlementTool:
    """Authoritative local entitlement reader with record-level authorization."""

    def __init__(self, database: Database, as_of: date):
        self.database = database
        self.as_of = as_of

    def get_active_entitlements(
        self, customer_id: str, user_context: DemoUserContext
    ) -> list[dict[str, Any]]:
        if not user_context.can_access_customer(customer_id):
            raise CRMAuthorizationError(
                f"{user_context.user_id} cannot access entitlements for {customer_id}"
            )
        return self.database.fetch_active_entitlements(
            customer_id, self.as_of.isoformat()
        )


class EvidenceRetrievalTool:
    """Authorization boundary around the local evidence retriever."""

    def __init__(self, retriever: HybridRetrievalService, as_of: date):
        self.retriever = retriever
        self.as_of = as_of

    def search(
        self,
        query: str,
        *,
        user_context: DemoUserContext,
        ticket: dict[str, Any] | None,
        entitlements: list[dict[str, Any]],
    ) -> RetrievalResult:
        customer_id = ticket["customer_id"] if ticket else None
        case_id = ticket["ticket_id"] if ticket else None
        if customer_id and not user_context.can_access_customer(customer_id):
            raise CRMAuthorizationError(
                f"{user_context.user_id} cannot retrieve evidence for {customer_id}"
            )
        if any(
            entitlement["customer_id"] != customer_id
            or not user_context.can_access_customer(entitlement["customer_id"])
            for entitlement in entitlements
        ):
            raise CRMAuthorizationError("Untrusted entitlement context rejected")
        return self.retriever.search(
            query,
            as_of=self.as_of,
            customer_id=customer_id,
            case_id=case_id,
            entitlements=entitlements,
        )

