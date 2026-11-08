from datetime import date

import pytest

from servicing_agent.auth import DEMO_USERS
from servicing_agent.database import Database
from servicing_agent.retrieval import KnowledgeDocument, LexicalRetrievalService
from servicing_agent.service import ChatService
from servicing_agent.tools import CRMAuthorizationError


@pytest.fixture()
def service(tmp_path) -> ChatService:
    database = Database(tmp_path / "retrieval.db")
    database.initialize()
    return ChatService(database, as_of=date(2026, 11, 7))


def retrieve_for_case(service: ChatService, query: str, case_id: str):
    user = DEMO_USERS["demo-agent-alex"]
    ticket = service.crm_tool.get_ticket(case_id, user)
    entitlements = service.entitlement_tool.get_active_entitlements(
        ticket["customer_id"], user
    )
    return service.evidence_tool.search(
        query,
        user_context=user,
        ticket=ticket,
        entitlements=entitlements,
    )


def test_lexical_retrieval_ranks_relevant_case_note(service: ChatService) -> None:
    result = retrieve_for_case(
        service, "Why was CASE-4821 escalated?", "CASE-4821"
    )

    assert result.status == "sufficient"
    assert result.evidence[0].document.source_id == "CASE-NOTE-4821"
    assert all(
        item.document.customer_id in {None, "CUST-1001"}
        for item in result.evidence
    )


def test_current_approved_policy_excludes_outdated_and_draft_versions(
    service: ChatService,
) -> None:
    result = retrieve_for_case(
        service,
        "What initial response target applies to CASE-4821?",
        "CASE-4821",
    )

    versions = {
        item.document.version
        for item in result.evidence
        if item.document.source_id == "POL-TELEMETRY-RESPONSE"
    }
    assert versions == {"2.0"}
    assert any("no longer effective" in warning for warning in result.warnings)
    assert any("not approved" in warning for warning in result.warnings)


def test_policy_applicability_differs_for_standard_and_premium(
    service: ChatService,
) -> None:
    premium = service.chat(
        "What initial response target applies to CASE-4821?",
        DEMO_USERS["demo-agent-alex"],
    )
    standard = service.chat(
        "What initial response target applies to CASE-4824?",
        DEMO_USERS["demo-agent-alex"],
    )

    assert "Premium" in premium.reply
    assert "two clock hours" in premium.reply
    assert any(
        citation.source_id == "POL-TELEMETRY-RESPONSE"
        and citation.document_version == "2.0"
        for citation in premium.citations
    )
    assert "Standard" in standard.reply
    assert "eight business hours" in standard.reply
    assert any(
        citation.source_id == "POL-TELEMETRY-RESPONSE"
        and citation.document_version == "1.0"
        for citation in standard.citations
    )


def test_protected_evidence_tool_rechecks_record_authorization(
    service: ChatService,
) -> None:
    ticket = service.crm_tool.database.fetch_ticket("CASE-4821")
    assert ticket is not None

    with pytest.raises(CRMAuthorizationError):
        service.evidence_tool.search(
            "Why was the case escalated?",
            user_context=DEMO_USERS["demo-agent-jordan"],
            ticket=ticket,
            entitlements=[],
        )


def test_explicitly_conflicting_evidence_is_not_marked_sufficient() -> None:
    base = {
        "title": "Connector restart guidance",
        "document_type": "product_information",
        "version": "1.0",
        "approval_status": "approved",
        "effective_date": "2026-01-01",
        "expiration_date": None,
        "product_ids": ["PRD-TELEMETRY"],
        "subscription_tiers": ["Standard", "Premium"],
        "customer_id": None,
        "case_id": None,
        "access_scope": "public",
        "text": "Connector restart guidance applies after repeated retry failures.",
    }
    first = KnowledgeDocument.from_dict(
        {
            **base,
            "document_id": "KB-CONFLICT-A-v1",
            "source_id": "KB-CONFLICT-A",
            "conflicts_with": ["KB-CONFLICT-B-v1"],
        }
    )
    second = KnowledgeDocument.from_dict(
        {
            **base,
            "document_id": "KB-CONFLICT-B-v1",
            "source_id": "KB-CONFLICT-B",
            "conflicts_with": ["KB-CONFLICT-A-v1"],
        }
    )
    retriever = LexicalRetrievalService([first, second])

    result = retriever.search(
        "connector restart guidance",
        as_of=date(2026, 11, 7),
        customer_id=None,
        case_id=None,
        entitlements=[],
    )

    assert result.status == "conflicting"
    assert any("explicit conflict" in warning for warning in result.warnings)

