from __future__ import annotations

from datetime import date

from servicing_agent.auth import DEMO_USERS
from servicing_agent.database import Database
from servicing_agent.retrieval import (
    DocumentEmbeddingIndex,
    EmbeddingUnavailableError,
    HybridRetrievalService,
    KnowledgeDocument,
    RankedDocument,
    reciprocal_rank_fusion,
)
from servicing_agent.service import ChatService


class ControlledEmbeddingProvider:
    model_id = "controlled-test-embeddings"

    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "escalated" in lowered or (
                ("send" in lowered or "sent" in lowered) and "tier 2" in lowered
            ):
                vectors.append([1.0, 0.0, 0.0])
            elif "response target" in lowered or "response time" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class FailingEmbeddingProvider:
    model_id = "failing-test-embeddings"

    def encode(self, texts):
        del texts
        raise EmbeddingUnavailableError("mock embedding failure")


def make_document(document_id: str, text: str) -> KnowledgeDocument:
    return KnowledgeDocument.from_dict(
        {
            "document_id": document_id,
            "source_id": document_id,
            "title": document_id,
            "document_type": "product_information",
            "version": "1.0",
            "approval_status": "approved",
            "effective_date": "2026-01-01",
            "expiration_date": None,
            "product_ids": ["PRD-TEST"],
            "subscription_tiers": ["Standard"],
            "customer_id": None,
            "case_id": None,
            "access_scope": "public",
            "conflicts_with": [],
            "text": text,
        }
    )


def test_semantic_retrieval_uses_mocked_cosine_rankings(tmp_path) -> None:
    provider = ControlledEmbeddingProvider()
    documents = [
        make_document("DOC-ESCALATION", "The case was escalated to Tier 2."),
        make_document("DOC-POLICY", "The response target is two hours."),
    ]
    retriever = HybridRetrievalService(
        documents,
        mode="semantic",
        embedding_provider=provider,
        embedding_cache_dir=tmp_path,
        semantic_min_score=0.5,
    )

    result = retriever.search(
        "What caused support to send this to Tier 2?",
        as_of=date(2026, 11, 7),
        customer_id=None,
        case_id=None,
        entitlements=[],
    )

    assert result.evidence[0].document.document_id == "DOC-ESCALATION"
    assert result.diagnostics.effective_mode == "semantic"
    assert result.diagnostics.rankings["semantic"] == ("DOC-ESCALATION",)


def test_rrf_combines_rankings_deterministically() -> None:
    first = make_document("DOC-A", "alpha")
    second = make_document("DOC-B", "beta")
    third = make_document("DOC-C", "gamma")

    fused = reciprocal_rank_fusion(
        {
            "lexical": [RankedDocument(first, 10), RankedDocument(second, 9)],
            "semantic": [RankedDocument(second, 0.9), RankedDocument(third, 0.8)],
        },
        k=60,
    )

    assert [item.document.document_id for item in fused] == [
        "DOC-B",
        "DOC-A",
        "DOC-C",
    ]


def test_semantic_security_filter_excludes_other_customer_rankings(tmp_path) -> None:
    database = Database(tmp_path / "security.db")
    database.initialize()
    provider = ControlledEmbeddingProvider()
    retriever = HybridRetrievalService(
        mode="semantic",
        embedding_provider=provider,
        embedding_cache_dir=tmp_path / "cache",
        semantic_min_score=-1,
    )
    service = ChatService(
        database,
        as_of=date(2026, 11, 7),
        retriever=retriever,
    )
    user = DEMO_USERS["demo-agent-jordan"]
    ticket = service.crm_tool.get_ticket("CASE-4823", user)
    entitlements = service.entitlement_tool.get_active_entitlements(
        ticket["customer_id"], user
    )

    result = service.evidence_tool.search(
        "customer case investigation",
        user_context=user,
        ticket=ticket,
        entitlements=entitlements,
    )

    ranked_ids = set(result.diagnostics.rankings["semantic"])
    assert "CASE-NOTE-4823-v1.0" in ranked_ids
    assert "CASE-NOTE-4821-v1.0" not in ranked_ids
    assert "CALL-4821-20261105-v1.0" not in ranked_ids
    assert "CASE-NOTE-4824-v1.0" not in ranked_ids


def test_policy_applicability_is_preserved_in_semantic_mode(tmp_path) -> None:
    database = Database(tmp_path / "policy.db")
    database.initialize()
    retriever = HybridRetrievalService(
        mode="semantic",
        embedding_provider=ControlledEmbeddingProvider(),
        embedding_cache_dir=tmp_path / "cache",
        semantic_min_score=-1,
    )
    service = ChatService(
        database,
        as_of=date(2026, 11, 7),
        retriever=retriever,
    )
    user = DEMO_USERS["demo-agent-alex"]

    premium = service.chat(
        "What response time applies to CASE-4821?", user
    ).retrieval_debug
    standard = service.chat(
        "What response time applies to CASE-4824?", user
    ).retrieval_debug

    assert premium is not None and standard is not None
    assert "POL-TELEMETRY-RESPONSE-v2.0" in premium.rankings["semantic"]
    assert "POL-TELEMETRY-RESPONSE-v1.0" not in premium.rankings["semantic"]
    assert "POL-TELEMETRY-RESPONSE-v1.0" in standard.rankings["semantic"]
    assert "POL-TELEMETRY-RESPONSE-v2.0" not in standard.rankings["semantic"]


def test_hybrid_citations_only_reference_selected_evidence(tmp_path) -> None:
    database = Database(tmp_path / "citations.db")
    database.initialize()
    retriever = HybridRetrievalService(
        mode="hybrid",
        embedding_provider=ControlledEmbeddingProvider(),
        embedding_cache_dir=tmp_path / "cache",
        semantic_min_score=0.5,
    )
    service = ChatService(
        database,
        as_of=date(2026, 11, 7),
        retriever=retriever,
    )

    response = service.chat(
        "Why was CASE-4821 escalated?", DEMO_USERS["demo-agent-alex"]
    )

    assert response.retrieval_debug is not None
    selected_ids = set(response.retrieval_debug.selected_document_ids)
    assert selected_ids == {"CASE-NOTE-4821-v1.0"}
    assert {citation.source_id for citation in response.citations} == {
        "CASE-NOTE-4821"
    }


def test_document_embedding_cache_reuses_and_refreshes_index(tmp_path) -> None:
    documents = [make_document("DOC-A", "The case was escalated.")]
    first_provider = ControlledEmbeddingProvider()
    first = DocumentEmbeddingIndex(documents, first_provider, tmp_path)

    first.ensure()
    assert first.last_cache_status == "created"
    assert len(first_provider.calls) == 1

    second_provider = ControlledEmbeddingProvider()
    second = DocumentEmbeddingIndex(documents, second_provider, tmp_path)
    second.ensure()
    assert second.last_cache_status == "disk"
    assert second_provider.calls == []

    second.refresh()
    assert second.last_cache_status == "refreshed"
    assert len(second_provider.calls) == 1

    changed_provider = ControlledEmbeddingProvider()
    changed = DocumentEmbeddingIndex(
        [make_document("DOC-A", "The response target changed.")],
        changed_provider,
        tmp_path,
    )
    changed.ensure()
    assert changed.last_cache_status == "created"
    assert len(changed_provider.calls) == 1


def test_embedding_failure_falls_back_to_lexical(tmp_path) -> None:
    documents = [make_document("DOC-A", "Delayed telemetry batches need review.")]
    retriever = HybridRetrievalService(
        documents,
        mode="semantic",
        embedding_provider=FailingEmbeddingProvider(),
        embedding_cache_dir=tmp_path,
    )

    result = retriever.search(
        "delayed telemetry batches",
        as_of=date(2026, 11, 7),
        customer_id=None,
        case_id=None,
        entitlements=[],
    )

    assert result.diagnostics.requested_mode == "semantic"
    assert result.diagnostics.effective_mode == "lexical"
    assert result.diagnostics.fallback_reason == "mock embedding failure"
    assert result.evidence[0].document.document_id == "DOC-A"
