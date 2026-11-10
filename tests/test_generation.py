from __future__ import annotations

from datetime import date

from servicing_agent.generation import (
    DeterministicGroundedGenerator,
    OpenAICompatibleConfig,
    OpenAICompatibleGroundedGenerator,
    _claim_for_citation,
    _select_history_records,
    citation_supports_claim,
    classify_answer_type,
    document_citation,
    history_citation,
)
from servicing_agent.history import HistoryRecord, HistoryResult
from servicing_agent.retrieval import (
    KnowledgeDocument,
    RetrievalResult,
    RetrievedEvidence,
    load_default_corpus,
)


def ticket() -> dict[str, str]:
    return {
        "ticket_id": "CASE-4821",
        "customer_id": "CUST-1001",
        "subject": "Telemetry delay",
        "status": "In Progress",
        "priority": "High",
        "owner": "Tier 2 Support",
        "updated_at": "2026-11-06T19:30:00Z",
    }


def retrieval() -> RetrievalResult:
    document = KnowledgeDocument(
        document_id="CASE-NOTE-4821-v1.0",
        source_id="CASE-NOTE-4821",
        title="Tier 2 investigation note for CASE-4821",
        document_type="case_note",
        version="1.0",
        approval_status="approved",
        effective_date=date(2026, 11, 6),
        expiration_date=None,
        product_ids=("PRD-TELEMETRY",),
        subscription_tiers=("Premium",),
        customer_id="CUST-1001",
        case_id="CASE-4821",
        access_scope="customer_record",
        conflicts_with=(),
        text=(
            "CASE-4821 was escalated after three consecutive synchronization "
            "windows exceeded the latency baseline."
        ),
    )
    return RetrievalResult(
        (RetrievedEvidence(document, 1.0),),
        "sufficient",
        (),
    )


def corpus_retrieval(*document_ids: str) -> RetrievalResult:
    documents = {
        document.document_id: document for document in load_default_corpus()
    }
    return RetrievalResult(
        tuple(
            RetrievedEvidence(documents[document_id], 1.0 - index / 100)
            for index, document_id in enumerate(document_ids)
        ),
        "sufficient",
        (),
    )


def history() -> HistoryResult:
    return HistoryResult(
        records=(
            HistoryRecord(
                source_id="SF-CASE-HISTORY-HIST-001",
                source_table="CASE_HISTORY",
                customer_id="CUST-1001",
                case_id="CASE-4821",
                event_at="2026-11-06T18:00:00Z",
                title="UNDER_INVESTIGATION",
                summary="The case moved to Tier 2 investigation.",
            ),
        ),
        backend="snowflake",
        status="available",
        retrieved_at="2026-11-08T12:00:00Z",
    )


class FakeResponse:
    def __init__(
        self,
        content: str,
        finish_reason: str = "stop",
        output_tokens: int = 42,
    ):
        self.content = content
        self.finish_reason = finish_reason
        self.output_tokens = output_tokens

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {"content": self.content},
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": {"completion_tokens": self.output_tokens},
        }


def generator() -> OpenAICompatibleGroundedGenerator:
    return OpenAICompatibleGroundedGenerator(
        OpenAICompatibleConfig(
            api_key="synthetic-test-key",
            model="test-model",
            api_base="https://llm.example.test/v1",
        )
    )


def test_llm_generation_uses_grouped_authorized_evidence_and_citations(
    monkeypatch,
) -> None:
    captured: dict = {}
    content = (
        "CASE-4821 is currently in progress with Tier 2 Support "
        "[CRM-CASE-4821 record]. It was escalated after three consecutive "
        "degraded synchronization windows [CASE-NOTE-4821 v1.0]. Historical "
        "context shows it moved into investigation "
        "[SF-CASE-HISTORY-HIST-001 record]. The evidence does not confirm a "
        "root cause or resolution date."
    )

    def fake_post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return FakeResponse(content)

    monkeypatch.setattr("servicing_agent.generation.httpx.post", fake_post)

    outcome = generator().generate(
        question="Why was CASE-4821 escalated, and what is its current state?",
        ticket=ticket(),
        entitlements=[],
        retrieval=retrieval(),
        history=history(),
    )

    assert outcome.mode == "llm"
    assert outcome.latency_ms >= 0
    assert outcome.fallback_reason is None
    assert outcome.finish_reason == "stop"
    assert outcome.output_tokens == 42
    assert outcome.reply == content
    assert {citation.source_id for citation in outcome.citations} == {
        "CRM-CASE-4821",
        "CASE-NOTE-4821",
        "SF-CASE-HISTORY-HIST-001",
    }
    prompt = captured["json"]["messages"][1]["content"]
    assert "CONFIRMED CURRENT CRM FACTS" in prompt
    assert "APPROVED RETRIEVED KNOWLEDGE AND CASE EVIDENCE" in prompt
    assert "AUTHORIZED SNOWFLAKE HISTORICAL CONTEXT" in prompt
    assert "RESPONSE FORMAT REQUIREMENTS" in prompt
    assert "[CASE-NOTE-4821 v1.0]" in prompt
    assert "[SF-CASE-HISTORY-HIST-001 record]" in prompt
    assert captured["json"]["model"] == "test-model"


def test_qwen3_request_disables_thinking_on_the_user_turn(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse(
            "Escalation followed repeated degraded windows "
            "[CASE-NOTE-4821 v1.0]."
        )

    monkeypatch.setattr("servicing_agent.generation.httpx.post", fake_post)
    local_generator = OpenAICompatibleGroundedGenerator(
        OpenAICompatibleConfig(
            api_key="ollama",
            model="qwen3:4b-instruct",
            api_base="http://localhost:11434/v1",
            max_tokens=512,
            reasoning_effort="none",
        )
    )

    outcome = local_generator.generate(
        question="Why was CASE-4821 escalated?",
        ticket=ticket(),
        entitlements=[],
        retrieval=retrieval(),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert outcome.mode == "llm"
    assert captured["json"]["max_tokens"] == 512
    assert captured["json"]["reasoning_effort"] == "none"
    assert captured["json"]["messages"][1]["content"].endswith("/no_think")


def test_missing_evidence_bypasses_llm_and_does_not_fabricate(
    monkeypatch,
) -> None:
    def unexpected_call(*args, **kwargs):
        raise AssertionError("LLM must not receive a request without evidence")

    monkeypatch.setattr(
        "servicing_agent.generation.httpx.post", unexpected_call
    )
    outcome = generator().generate(
        question="What is the confirmed root cause?",
        ticket=ticket(),
        entitlements=[],
        retrieval=RetrievalResult(
            (),
            "insufficient",
            ("No relevant, current, authorized evidence was found.",),
        ),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert outcome.mode == "deterministic"
    assert outcome.fallback_reason is None
    assert outcome.citations == ()
    assert "insufficient" in outcome.reply
    assert "will not infer" in outcome.reply


def test_unknown_or_incomplete_llm_citations_trigger_readable_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "servicing_agent.generation.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            "A root cause was confirmed [UNAUTHORIZED-SOURCE record]."
        ),
    )

    outcome = generator().generate(
        question="Why was CASE-4821 escalated?",
        ticket=ticket(),
        entitlements=[],
        retrieval=retrieval(),
        history=history(),
    )

    assert outcome.mode == "deterministic"
    assert outcome.fallback_reason == "LLM output contained an unrecognized citation."
    assert "**Confirmed current CRM facts**" in outcome.reply
    assert "**Historical context**" in outcome.reply
    assert "**What remains unconfirmed**" in outcome.reply
    assert "not generated by an LLM" in outcome.reply
    assert "A root cause was confirmed" not in outcome.reply
    assert "LLM output contained an unrecognized citation." not in outcome.warnings
    assert all(
        citation.citation_label in outcome.reply
        for citation in outcome.citations
    )


def test_deterministic_history_answer_separates_facts_and_unknowns() -> None:
    outcome = DeterministicGroundedGenerator().generate(
        question="Summarize the history of CASE-4821.",
        ticket=ticket(),
        entitlements=[],
        retrieval=retrieval(),
        history=history(),
    )

    assert outcome.mode == "deterministic"
    assert "three consecutive synchronization windows" in outcome.reply
    assert "UNDER_INVESTIGATION" in outcome.reply
    assert "confirmed root cause" in outcome.reply
    assert "confirmed resolution date" in outcome.reply
    assert {citation.source_id for citation in outcome.citations} == {
        "CRM-CASE-4821",
        "CASE-NOTE-4821",
        "SF-CASE-HISTORY-HIST-001",
    }


def test_answer_type_classification_covers_customer_request_shapes() -> None:
    assert classify_answer_type("What is the status of CASE-4821?") == "status"
    assert classify_answer_type("Why is CASE-4821 still under investigation?") == (
        "explanation"
    )
    assert classify_answer_type("When will CASE-4821 be resolved?") == "timeline"
    assert classify_answer_type(
        "Can you guarantee CASE-4821 will be resolved tomorrow?"
    ) == "guarantee"
    assert classify_answer_type("Do we know the root cause?") == (
        "unknown_information"
    )
    assert classify_answer_type("Why was CASE-4821 escalated?") == "escalation"
    assert classify_answer_type("Please escalate CASE-4821") == "action_request"


def test_deterministic_fallback_answers_why_still_investigating_directly() -> None:
    outcome = DeterministicGroundedGenerator().generate(
        question="Why is CASE-4821 still under investigation?",
        ticket=ticket(),
        entitlements=[],
        retrieval=retrieval(),
        history=history(),
    )

    assert outcome.mode == "deterministic"
    assert "does not confirm a single reason" in outcome.reply
    assert "currently In Progress" in outcome.reply
    assert "three consecutive synchronization windows" in outcome.reply
    assert "confirmed resolution date" in outcome.reply
    assert "We are given" not in outcome.reply


def test_deterministic_fallback_refuses_unsupported_resolution_guarantee() -> None:
    outcome = DeterministicGroundedGenerator().generate(
        question="Can you guarantee CASE-4821 will be resolved tomorrow?",
        ticket=ticket(),
        entitlements=[],
        retrieval=retrieval(),
        history=history(),
    )

    assert outcome.reply.startswith("**Answer**\n\nNo.")
    assert "cannot guarantee" in outcome.reply
    assert "no confirmed resolution date" in outcome.reply
    assert "three consecutive synchronization windows" not in outcome.reply
    assert {citation.source_id for citation in outcome.citations} == {
        "CRM-CASE-4821"
    }


def test_reasoning_style_llm_output_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        "servicing_agent.generation.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            "We are given the customer question and need to analyze the evidence. "
            "The case is under investigation [CASE-NOTE-4821 v1.0]."
        ),
    )

    outcome = generator().generate(
        question="Why is CASE-4821 still under investigation?",
        ticket=ticket(),
        entitlements=[],
        retrieval=retrieval(),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert outcome.mode == "deterministic"
    assert outcome.fallback_reason == (
        "LLM output exposed reasoning or planning text."
    )
    assert "We are given" not in outcome.reply
    assert "does not confirm a single reason" in outcome.reply


def test_length_truncated_llm_output_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        "servicing_agent.generation.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            "The case was escalated after repeated synchronization windows "
            "[CASE-NOTE-4821 v1.0].",
            finish_reason="length",
        ),
    )

    outcome = generator().generate(
        question="Why was CASE-4821 escalated?",
        ticket=ticket(),
        entitlements=[],
        retrieval=retrieval(),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert outcome.mode == "deterministic"
    assert outcome.fallback_reason == (
        "LLM output was incomplete (finish_reason=length)."
    )
    assert outcome.finish_reason == "length"
    assert outcome.output_tokens == 42


def test_llm_citation_must_support_the_associated_claim(monkeypatch) -> None:
    monkeypatch.setattr(
        "servicing_agent.generation.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            "The reason is that a replacement spacecraft was launched "
            "[CASE-NOTE-4821 v1.0]."
        ),
    )

    outcome = generator().generate(
        question="Why was CASE-4821 escalated?",
        ticket=ticket(),
        entitlements=[],
        retrieval=retrieval(),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert outcome.mode == "deterministic"
    assert outcome.fallback_reason == (
        "LLM citation did not substantively support its associated claim."
    )
    assert "spacecraft" not in outcome.reply


def test_llm_cannot_infer_why_investigation_remains_open(monkeypatch) -> None:
    monkeypatch.setattr(
        "servicing_agent.generation.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            "CASE-4821 remains under investigation because an index rebuild is "
            "scheduled [CASE-NOTE-4821 v1.0]."
        ),
    )

    outcome = generator().generate(
        question="Why is CASE-4821 still under investigation?",
        ticket=ticket(),
        entitlements=[],
        retrieval=retrieval(),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert outcome.mode == "deterministic"
    assert outcome.fallback_reason == (
        "LLM output inferred why the investigation remains open."
    )
    assert "does not confirm a single reason" in outcome.reply


def test_why_still_prompt_requires_explicit_uncertainty(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse(
            "The evidence does not confirm why CASE-4821 remains under "
            "investigation. The case note records repeated synchronization "
            "windows above the latency baseline [CASE-NOTE-4821 v1.0]."
        )

    monkeypatch.setattr("servicing_agent.generation.httpx.post", fake_post)
    outcome = generator().generate(
        question="Why is CASE-4821 still under investigation?",
        ticket=ticket(),
        entitlements=[],
        retrieval=retrieval(),
        history=HistoryResult(backend="mock", status="empty"),
    )

    prompt = captured["json"]["messages"][1]["content"]
    assert outcome.mode == "llm"
    assert "does not confirm why the investigation remains open" in prompt
    assert "context, not a proven reason" in prompt


def test_guarantee_bypasses_llm_and_excludes_irrelevant_evidence(
    monkeypatch,
) -> None:
    def unexpected_post(*args, **kwargs):
        raise AssertionError("A deterministic guarantee answer must not call an LLM")

    monkeypatch.setattr("servicing_agent.generation.httpx.post", unexpected_post)
    outcome = generator().generate(
        question="Can you guarantee CASE-4821 will be resolved tomorrow?",
        ticket=ticket(),
        entitlements=[],
        retrieval=retrieval(),
        history=history(),
    )

    assert outcome.mode == "deterministic"
    assert "cannot guarantee" in outcome.reply
    assert "CASE-NOTE-4821" not in outcome.reply
    assert "SF-CASE-HISTORY" not in outcome.reply
    assert {citation.source_id for citation in outcome.citations} == {
        "CRM-CASE-4821"
    }


def test_history_citation_never_substitutes_retrieval_time_for_event_time() -> None:
    record = HistoryRecord(
        source_id="SF-CASE-HISTORY-HIST-002",
        source_table="CASE_HISTORY",
        customer_id="CUST-1001",
        case_id="CASE-4821",
        event_at=None,
        title="UNDER_INVESTIGATION",
        summary="The historical event has no event timestamp.",
    )

    citation = history_citation(record, "2026-11-09T12:00:00Z")

    assert citation.effective_date == "historical event timestamp unavailable"
    assert "2026-11-09" not in citation.effective_date


def test_case_4821_fallback_preserves_observed_delay_and_unverified_rebuild() -> None:
    outcome = DeterministicGroundedGenerator().generate(
        question="Why is CASE-4821 still under investigation?",
        ticket=ticket(),
        entitlements=[],
        retrieval=corpus_retrieval("CASE-NOTE-4821-v1.0"),
        history=HistoryResult(backend="mock", status="empty"),
    )

    reply = outcome.reply.lower()
    assert "do document delayed synchronization" in reply
    assert "three consecutive synchronization windows" in reply
    assert "connector retries" in reply
    assert "does not confirm that either action was completed" in reply
    assert "no evidence of delayed synchronization" not in reply
    assert "rebuild was not completed" not in reply


def test_case_4821_escalation_answer_states_observation_not_root_cause() -> None:
    outcome = DeterministicGroundedGenerator().generate(
        question="Why was CASE-4821 escalated?",
        ticket=ticket(),
        entitlements=[],
        retrieval=corpus_retrieval("CASE-NOTE-4821-v1.0"),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert "three consecutive synchronization windows" in outcome.reply
    assert "connector retries" in outcome.reply
    assert "definitive root cause was connector retries" not in outcome.reply


def test_case_4825_completion_requires_explicit_completed_evidence() -> None:
    resolved_ticket = {
        "ticket_id": "CASE-4825",
        "customer_id": "CUST-1001",
        "subject": "Telemetry export backlog",
        "status": "Resolved",
        "priority": "High",
        "owner": "Tier 2 Support",
        "updated_at": "2026-11-01T12:40:00Z",
    }
    outcome = DeterministicGroundedGenerator().generate(
        question="What was confirmed and completed to resolve CASE-4825?",
        ticket=resolved_ticket,
        entitlements=[],
        retrieval=corpus_retrieval("CASE-NOTE-4825-v1.0"),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert "expired connector certificate" in outcome.reply
    assert "certificate rotation was completed" in outcome.reply
    assert "queued export replay completed" in outcome.reply
    assert "backlog cleared" in outcome.reply
    assert "**Verified completion or resolution**" in outcome.reply


def test_case_4823_failed_reproduction_does_not_become_nonoccurrence() -> None:
    dashboard_ticket = {
        "ticket_id": "CASE-4823",
        "customer_id": "CUST-1003",
        "subject": "Mobile dashboard does not refresh",
        "status": "Awaiting Customer",
        "priority": "Low",
        "owner": "Tier 1 Support",
        "updated_at": "2026-11-04T16:45:00Z",
    }
    outcome = DeterministicGroundedGenerator().generate(
        question=(
            "Does the unsuccessful reproduction attempt mean CASE-4823 never occurred?"
        ),
        ticket=dashboard_ticket,
        entitlements=[],
        retrieval=corpus_retrieval("CASE-NOTE-4823-v1.0"),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert outcome.reply.startswith("**Answer**\n\nNo.")
    assert "does not establish that the reported issue never occurred" in outcome.reply
    assert "does not prove the issue did not occur" in outcome.reply


def test_case_4824_standard_policy_answer_does_not_infer_escalation() -> None:
    standard_ticket = {
        "ticket_id": "CASE-4824",
        "customer_id": "CUST-1002",
        "subject": "Intermittent telemetry export latency",
        "status": "Monitoring",
        "priority": "Medium",
        "owner": "Tier 1 Support",
        "updated_at": "2026-11-02T18:20:00Z",
    }
    outcome = DeterministicGroundedGenerator().generate(
        question="What initial response target applies to CASE-4824?",
        ticket=standard_ticket,
        entitlements=[
            {
                "entitlement_id": "ENT-1002-TELEMETRY",
                "customer_id": "CUST-1002",
                "product_id": "PRD-TELEMETRY",
                "subscription_tier": "Standard",
                "status": "Active",
                "start_date": "2025-08-15",
                "end_date": None,
            }
        ],
        retrieval=corpus_retrieval("POL-TELEMETRY-RESPONSE-v1.0"),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert "eight business hours" in outcome.reply
    assert "Standard" in outcome.reply
    assert "Escalation review" not in outcome.reply


def test_claim_validation_rejects_absence_and_completion_certainty_errors() -> None:
    item = corpus_retrieval("CASE-NOTE-4821-v1.0").evidence[0]
    citation = document_citation(item)

    assert not citation_supports_claim(
        "There is no evidence of delayed synchronization "
        f"{citation.citation_label}.",
        citation,
    )
    assert not citation_supports_claim(
        "The scheduled index rebuild was not completed "
        f"{citation.citation_label}.",
        citation,
    )
    assert citation_supports_claim(
        "The record does not confirm that the scheduled rebuild was completed "
        f"{citation.citation_label}.",
        citation,
    )


def test_llm_certainty_error_triggers_grounded_fallback(monkeypatch) -> None:
    citation_label = "[CASE-NOTE-4821 v1.0]"
    monkeypatch.setattr(
        "servicing_agent.generation.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            "There is no evidence of delayed synchronization "
            f"{citation_label}."
        ),
    )

    outcome = generator().generate(
        question="Why is CASE-4821 still under investigation?",
        ticket=ticket(),
        entitlements=[],
        retrieval=corpus_retrieval("CASE-NOTE-4821-v1.0"),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert outcome.mode == "deterministic"
    assert outcome.fallback_reason == (
        "LLM output converted uncertain or historical evidence into an "
        "unsupported certainty or negation."
    )
    assert "do document delayed synchronization" in outcome.reply.lower()


def test_adjacent_citations_share_the_same_preceding_claim() -> None:
    content = (
        "Connector retries were observed and delayed batches eventually arrived "
        "[CASE-NOTE-4821 v1.0][SF-CASE-HISTORY-HIST-001 record]."
    )

    first = _claim_for_citation(content, "[CASE-NOTE-4821 v1.0]")
    second = _claim_for_citation(
        content, "[SF-CASE-HISTORY-HIST-001 record]"
    )

    assert first == second
    assert first.startswith("Connector retries were observed")


def test_llm_cannot_turn_unrecorded_log_receipt_into_not_received(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "servicing_agent.generation.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            "Support requested connector diagnostic logs, but the logs were not "
            "received [CASE-NOTE-4821 v1.0]."
        ),
    )

    outcome = generator().generate(
        question="Were the requested connector diagnostic logs received for CASE-4821?",
        ticket=ticket(),
        entitlements=[],
        retrieval=corpus_retrieval("CASE-NOTE-4821-v1.0"),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert outcome.mode == "deterministic"
    assert outcome.fallback_reason == (
        "LLM output converted uncertain or historical evidence into an "
        "unsupported certainty or negation."
    )
    assert "receipt of the requested logs is not recorded" in outcome.reply.lower()


def test_policy_prompt_focuses_on_applicable_policy_and_entitlement(
    monkeypatch,
) -> None:
    captured: dict = {}
    policy_ticket = {
        "ticket_id": "CASE-4824",
        "customer_id": "CUST-1002",
        "subject": "Intermittent telemetry export latency",
        "status": "Monitoring",
        "priority": "Medium",
        "owner": "Tier 1 Support",
        "updated_at": "2026-11-02T18:20:00Z",
    }
    entitlements = [
        {
            "entitlement_id": "ENT-1002-TELEMETRY",
            "customer_id": "CUST-1002",
            "product_id": "PRD-TELEMETRY",
            "subscription_tier": "Standard",
            "status": "Active",
            "start_date": "2025-08-15",
            "end_date": None,
        }
    ]
    policy_history = HistoryResult(
        records=(
            HistoryRecord(
                source_id="SF-CASE-HISTORY-HIST-4824-001",
                source_table="CASE_HISTORY",
                customer_id="CUST-1002",
                case_id="CASE-4824",
                event_at="2026-11-02T18:20:00Z",
                title="MONITORING",
                summary="A retry completed normally.",
            ),
        ),
        backend="snowflake",
        status="available",
    )

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse(
            "The initial response target is eight business hours "
            "[POL-TELEMETRY-RESPONSE v1.0]. The verified Standard entitlement "
            "makes that policy applicable [ENT-1002-TELEMETRY record]."
        )

    monkeypatch.setattr("servicing_agent.generation.httpx.post", fake_post)
    outcome = generator().generate(
        question="What initial response target applies to CASE-4824?",
        ticket=policy_ticket,
        entitlements=entitlements,
        retrieval=corpus_retrieval("POL-TELEMETRY-RESPONSE-v1.0"),
        history=policy_history,
    )

    assert outcome.mode == "llm"
    assert {citation.source_id for citation in outcome.citations} == {
        "POL-TELEMETRY-RESPONSE",
        "ENT-1002-TELEMETRY",
    }
    prompt = captured["json"]["messages"][1]["content"]
    unused = "Not supplied because it is not needed for this question."
    assert f"CONFIRMED CURRENT CRM FACTS:\n{unused}" in prompt
    assert f"AUTHORIZED SNOWFLAKE HISTORICAL CONTEXT:\n{unused}" in prompt
    assert "This is a policy-applicability question" in prompt
    assert "Do not add case history, monitoring, or escalation details" in prompt


def test_policy_answer_without_policy_citation_still_falls_back(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "servicing_agent.generation.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            "The verified Standard entitlement applies and the target is eight "
            "business hours [ENT-1002-TELEMETRY record]."
        ),
    )
    outcome = generator().generate(
        question="What initial response target applies to CASE-4824?",
        ticket={
            "ticket_id": "CASE-4824",
            "customer_id": "CUST-1002",
            "subject": "Telemetry latency",
            "status": "Monitoring",
            "priority": "Medium",
            "owner": "Tier 1 Support",
            "updated_at": "2026-11-02T18:20:00Z",
        },
        entitlements=[
            {
                "entitlement_id": "ENT-1002-TELEMETRY",
                "customer_id": "CUST-1002",
                "product_id": "PRD-TELEMETRY",
                "subscription_tier": "Standard",
                "status": "Active",
                "start_date": "2025-08-15",
                "end_date": None,
            }
        ],
        retrieval=corpus_retrieval("POL-TELEMETRY-RESPONSE-v1.0"),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert outcome.mode == "deterministic"
    assert outcome.fallback_reason is not None
    assert "applicable policy" in outcome.fallback_reason
    assert {citation.source_id for citation in outcome.citations} == {
        "POL-TELEMETRY-RESPONSE",
        "ENT-1002-TELEMETRY",
    }


def test_unsuccessful_reproduction_uncertainty_is_valid_llm_output(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "servicing_agent.generation.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            "The documented attempt did not reproduce the dashboard issue, but "
            "that result does not prove the issue did not occur "
            "[CASE-NOTE-4823 v1.0]."
        ),
    )
    outcome = generator().generate(
        question="Does CASE-4823 not being reproduced prove the issue did not occur?",
        ticket={
            "ticket_id": "CASE-4823",
            "customer_id": "CUST-1003",
            "subject": "Dashboard refresh issue",
            "status": "Awaiting Customer",
            "priority": "Low",
            "owner": "Tier 1 Support",
            "updated_at": "2026-11-04T16:45:00Z",
        },
        entitlements=[],
        retrieval=corpus_retrieval("CASE-NOTE-4823-v1.0"),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert outcome.mode == "llm"
    assert outcome.fallback_reason is None
    assert {citation.source_id for citation in outcome.citations} == {
        "CASE-NOTE-4823"
    }


def test_unsuccessful_reproduction_still_cannot_prove_nonoccurrence(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "servicing_agent.generation.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            "The issue did not occur because support could not reproduce it "
            "[CASE-NOTE-4823 v1.0]."
        ),
    )
    outcome = generator().generate(
        question="Does CASE-4823 not being reproduced prove the issue did not occur?",
        ticket={
            "ticket_id": "CASE-4823",
            "customer_id": "CUST-1003",
            "subject": "Dashboard refresh issue",
            "status": "Awaiting Customer",
            "priority": "Low",
            "owner": "Tier 1 Support",
            "updated_at": "2026-11-04T16:45:00Z",
        },
        entitlements=[],
        retrieval=corpus_retrieval("CASE-NOTE-4823-v1.0"),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert outcome.mode == "deterministic"
    assert outcome.fallback_reason == (
        "LLM output converted uncertain or historical evidence into an "
        "unsupported certainty or negation."
    )


def test_confirmed_certificate_root_cause_is_valid_llm_output(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "servicing_agent.generation.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            "The root cause of CASE-4825 was an expired connector certificate "
            "that blocked telemetry exports [CASE-NOTE-4825 v1.0]."
        ),
    )
    outcome = generator().generate(
        question="What was the root cause of CASE-4825?",
        ticket={
            "ticket_id": "CASE-4825",
            "customer_id": "CUST-1001",
            "subject": "Telemetry export backlog",
            "status": "Resolved",
            "priority": "High",
            "owner": "Tier 2 Support",
            "updated_at": "2026-11-01T12:40:00Z",
        },
        entitlements=[],
        retrieval=corpus_retrieval("CASE-NOTE-4825-v1.0"),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert outcome.mode == "llm"
    assert outcome.fallback_reason is None
    assert {citation.source_id for citation in outcome.citations} == {
        "CASE-NOTE-4825"
    }


def test_explicit_adjacent_root_cause_citation_reference_is_valid(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "servicing_agent.generation.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            "The root cause of CASE-4825 was an expired connector certificate "
            "that blocked telemetry exports. This root cause is supported by "
            "[CASE-NOTE-4825 v1.0]."
        ),
    )
    outcome = generator().generate(
        question="What was the root cause of CASE-4825?",
        ticket={
            "ticket_id": "CASE-4825",
            "customer_id": "CUST-1001",
            "subject": "Telemetry export backlog",
            "status": "Resolved",
            "priority": "High",
            "owner": "Tier 2 Support",
            "updated_at": "2026-11-01T12:40:00Z",
        },
        entitlements=[],
        retrieval=corpus_retrieval("CASE-NOTE-4825-v1.0"),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert outcome.mode == "llm"
    assert outcome.fallback_reason is None
    assert {citation.source_id for citation in outcome.citations} == {
        "CASE-NOTE-4825"
    }


def test_root_cause_claim_without_inline_citation_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "servicing_agent.generation.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            "The root cause of CASE-4825 was an expired connector certificate. "
            "Certificate rotation and replay completed successfully "
            "[CASE-NOTE-4825 v1.0]."
        ),
    )
    outcome = generator().generate(
        question="What was the root cause of CASE-4825?",
        ticket={
            "ticket_id": "CASE-4825",
            "customer_id": "CUST-1001",
            "subject": "Telemetry export backlog",
            "status": "Resolved",
            "priority": "High",
            "owner": "Tier 2 Support",
            "updated_at": "2026-11-01T12:40:00Z",
        },
        entitlements=[],
        retrieval=corpus_retrieval("CASE-NOTE-4825-v1.0"),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert outcome.mode == "deterministic"
    assert outcome.fallback_reason == (
        "LLM root-cause claim lacked an inline supporting citation."
    )


def test_history_selection_excludes_other_case_records_and_policy_history() -> None:
    records = (
        HistoryRecord(
            source_id="SF-CUSTOMER-ACTIVITY-ACT-001",
            source_table="CUSTOMER_ACTIVITY",
            customer_id="CUST-1001",
            case_id=None,
            event_at="2026-11-05T09:12:00Z",
            title="TELEMETRY_DELAY_OBSERVED",
            summary="CASE-4821 had delayed synchronization windows.",
        ),
        HistoryRecord(
            source_id="SF-CUSTOMER-ACTIVITY-ACT-4825-001",
            source_table="CUSTOMER_ACTIVITY",
            customer_id="CUST-1001",
            case_id=None,
            event_at="2026-11-01T12:30:00Z",
            title="TELEMETRY_BACKLOG_CLEARED",
            summary="For CASE-4825, the backlog cleared after certificate rotation.",
        ),
    )
    history_result = HistoryResult(
        records=records,
        backend="snowflake",
        status="available",
    )

    selected = _select_history_records(
        "What was the root cause of CASE-4825?",
        history_result,
        "explanation",
        5,
    )
    policy_selected = _select_history_records(
        "What initial response target applies to CASE-4825?",
        history_result,
        "unknown_information",
        5,
    )

    assert [record.source_id for record in selected] == [
        "SF-CUSTOMER-ACTIVITY-ACT-4825-001"
    ]
    assert policy_selected == []


def test_negated_root_cause_statement_is_not_a_positive_assertion(
    monkeypatch,
) -> None:
    content = (
        "CASE-4821 was escalated because three consecutive synchronization "
        "windows exceeded the latency baseline "
        "[SF-CASE-HISTORY-HIST-001 record]. The customer observed telemetry "
        "arriving forty to sixty minutes late "
        "[CALL-4821-20261105 v1.0]. Connector retries were "
        "observed, but no definitive root cause was established "
        "[CASE-NOTE-4821 v1.0]."
    )
    monkeypatch.setattr(
        "servicing_agent.generation.httpx.post",
        lambda *args, **kwargs: FakeResponse(content),
    )
    multi_history = HistoryResult(
        records=(
            HistoryRecord(
                source_id="SF-CASE-HISTORY-HIST-001",
                source_table="CASE_HISTORY",
                customer_id="CUST-1001",
                case_id="CASE-4821",
                event_at="2026-11-06T18:00:00Z",
                title="UNDER_INVESTIGATION",
                summary=(
                    "CASE-4821 was escalated after three consecutive "
                    "synchronization windows exceeded the latency baseline."
                ),
            ),
            HistoryRecord(
                source_id="SF-SUPPORT-INTERACTIONS-INT-001",
                source_table="SUPPORT_INTERACTIONS",
                customer_id="CUST-1001",
                case_id="CASE-4821",
                event_at="2026-11-05T21:15:00Z",
                title="PHONE",
                summary=(
                    "The customer reported telemetry arriving forty to sixty "
                    "minutes late during evening windows."
                ),
            ),
        ),
        backend="snowflake",
        status="available",
    )

    outcome = generator().generate(
        question=(
            "Using the case notes and phone conversation, summarize why CASE-4821 "
            "was escalated and what delay the customer observed."
        ),
        ticket=ticket(),
        entitlements=[],
        retrieval=corpus_retrieval(
            "CASE-NOTE-4821-v1.0", "CALL-4821-20261105-v1.0"
        ),
        history=multi_history,
    )

    assert outcome.mode == "llm"
    assert outcome.fallback_reason is None
    assert len(outcome.citations) == 3


def test_uncertain_receipt_prompt_excludes_redundant_history(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse(
            "The evidence does not confirm whether the requested connector "
            "diagnostic logs were received [CASE-DIAG-4821 v1.0]."
        )

    monkeypatch.setattr("servicing_agent.generation.httpx.post", fake_post)
    outcome = generator().generate(
        question="Were the requested connector diagnostic logs received for CASE-4821?",
        ticket=ticket(),
        entitlements=[],
        retrieval=corpus_retrieval("CASE-DIAG-4821-v1.0"),
        history=history(),
    )

    assert outcome.mode == "llm"
    assert {citation.source_id for citation in outcome.citations} == {
        "CASE-DIAG-4821"
    }
    prompt = captured["json"]["messages"][1]["content"]
    assert "Never rewrite 'receipt is not recorded' as 'not received.'" in prompt
    assert "[SF-CASE-HISTORY-HIST-001 record]" not in prompt


def test_confirmed_resolution_requires_claim_local_citations(monkeypatch) -> None:
    resolution_history = HistoryResult(
        records=(
            HistoryRecord(
                source_id="SF-CASE-HISTORY-HIST-4825-001",
                source_table="CASE_HISTORY",
                customer_id="CUST-1001",
                case_id="CASE-4825",
                event_at="2026-11-01T12:40:00Z",
                title="RESOLVED",
                summary=(
                    "Certificate rotation and queued export replay completed, "
                    "the backlog cleared, and CASE-4825 was marked Resolved."
                ),
            ),
            HistoryRecord(
                source_id="SF-SUPPORT-INTERACTIONS-INT-4825-001",
                source_table="SUPPORT_INTERACTIONS",
                customer_id="CUST-1001",
                case_id="CASE-4825",
                event_at="2026-11-01T11:30:00Z",
                title="INTERNAL",
                summary="Tier 2 began certificate troubleshooting.",
            ),
        ),
        backend="snowflake",
        status="available",
    )
    responses = iter(
        (
            FakeResponse(
                "Tier 2 confirmed an expired connector certificate blocked "
                "telemetry exports [CASE-NOTE-4825 v1.0]. Certificate rotation "
                "and queued export replay completed, the backlog cleared, and "
                "the case was marked Resolved "
                "[SF-CASE-HISTORY-HIST-4825-001 record]."
            ),
            FakeResponse(
                "Tier 2 confirmed an expired connector certificate. The case was "
                "marked Resolved [CASE-NOTE-4825 v1.0] "
                "[SF-SUPPORT-INTERACTIONS-INT-4825-001 record]."
            ),
        )
    )
    captured: list[dict] = []

    def fake_post(url, **kwargs):
        captured.append(kwargs)
        return next(responses)

    monkeypatch.setattr("servicing_agent.generation.httpx.post", fake_post)
    arguments = {
        "question": "What was confirmed and completed to resolve CASE-4825?",
        "ticket": {
            "ticket_id": "CASE-4825",
            "customer_id": "CUST-1001",
            "subject": "Telemetry export backlog",
            "status": "Resolved",
            "priority": "High",
            "owner": "Tier 2 Support",
            "updated_at": "2026-11-01T12:40:00Z",
        },
        "entitlements": [],
        "retrieval": corpus_retrieval("CASE-NOTE-4825-v1.0"),
        "history": resolution_history,
    }

    supported = generator().generate(**arguments)
    stacked = generator().generate(**arguments)

    assert supported.mode == "llm", supported.fallback_reason
    assert stacked.mode == "deterministic"
    assert stacked.fallback_reason == "LLM output contained an unrecognized citation."
    prompt = captured[0]["json"]["messages"][1]["content"]
    assert "[SF-CASE-HISTORY-HIST-4825-001 record]" in prompt
    assert "[SF-SUPPORT-INTERACTIONS-INT-4825-001 record]" not in prompt
    assert "Never stack source labels after a later claim" in prompt


def test_confirmed_resolution_cannot_omit_confirmed_cause(monkeypatch) -> None:
    monkeypatch.setattr(
        "servicing_agent.generation.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            "Certificate rotation and queued export replay completed, the backlog "
            "cleared, and CASE-4825 was marked Resolved "
            "[CASE-NOTE-4825 v1.0]."
        ),
    )
    outcome = generator().generate(
        question="What was confirmed and completed to resolve CASE-4825?",
        ticket={
            "ticket_id": "CASE-4825",
            "customer_id": "CUST-1001",
            "subject": "Telemetry export backlog",
            "status": "Resolved",
            "priority": "High",
            "owner": "Tier 2 Support",
            "updated_at": "2026-11-01T12:40:00Z",
        },
        entitlements=[],
        retrieval=corpus_retrieval("CASE-NOTE-4825-v1.0"),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert outcome.mode == "deterministic"
    assert outcome.fallback_reason == (
        "LLM output omitted the evidence-confirmed cause."
    )


def test_supported_blocking_condition_is_a_cited_cause(monkeypatch) -> None:
    monkeypatch.setattr(
        "servicing_agent.generation.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            "An expired connector certificate blocked telemetry exports "
            "[CASE-NOTE-4825 v1.0]. Certificate rotation and queued export replay "
            "completed, the backlog cleared, and CASE-4825 was marked Resolved "
            "[CASE-NOTE-4825 v1.0]."
        ),
    )
    outcome = generator().generate(
        question="What was confirmed and completed to resolve CASE-4825?",
        ticket={
            "ticket_id": "CASE-4825",
            "customer_id": "CUST-1001",
            "subject": "Telemetry export backlog",
            "status": "Resolved",
            "priority": "High",
            "owner": "Tier 2 Support",
            "updated_at": "2026-11-01T12:40:00Z",
        },
        entitlements=[],
        retrieval=corpus_retrieval("CASE-NOTE-4825-v1.0"),
        history=HistoryResult(backend="mock", status="empty"),
    )

    assert outcome.mode == "llm"
    assert outcome.fallback_reason is None
