from __future__ import annotations

import json
import re
from pathlib import Path

from servicing_agent.auth import DEMO_USERS
from servicing_agent.database import Database, TICKETS
from servicing_agent.service import ChatService


PROJECT_ROOT = Path(__file__).parents[1]


def _corpus() -> list[dict[str, object]]:
    return json.loads(
        (PROJECT_ROOT / "src/servicing_agent/corpus/knowledge.json").read_text(
            encoding="utf-8"
        )
    )


def _golden() -> list[dict[str, object]]:
    return json.loads(
        (PROJECT_ROOT / "evaluations/golden_questions.json").read_text(
            encoding="utf-8"
        )
    )


def test_case_linked_corpus_records_use_canonical_crm_identifiers() -> None:
    ticket_customers = {ticket[0]: ticket[1] for ticket in TICKETS}

    for document in _corpus():
        case_id = document["case_id"]
        if case_id is None:
            continue
        assert case_id in ticket_customers, document["document_id"]
        assert document["customer_id"] == ticket_customers[case_id], document[
            "document_id"
        ]


def test_case_4821_preserves_unknowns_and_case_4825_confirms_completion() -> None:
    documents = {item["document_id"]: item for item in _corpus()}
    ongoing = str(documents["CASE-NOTE-4821-v1.0"]["text"])
    diagnostics = str(documents["CASE-DIAG-4821-v1.0"]["text"])
    resolved = str(documents["CASE-NOTE-4825-v1.0"]["text"])

    assert "does not confirm that either action was completed" in ongoing
    assert "No confirmed root cause or resolution date" in ongoing
    assert "Receipt of the requested logs is not recorded" in diagnostics
    assert "not evidence that the customer declined" in diagnostics
    assert "certificate rotation was completed" in resolved
    assert "queued export replay completed" in resolved
    assert "must not be inferred for CASE-4821" in resolved


def test_snowflake_seed_is_idempotent_and_keeps_reader_role_read_only() -> None:
    sql = (
        PROJECT_ROOT / "snowflake/seed_synthetic_history.sql"
    ).read_text(encoding="utf-8")
    upper = sql.upper()

    assert "MERGE INTO CUSTOMER_ACTIVITY" in upper
    assert "MERGE INTO CASE_HISTORY" in upper
    assert "MERGE INTO SUPPORT_INTERACTIONS" in upper
    assert "WHEN MATCHED THEN UPDATE" in upper
    assert "WHEN NOT MATCHED THEN INSERT" in upper
    assert "'HIST-001', 'CASE-4821', 'CUST-1001'" in sql
    assert "'HIST-4825-001', 'CASE-4825', 'CUST-1001'" in sql
    customer_merge = upper.split("MERGE INTO CUSTOMER_ACTIVITY", 1)[1].split(
        "MERGE INTO CASE_HISTORY", 1
    )[0]
    assert "PRODUCT_ID = SOURCE.PRODUCT_ID" in customer_merge
    assert "EVENT_TYPE = SOURCE.EVENT_TYPE" in customer_merge
    assert "EVENT_DESCRIPTION = SOURCE.EVENT_DESCRIPTION" in customer_merge
    assert "ACTIVITY_TYPE" not in customer_merge
    assert "'ACT-4822-001', 'CUST-1002', 'PRD-BILLING'" in sql
    assert "'ACT-4823-001', 'CUST-1003', 'PRD-DASHBOARD'" in sql
    support_merge = upper.split("MERGE INTO SUPPORT_INTERACTIONS", 1)[1].split(
        "COMMIT", 1
    )[0]
    assert (
        "TRANSCRIPT_ID = COALESCE(SOURCE.TRANSCRIPT_ID, TARGET.TRANSCRIPT_ID)"
        in support_merge
    )
    assert "INTERACTION_TYPE = SOURCE.INTERACTION_TYPE" in support_merge
    assert "CHANNEL" not in support_merge
    assert "'CALL-4821-20261105', 'PHONE'" in sql
    assert "TO_TIMESTAMP_TZ" not in upper
    assert upper.count("TO_TIMESTAMP_NTZ(") == 6
    assert re.search(r"(?im)^\s*GRANT\b", sql) is None
    assert "SERVICING_HISTORY_READER" in upper


def test_expanded_golden_scenarios_declare_grounding_contracts() -> None:
    scenarios = {item["id"]: item for item in _golden()}
    expected = {
        "case-4821-unresolved-investigation",
        "case-4821-diagnostic-logs-unknown",
        "resolved-telemetry-incident",
        "resolved-billing-service-request",
        "dashboard-standard-policy-applicability",
        "unauthorized-cross-customer-dashboard",
    }

    assert expected.issubset(scenarios)
    for scenario_id in expected:
        scenario = scenarios[scenario_id]
        assert "expected_facts" in scenario
        assert "unsupported_claims" in scenario
        assert "required_evidence" in scenario
        assert scenario["authorization_expectation"] in {"allowed", "denied"}
        assert scenario["answer_quality_checks"]


def test_resolved_and_unknown_cases_are_grounded_and_tenant_isolated(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "expanded-scenarios.db")
    database.initialize()
    service = ChatService(database)

    resolved = service.chat(
        "What was confirmed and completed to resolve CASE-4825?",
        DEMO_USERS["demo-agent-alex"],
    )
    unknown = service.chat(
        "Were the requested connector diagnostic logs received for CASE-4821?",
        DEMO_USERS["demo-agent-alex"],
    )
    denied = service.chat(
        "What diagnostic information is missing for CASE-4823?",
        DEMO_USERS["demo-agent-alex"],
    )

    assert "queued export replay completed" in resolved.reply
    assert any(item.source_id == "CASE-NOTE-4825" for item in resolved.citations)
    assert "Receipt of the requested logs is not recorded" in unknown.reply
    assert any(item.source_id == "CASE-DIAG-4821" for item in unknown.citations)
    assert "not authorized" in denied.reply.lower()
    assert denied.citations == []
    assert denied.retrieval_debug is None
