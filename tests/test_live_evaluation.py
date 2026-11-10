from __future__ import annotations

import json
from pathlib import Path

from servicing_agent.benchmark_ollama import (
    evaluate_live_quality,
    evaluate_live_response,
    load_live_contracts,
    reevaluate_saved_report,
    summarize_live_runs,
    write_live_report,
)


def dataset_path() -> Path:
    return Path(__file__).parents[1] / "evaluations" / "golden_questions.json"


def test_live_contracts_are_exactly_demo_scenarios_two_through_six() -> None:
    contracts = load_live_contracts(dataset_path())

    assert [scenario.key for scenario, _contract in contracts] == [
        "multi_source",
        "premium_policy",
        "standard_policy",
        "unknown_diagnostics",
        "resolved_incident",
    ]
    assert all(
        scenario.prompt == contract["question"]
        for scenario, contract in contracts
    )


def test_live_response_uses_existing_golden_contract_and_citation_checks() -> None:
    scenario, contract = load_live_contracts(dataset_path())[0]
    reply = (
        "CASE-4821 was escalated after three consecutive synchronization windows "
        "[CASE-NOTE-4821 v1.0]. The customer observed telemetry forty to sixty "
        "minutes late [CALL-4821-20261105 v1.0]."
    )
    body = {
        "reply": reply,
        "intent": "complex_request",
        "generation_mode": "llm",
        "evidence_status": "sufficient",
        "citations": [
            {
                "source_id": "CASE-NOTE-4821",
                "document_version": "1.0",
                "title": "CASE-4821 escalation note",
                "document_type": "case_note",
                "effective_date": "2026-11-06",
                "snippet": (
                    "CASE-4821 was escalated after three consecutive "
                    "synchronization windows."
                ),
                "citation_label": "[CASE-NOTE-4821 v1.0]",
            },
            {
                "source_id": "CALL-4821-20261105",
                "document_version": "1.0",
                "title": "CASE-4821 customer phone conversation",
                "document_type": "call_transcript",
                "effective_date": "2026-11-05",
                "snippet": "Telemetry appeared forty to sixty minutes late.",
                "citation_label": "[CALL-4821-20261105 v1.0]",
            },
        ],
        "retrieval_debug": {
            "effective_mode": "hybrid",
            "selected_document_ids": [
                "CASE-NOTE-4821-v1.0",
                "CALL-4821-20261105-v1.0",
            ],
            "latency_ms": 10.0,
            "fallback_reason": None,
        },
        "history_debug": {"status": "available", "latency_ms": 100.0},
        "generation_debug": {
            "latency_ms": 1000.0,
            "fallback_reason": None,
            "finish_reason": "stop",
            "output_tokens": 50,
        },
        "total_latency_ms": 1200.0,
    }

    result = evaluate_live_response(scenario, contract, body, 1)

    assert result["llm_success"] is True
    assert result["citation_valid"] is True
    assert result["citation_relevance"] is True
    assert result["semantic_answer_correct"] is True
    assert result["factual_completeness"] is True
    assert result["claim_grounding"] is True
    assert result["formatting_compliance"] is True
    assert result["answer_correct"] is True
    assert result["failure_categories"] == []


def _quality(contract: dict, reply: str) -> dict:
    return evaluate_live_quality(
        contract,
        reply,
        intent_correct=True,
        evidence_status_correct=True,
        citation_valid=True,
        citation_relevance=True,
        required_evidence_present=True,
    )


def test_live_quality_accepts_uncertainty_paraphrase_without_answer_heading() -> None:
    contracts = dict(
        (scenario.key, contract)
        for scenario, contract in load_live_contracts(dataset_path())
    )
    result = _quality(
        contracts["unknown_diagnostics"],
        "The evidence does not confirm whether the requested connector "
        "diagnostic logs were received. [CASE-DIAG-4821 v1.0]",
    )

    assert result["semantic_answer_correctness"] is True
    assert result["factual_completeness"] is True
    assert result["claim_grounding"] is True
    assert result["formatting_compliance"] is True


def test_live_quality_rejects_missing_material_fact_and_unsupported_certainty() -> None:
    contracts = dict(
        (scenario.key, contract)
        for scenario, contract in load_live_contracts(dataset_path())
    )
    missing = _quality(
        contracts["unknown_diagnostics"],
        "The available evidence is insufficient. [CASE-DIAG-4821 v1.0]",
    )
    unsupported = _quality(
        contracts["unknown_diagnostics"],
        "The customer declined to provide the logs. [CASE-DIAG-4821 v1.0]",
    )

    assert missing["semantic_answer_correctness"] is False
    assert missing["factual_completeness"] is False
    assert unsupported["semantic_answer_correctness"] is False
    assert unsupported["claim_grounding"] is False


def test_live_quality_rejects_negated_completed_remediation() -> None:
    contracts = dict(
        (scenario.key, contract)
        for scenario, contract in load_live_contracts(dataset_path())
    )
    result = _quality(
        contracts["resolved_incident"],
        "An expired connector certificate blocked exports. The certificate "
        "rotation was not completed, although the queued export replay completed "
        "and the backlog cleared. [CASE-NOTE-4825 v1.0]",
    )

    assert result["semantic_answer_correctness"] is False
    assert result["factual_completeness"] is False
    assert result["semantic_fact_contradictions"][
        "certificate rotation was completed"
    ] is True


def test_live_quality_keeps_invalid_citations_separate_and_failing() -> None:
    contracts = dict(
        (scenario.key, contract)
        for scenario, contract in load_live_contracts(dataset_path())
    )
    contract = contracts["unknown_diagnostics"]
    result = evaluate_live_quality(
        contract,
        "The evidence does not confirm whether the requested connector "
        "diagnostic logs were received. [UNKNOWN record]",
        intent_correct=True,
        evidence_status_correct=True,
        citation_valid=False,
        citation_relevance=False,
        required_evidence_present=False,
    )

    assert result["semantic_answer_correctness"] is True
    assert result["factual_completeness"] is True
    assert result["citation_validity"] is False
    assert result["claim_grounding"] is False
    assert result["quality_pass"] is False


def test_saved_live_artifact_is_reevaluated_without_integrations() -> None:
    artifact = dataset_path().parent / "live_ollama" / "latest.json"
    saved = json.loads(artifact.read_text(encoding="utf-8"))

    revised = reevaluate_saved_report(saved, dataset_path())
    overall = revised["summary"]["overall"]

    assert revised["rescored_without_live_calls"] is True
    assert overall["semantic_answer_correctness_rate"] == 1.0
    assert overall["factual_completeness_rate"] == 1.0
    assert overall["claim_grounding_rate"] == 1.0
    assert overall["citation_validity_rate"] == 1.0
    assert overall["formatting_compliance_rate"] == 1.0
    assert overall["legacy_exact_contract_rate"] == 0.6
    assert len(
        revised["score_revision"]["grounding_contract_failures"]
    ) == 5
    assert len(revised["score_revision"]["formatting_failures"]) == 10

    repeated = reevaluate_saved_report(revised, dataset_path())
    assert repeated["summary"]["overall"]["legacy_exact_contract_rate"] == 0.6
    assert len(
        repeated["score_revision"]["grounding_contract_failures"]
    ) == 5


def test_live_summary_and_timestamped_artifacts_preserve_individual_runs(
    tmp_path: Path,
) -> None:
    runs = []
    for scenario_key in (
        "multi_source",
        "premium_policy",
        "standard_policy",
        "unknown_diagnostics",
        "resolved_incident",
    ):
        for run_number, latency in enumerate((100.0, 200.0), start=1):
            runs.append(
                {
                    "scenario_key": scenario_key,
                    "run_number": run_number,
                    "llm_success": True,
                    "fallback": False,
                    "fallback_reason": None,
                    "finish_reason": "stop",
                    "output_tokens": 40,
                    "citation_valid": True,
                    "citation_relevance": True,
                    "semantic_answer_correct": True,
                    "factual_completeness": True,
                    "claim_grounding": True,
                    "formatting_compliance": True,
                    "quality_pass": True,
                    "quality_failure_dimensions": [],
                    "legacy_exact_contract_pass": (
                        scenario_key != "unknown_diagnostics"
                    ),
                    "legacy_grounding_contract_pass": True,
                    "legacy_failure_categories": (
                        ["expected_text"]
                        if scenario_key == "unknown_diagnostics"
                        else []
                    ),
                    "generation_latency_ms": latency,
                    "total_latency_ms": latency + 50,
                }
            )

    summary = summarize_live_runs(runs)
    report = {
        "schema_version": 2,
        "evaluation_type": "live_ollama",
        "generated_at_utc": "2026-11-10T12:34:56Z",
        "summary": summary,
        "runs": runs,
    }
    timestamped, latest = write_live_report(report, tmp_path)

    assert summary["overall"]["llm_success_rate"] == 1.0
    assert summary["overall"]["semantic_answer_correctness_rate"] == 1.0
    assert summary["overall"]["legacy_exact_contract_rate"] == 0.8
    assert summary["overall"]["generation_latency"] == {
        "p50_ms": 150.0,
        "p95_ms": 200.0,
    }
    assert timestamped.name == "live_ollama_20261110T123456Z.json"
    assert json.loads(latest.read_text(encoding="utf-8"))["runs"] == runs
