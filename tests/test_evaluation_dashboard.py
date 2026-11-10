from __future__ import annotations

import json

import pytest

from servicing_agent.evaluation_dashboard import (
    EvaluationReportError,
    comparison_rows,
    latency_rows,
    live_evaluation_status,
    live_scenario_detail_rows,
    load_evaluation_report,
    load_live_ollama_report,
    live_scenario_rows,
    scenario_rows,
)


def sample_report() -> dict:
    return {
        "schema_version": 2,
        "generated_at_utc": "2026-11-08T12:00:00Z",
        "scenario_count": 1,
        "runs_per_scenario": 1,
        "model": "mock",
        "rrf_k": 60,
        "generation_evaluation": {"declared_effective": "deterministic"},
        "business_outcomes": {},
        "metric_definitions": {},
        "modes": {
            "hybrid": {
                "task_success_rate": 1.0,
                "recall_at_5": 1.0,
                "precision_at_3": 0.3333,
                "grounded_answer_accuracy": 1.0,
                "policy_applicability_accuracy": 1.0,
                "authorization_safety_rate": 1.0,
                "action_safety_rate": 1.0,
                "escalation_correctness": 1.0,
                "recovery_success_rate": 1.0,
                "failed_scenario_runs": 0,
                "retrieval_latency_p50_ms": 1.0,
                "retrieval_latency_p95_ms": 2.0,
                "tool_execution_latency_p50_ms": 3.0,
                "tool_execution_latency_p95_ms": 4.0,
                "end_to_end_latency_p50_ms": 5.0,
                "end_to_end_latency_p95_ms": 6.0,
                "failure_categories": {},
                "examples": [
                    {
                        "id": "example",
                        "scenario_type": "chat",
                        "category": "exact_match",
                        "success": True,
                        "failure_reason": None,
                        "end_to_end_latency_p50_ms": 5.0,
                    }
                ],
            }
        },
    }


def test_dashboard_loads_actual_report_and_builds_tables(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(sample_report()), encoding="utf-8")

    report = load_evaluation_report(path)

    assert comparison_rows(report)[0]["Task success"] == 1.0
    assert latency_rows(report)[0]["End-to-end P95 ms"] == 6.0
    assert scenario_rows(report, "hybrid")[0]["Outcome"] == "PASS"


def test_dashboard_rejects_missing_or_old_report(tmp_path) -> None:
    with pytest.raises(EvaluationReportError, match="No evaluation report"):
        load_evaluation_report(tmp_path / "missing.json")

    old = tmp_path / "old.json"
    old.write_text(json.dumps({"modes": {"lexical": {}}}), encoding="utf-8")
    with pytest.raises(EvaluationReportError, match="Milestone 4 schema"):
        load_evaluation_report(old)


def test_dashboard_loads_live_ollama_report_without_running_integrations(
    tmp_path,
) -> None:
    path = tmp_path / "latest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "evaluation_type": "live_ollama",
                "summary": {
                    "overall": {},
                    "scenarios": {
                        "unknown_diagnostics": {
                            "run_count": 5,
                            "llm_success_rate": 1.0,
                            "fallback_rate": 0.0,
                            "citation_validity_rate": 1.0,
                            "citation_relevance_rate": 1.0,
                            "semantic_answer_correctness_rate": 1.0,
                            "factual_completeness_rate": 0.8,
                            "claim_grounding_rate": 1.0,
                            "formatting_compliance_rate": 1.0,
                            "legacy_exact_contract_rate": 0.0,
                            "generation_latency": {
                                "p50_ms": 10.0,
                                "p95_ms": 20.0,
                            },
                            "total_latency": {
                                "p50_ms": 30.0,
                                "p95_ms": 40.0,
                            },
                            "finish_reasons": {"stop": 5},
                            "fallback_reasons": {},
                            "failures": [],
                        }
                    },
                },
                "runs": [],
            }
        ),
        encoding="utf-8",
    )

    report = load_live_ollama_report(path)
    row = live_scenario_rows(report)[0]
    detail = live_scenario_detail_rows(report)[0]

    assert row["Scenario"] == "Unknown Diagnostics"
    assert row["LLM success"] == "100.0%"
    assert row["Fallback rate"] == "0.0%"
    assert row["Semantic correctness"] == "100.0%"
    assert row["Factual completeness"] == "80.0%"
    assert row["Claim grounding"] == "100.0%"
    assert row["Generation P95 (s)"] == "0.02"
    assert row["Total P95 (s)"] == "0.04"
    assert detail["Citation validity"] == "100.0%"
    assert detail["Formatting compliance"] == "100.0%"
    assert detail["Legacy exact contract"] == "0.0%"


def test_live_scenario_names_are_readable() -> None:
    metrics = {
        "run_count": 5,
        "llm_success_rate": 1.0,
        "fallback_rate": 0.0,
        "semantic_answer_correctness_rate": 1.0,
        "factual_completeness_rate": 1.0,
        "claim_grounding_rate": 1.0,
        "generation_latency": {"p50_ms": 1000.0, "p95_ms": 2000.0},
        "total_latency": {"p50_ms": 3000.0, "p95_ms": 4000.0},
    }
    report = {
        "summary": {
            "scenarios": {
                key: metrics
                for key in (
                    "multi_source",
                    "premium_policy",
                    "standard_policy",
                    "unknown_diagnostics",
                    "resolved_incident",
                )
            }
        }
    }

    assert [row["Scenario"] for row in live_scenario_rows(report)] == [
        "Multi-source Investigation",
        "Premium Policy",
        "Standard Policy",
        "Unknown Diagnostics",
        "Confirmed Resolution",
    ]


def test_live_success_message_requires_measured_zero_fallbacks_and_failures() -> None:
    success = {
        "summary": {
            "overall": {
                "run_count": 25,
                "fallback_rate": 0.0,
                "failure_count": 0,
            }
        }
    }
    fallback = {
        "summary": {
            "overall": {
                "run_count": 25,
                "fallback_rate": 0.04,
                "failure_count": 0,
            }
        }
    }
    revised_failure = {
        "summary": {
            "overall": {
                "run_count": 25,
                "fallback_rate": 0.0,
                "failure_count": 1,
            }
        }
    }

    assert live_evaluation_status(success) == (
        "success",
        "No fallbacks or failures under the revised live evaluation checks. "
        "Legacy strict-contract mismatches are reported separately.",
    )
    assert live_evaluation_status(fallback)[0] == "warning"
    assert live_evaluation_status(revised_failure)[0] == "warning"
    assert live_evaluation_status({"summary": {"overall": {}}})[0] == "warning"


def test_dashboard_rejects_missing_or_invalid_live_report(tmp_path) -> None:
    with pytest.raises(EvaluationReportError, match="No live Ollama"):
        load_live_ollama_report(tmp_path / "missing.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(EvaluationReportError, match="expected schema"):
        load_live_ollama_report(invalid)
