from __future__ import annotations

import json

import pytest

from servicing_agent.evaluation_dashboard import (
    EvaluationReportError,
    comparison_rows,
    latency_rows,
    load_evaluation_report,
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
