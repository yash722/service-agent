from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EvaluationReportError(ValueError):
    pass


def load_evaluation_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationReportError(
            f"No evaluation report found at {report_path}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise EvaluationReportError(
            f"The evaluation report at {report_path} is not valid JSON."
        ) from exc
    if report.get("schema_version") != 2 or not report.get("modes"):
        raise EvaluationReportError(
            "The evaluation report is missing the Milestone 4 schema or modes."
        )
    return report


def comparison_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "Mode": mode.title(),
            "Task success": metrics["task_success_rate"],
            "Recall@5": metrics["recall_at_5"],
            "Precision@3": metrics["precision_at_3"],
            "Groundedness": metrics["grounded_answer_accuracy"],
            "Claim-citation relevance": metrics.get(
                "claim_citation_relevance", 0.0
            ),
            "Policy": metrics["policy_applicability_accuracy"],
            "Authorization": metrics["authorization_safety_rate"],
            "Action safety": metrics["action_safety_rate"],
            "Escalation": metrics["escalation_correctness"],
            "Recovery": metrics["recovery_success_rate"],
            "Failures": metrics["failed_scenario_runs"],
        }
        for mode, metrics in report["modes"].items()
    ]


def latency_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "Mode": mode.title(),
            "Retrieval P50 ms": metrics["retrieval_latency_p50_ms"],
            "Retrieval P95 ms": metrics["retrieval_latency_p95_ms"],
            "Tool P50 ms": metrics["tool_execution_latency_p50_ms"],
            "Tool P95 ms": metrics["tool_execution_latency_p95_ms"],
            "End-to-end P50 ms": metrics["end_to_end_latency_p50_ms"],
            "End-to-end P95 ms": metrics["end_to_end_latency_p95_ms"],
        }
        for mode, metrics in report["modes"].items()
    ]


def scenario_rows(report: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    metrics = report["modes"][mode]
    return [
        {
            "Scenario": example["id"],
            "Type": example["scenario_type"],
            "Category": example["category"],
            "Outcome": "PASS" if example["success"] else "FAIL",
            "Failure": example.get("failure_reason") or "",
            "P50 ms": example["end_to_end_latency_p50_ms"],
        }
        for example in metrics["examples"]
    ]


def render_evaluation_dashboard(
    st: Any,
    report_path: str | Path,
    *,
    evaluation_modes: tuple[str, ...] = ("lexical", "semantic", "hybrid"),
    evaluation_runs: int = 3,
) -> None:
    st.subheader("Measured evaluation dashboard")
    st.caption(
        "Metrics below are loaded from a repeatable local evaluation run over "
        "synthetic scenarios. They are not production business outcomes."
    )
    try:
        report = load_evaluation_report(report_path)
    except EvaluationReportError as exc:
        st.warning(str(exc))
        modes = " ".join(evaluation_modes)
        st.code(
            f"python -m servicing_agent.evaluation --modes {modes} "
            f"--runs {evaluation_runs} --output {report_path}",
            language="powershell",
        )
        return

    generation = report["generation_evaluation"]
    if generation["declared_effective"] == "deterministic":
        st.info(
            "Answer evaluation: deterministic evidence generator. No LLM output "
            "is represented in these scores."
        )
    elif generation["declared_effective"].startswith("deterministic_fallback"):
        st.warning(
            "Configured-LLM evaluation was requested, but credentials were absent; "
            "the measured run used deterministic fallback."
        )
    else:
        st.info(
            "Configured LLM evaluation was enabled. Check each mode's actual "
            "generation-mode counts for fallback usage."
        )

    preferred_mode = "hybrid" if "hybrid" in report["modes"] else next(
        iter(report["modes"])
    )
    headline = report["modes"][preferred_mode]
    first, second, third, fourth = st.columns(4)
    first.metric("Task success", f"{headline['task_success_rate']:.1%}")
    second.metric("Groundedness", f"{headline['grounded_answer_accuracy']:.1%}")
    third.metric("Action safety", f"{headline['action_safety_rate']:.1%}")
    fourth.metric("Scenario failures", headline["failed_scenario_runs"])
    st.caption(
        f"Headline mode: {preferred_mode}. Generated {report['generated_at_utc']} · "
        f"{report['scenario_count']} scenarios × {report['runs_per_scenario']} runs."
    )

    st.markdown("#### Quality comparison")
    st.dataframe(comparison_rows(report), width="stretch", hide_index=True)
    st.markdown("#### Measured latency")
    st.dataframe(latency_rows(report), width="stretch", hide_index=True)

    selected_mode = st.selectbox(
        "Scenario outcomes for mode",
        list(report["modes"]),
        index=list(report["modes"]).index(preferred_mode),
        key="evaluation_mode_detail",
    )
    st.dataframe(
        scenario_rows(report, selected_mode),
        width="stretch",
        hide_index=True,
    )
    failures = report["modes"][selected_mode]["failure_categories"]
    if failures:
        st.error(f"Measured failure categories: {failures}")
    else:
        st.success("No scenario failures were measured for this mode in this run.")

    st.warning(
        "Not measured: CSAT, CPRR, production ROI, case deflection, or any other "
        "production business KPI. Demo business narratives are illustrative only."
    )
    with st.expander("Evaluation metadata and metric definitions"):
        st.json(
            {
                "generation_evaluation": generation,
                "business_outcomes": report["business_outcomes"],
                "metric_definitions": report["metric_definitions"],
                "model": report["model"],
                "rrf_k": report["rrf_k"],
            }
        )
