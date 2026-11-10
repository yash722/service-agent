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


def load_live_ollama_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationReportError(
            f"No live Ollama evaluation report found at {report_path}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise EvaluationReportError(
            f"The live Ollama report at {report_path} is not valid JSON."
        ) from exc
    if (
        report.get("schema_version") not in {1, 2}
        or report.get("evaluation_type") != "live_ollama"
        or "summary" not in report
        or "runs" not in report
    ):
        raise EvaluationReportError(
            "The live Ollama report is missing the expected schema."
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


def _rate_text(value: float | None) -> str:
    return "Not measured" if value is None else f"{value:.1%}"


def _latency_text(value: float | None) -> str | float:
    return "Not measured" if value is None else value


LIVE_SCENARIO_NAMES = {
    "multi_source": "Multi-source Investigation",
    "premium_policy": "Premium Policy",
    "standard_policy": "Standard Policy",
    "unknown_diagnostics": "Unknown Diagnostics",
    "resolved_incident": "Confirmed Resolution",
}


def _live_scenario_name(scenario_key: str) -> str:
    return LIVE_SCENARIO_NAMES.get(
        scenario_key, scenario_key.replace("_", " ").title()
    )


def _latency_seconds_text(value: float | None) -> str:
    if value is None:
        return "Not measured"
    return f"{value / 1000:.2f}"


def _failure_counts(failures: list[dict[str, Any]]) -> dict[str, int]:
    categories: dict[str, int] = {}
    for failure in failures:
        for category in failure.get("categories", []):
            categories[category] = categories.get(category, 0) + 1
    return categories


def live_scenario_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_key, metrics in report["summary"]["scenarios"].items():
        rows.append(
            {
                "Scenario": _live_scenario_name(scenario_key),
                "Runs": metrics["run_count"],
                "LLM success": _rate_text(metrics.get("llm_success_rate")),
                "Fallback rate": _rate_text(metrics.get("fallback_rate")),
                "Semantic correctness": _rate_text(
                    metrics.get("semantic_answer_correctness_rate")
                ),
                "Factual completeness": _rate_text(
                    metrics.get("factual_completeness_rate")
                ),
                "Claim grounding": _rate_text(
                    metrics.get("claim_grounding_rate")
                ),
                "Generation P50 (s)": _latency_seconds_text(
                    metrics["generation_latency"].get("p50_ms")
                ),
                "Generation P95 (s)": _latency_seconds_text(
                    metrics["generation_latency"].get("p95_ms")
                ),
                "Total P50 (s)": _latency_seconds_text(
                    metrics["total_latency"].get("p50_ms")
                ),
                "Total P95 (s)": _latency_seconds_text(
                    metrics["total_latency"].get("p95_ms")
                ),
            }
        )
    return rows


def live_scenario_detail_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_key, metrics in report["summary"]["scenarios"].items():
        rows.append(
            {
                "Scenario": _live_scenario_name(scenario_key),
                "Citation validity": _rate_text(
                    metrics.get("citation_validity_rate")
                ),
                "Citation relevance": _rate_text(
                    metrics.get("citation_relevance_rate")
                ),
                "Formatting compliance": _rate_text(
                    metrics.get("formatting_compliance_rate")
                ),
                "Quality pass": _rate_text(metrics.get("quality_pass_rate")),
                "Legacy exact contract": _rate_text(
                    metrics.get(
                        "legacy_exact_contract_rate",
                        metrics.get("answer_correctness_rate"),
                    )
                ),
                "Finish reasons": metrics.get("finish_reasons", {}),
                "Fallback reasons": metrics.get("fallback_reasons", {}),
                "Revised failures": _failure_counts(
                    metrics.get("failures", [])
                ),
                "Legacy failures": _failure_counts(
                    metrics.get("legacy_failures", [])
                ),
            }
        )
    return rows


def live_evaluation_status(report: dict[str, Any]) -> tuple[str, str]:
    overall = report.get("summary", {}).get("overall", {})
    run_count = overall.get("run_count")
    fallback_rate = overall.get("fallback_rate")
    failure_count = overall.get("failure_count")
    if failure_count is None and "failures" in overall:
        failure_count = len(overall["failures"])

    complete_success = (
        isinstance(run_count, int)
        and run_count > 0
        and fallback_rate == 0.0
        and failure_count == 0
    )
    if complete_success:
        return (
            "success",
            "No fallbacks or failures under the revised live evaluation checks. "
            "Legacy strict-contract mismatches are reported separately.",
        )

    if (fallback_rate or 0) > 0 or (failure_count or 0) > 0:
        return (
            "warning",
            "The measured live results include fallbacks or failures under the "
            "revised checks. Review the detailed per-scenario diagnostics and "
            "individual saved runs.",
        )
    return (
        "warning",
        "The saved report does not contain enough measured revised-check data "
        "to declare a failure-free live run.",
    )


def _render_live_ollama_dashboard(st: Any, report_path: Path) -> None:
    st.markdown("---")
    st.subheader("Live Ollama Evaluation")
    st.caption(
        "Separate opt-in measurements from real Snowflake, hybrid RAG, and the "
        "configured local Ollama model. Rendering this section only reads JSON; "
        "it never calls Snowflake or Ollama."
    )
    try:
        report = load_live_ollama_report(report_path)
    except EvaluationReportError as exc:
        st.info(str(exc))
        columns = st.columns(5)
        for column, label in zip(
            columns,
            (
                "Semantic correctness",
                "Factual completeness",
                "Claim grounding",
                "Citation validity",
                "Formatting compliance",
            ),
            strict=True,
        ):
            column.metric(label, "Not measured")
        st.code(
            "python -m servicing_agent.benchmark_ollama --runs 5",
            language="powershell",
        )
        st.caption("Human-reviewed answer quality: Not measured")
        return

    overall = report["summary"]["overall"]
    first, second, third, fourth, fifth = st.columns(5)
    first.metric(
        "Semantic correctness",
        _rate_text(overall.get("semantic_answer_correctness_rate")),
    )
    second.metric(
        "Factual completeness",
        _rate_text(overall.get("factual_completeness_rate")),
    )
    third.metric(
        "Claim grounding", _rate_text(overall.get("claim_grounding_rate"))
    )
    fourth.metric(
        "Citation validity", _rate_text(overall.get("citation_validity_rate"))
    )
    fifth.metric(
        "Formatting compliance",
        _rate_text(overall.get("formatting_compliance_rate")),
    )
    secondary = st.columns(5)
    secondary[0].metric(
        "LLM success", _rate_text(overall.get("llm_success_rate"))
    )
    secondary[1].metric("Fallback", _rate_text(overall.get("fallback_rate")))
    secondary[2].metric(
        "Citation relevance",
        _rate_text(overall.get("citation_relevance_rate")),
    )
    secondary[3].metric(
        "Legacy exact contract",
        _rate_text(
            overall.get(
                "legacy_exact_contract_rate",
                overall.get("answer_correctness_rate"),
            )
        ),
    )
    secondary[4].metric("Human review", "Not measured")
    st.caption(
        f"Model: {report['model']} · generated {report['generated_at_utc']} · "
        f"{report['scenario_count']} scenarios × {report['runs_per_scenario']} "
        f"measured runs · warm-up excluded."
    )
    st.warning(
        "Citation validity means the application validator accepted the response "
        "and citation labels appeared inline. It is not definitive factual "
        "correctness or human review."
    )

    st.markdown("#### Per-scenario live results")
    rows = live_scenario_rows(report)
    st.dataframe(rows, width="stretch", hide_index=True)
    with st.expander("Detailed per-scenario metrics and diagnostics"):
        st.dataframe(
            live_scenario_detail_rows(report),
            width="stretch",
            hide_index=True,
        )
    status_level, status_message = live_evaluation_status(report)
    getattr(st, status_level)(status_message)

    cold = report.get("cold_start", {})
    st.markdown("#### Cold start versus warm requests")
    st.dataframe(
        [
            {
                "Cold total ms": _latency_text(cold.get("total_latency_ms")),
                "Cold retrieval ms": _latency_text(
                    cold.get("retrieval_latency_ms")
                ),
                "Cold Snowflake ms": _latency_text(
                    cold.get("snowflake_latency_ms")
                ),
                "Cold generation ms": _latency_text(
                    cold.get("generation_latency_ms")
                ),
                "Warm multi-source total P50 ms": _latency_text(
                    cold.get("warm_multi_source_total_p50_ms")
                ),
                "Cold-start excess ms": _latency_text(
                    cold.get("cold_start_excess_ms")
                ),
            }
        ],
        width="stretch",
        hide_index=True,
    )
    human_review = report.get("human_review", {})
    st.info(
        "Human-reviewed answer quality: "
        + (
            "Not measured"
            if human_review.get("score") is None
            else str(human_review["score"])
        )
    )
    with st.expander("Live evaluation metadata and individual runs"):
        st.json(
            {
                "metric_notes": report.get("metric_notes", {}),
                "scoring_methodology": report.get("scoring_methodology", {}),
                "score_revision": report.get("score_revision", {}),
                "human_review": human_review,
                "cold_start": cold,
                "runs": report["runs"],
            }
        )


def render_evaluation_dashboard(
    st: Any,
    report_path: str | Path,
    *,
    evaluation_modes: tuple[str, ...] = ("lexical", "semantic", "hybrid"),
    evaluation_runs: int = 3,
) -> None:
    st.subheader("Measured evaluation dashboard")
    st.markdown("### Offline deterministic evaluation")
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
        _render_live_ollama_dashboard(
            st, Path(report_path).parent / "live_ollama" / "latest.json"
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
    _render_live_ollama_dashboard(
        st, Path(report_path).parent / "live_ollama" / "latest.json"
    )
