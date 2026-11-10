from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import httpx

from .api import create_app
from .demo_scenarios import DEMO_SCENARIOS, DemoScenario
from .evaluation import asserts_unsupported_claim, percentile
from .generation import citation_supports_claim
from .logging_config import configure_logging
from .schemas import SourceCitation
from .settings import ConfigurationError, Settings


LIVE_REPORT_SCHEMA_VERSION = 2
LIVE_SCENARIO_GOLDEN_IDS = {
    "multi_source": "multi-source-case-investigation",
    "premium_policy": "premium-policy-applicability",
    "standard_policy": "standard-policy-applicability",
    "unknown_diagnostics": "case-4821-diagnostic-logs-unknown",
    "resolved_incident": "resolved-telemetry-incident",
}


def _rate(values: Sequence[bool]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _latency(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50_ms": None, "p95_ms": None}
    return {
        "p50_ms": round(percentile(values, 0.50), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
    }


def load_live_contracts(
    dataset_path: str | Path,
) -> list[tuple[DemoScenario, dict[str, Any]]]:
    path = Path(dataset_path)
    examples = json.loads(path.read_text(encoding="utf-8"))
    by_id = {example["id"]: example for example in examples}
    scenarios = {scenario.key: scenario for scenario in DEMO_SCENARIOS}
    contracts: list[tuple[DemoScenario, dict[str, Any]]] = []
    for scenario_key, golden_id in LIVE_SCENARIO_GOLDEN_IDS.items():
        try:
            scenario = scenarios[scenario_key]
            contract = by_id[golden_id]
        except KeyError as exc:
            raise ValueError(
                f"Missing live scenario or golden contract: {exc.args[0]}"
            ) from exc
        if scenario.prompt != contract["question"]:
            raise ValueError(
                f"Demo scenario {scenario_key} no longer matches {golden_id}."
            )
        contracts.append((scenario, contract))
    return contracts


def _legacy_quality_results(
    contract: dict[str, Any],
    reply: str,
    citations: list[SourceCitation],
    unsupported_claims_avoided: bool,
) -> dict[str, bool]:
    lowered = reply.lower()
    expectations = contract.get("answer_quality_checks", {})
    results: dict[str, bool] = {}
    if expectations.get("must_answer_directly"):
        # Preserve the deterministic golden evaluator's existing contract.
        results["must_answer_directly"] = reply.lstrip().startswith("**Answer**")
    if expectations.get("must_state_uncertainty"):
        results["must_state_uncertainty"] = any(
            marker in lowered
            for marker in (
                "does not confirm",
                "not recorded",
                "not establish",
                "not available",
                "unknown",
                "unconfirmed",
                "insufficient",
            )
        )
    if expectations.get("must_state_confirmed_completion"):
        results["must_state_confirmed_completion"] = any(
            marker in lowered
            for marker in ("completed", "confirmed", "marked resolved")
        )
    if expectations.get("must_use_citations"):
        results["must_use_citations"] = bool(citations) and all(
            citation.citation_label in reply for citation in citations
        )
    if expectations.get("must_return_no_citations"):
        results["must_return_no_citations"] = not citations
    if expectations.get("must_avoid_unsupported_claims"):
        results["must_avoid_unsupported_claims"] = unsupported_claims_avoided
    return results


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}
_TOKEN_ALIASES = {
    "backlogs": "backlog",
    "blocked": "block",
    "blocking": "block",
    "certificate": "certificate",
    "certificates": "certificate",
    "cleared": "clear",
    "clearing": "clear",
    "completed": "complete",
    "completion": "complete",
    "confirmed": "confirm",
    "confirmation": "confirm",
    "diagnostics": "diagnostic",
    "expired": "expire",
    "expiration": "expire",
    "exports": "export",
    "logs": "log",
    "marked": "mark",
    "received": "receive",
    "receiving": "receive",
    "receipt": "receive",
    "recorded": "record",
    "records": "record",
    "requested": "request",
    "requests": "request",
    "resolved": "resolve",
    "resolution": "resolve",
}
_UNCERTAINTY_MARKERS = (
    "cannot confirm",
    "can't confirm",
    "does not confirm",
    "doesn't confirm",
    "not confirmed",
    "not recorded",
    "not establish",
    "not available",
    "no record",
    "no evidence",
    "unknown",
    "unconfirmed",
    "unverified",
    "insufficient",
)
_REASONING_MARKERS = (
    "we are given the customer question",
    "we need answer",
    "let's analyze",
    "analysis:",
    "<think>",
    "</think>",
)


def _canonical_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in _TOKEN_PATTERN.findall(text.lower()):
        if raw in _STOP_WORDS:
            continue
        tokens.add(_TOKEN_ALIASES.get(raw, raw))
    return tokens


def _states_uncertainty(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _UNCERTAINTY_MARKERS)


def _locally_negates(text: str, expected_tokens: set[str]) -> bool:
    """Detect direct negation of a material predicate in an expected fact."""

    predicate_forms = {
        "block": r"block(?:ed|ing)?",
        "clear": r"clear(?:ed|ing)?",
        "complete": r"complet(?:e|ed|ion)",
        "confirm": r"confirm(?:ed|ation)?",
        "expire": r"expir(?:e|ed|ation)",
        "mark": r"mark(?:ed|ing)?",
        "receive": r"receiv(?:e|ed|ing)",
        "request": r"request(?:ed|ing)?",
        "resolve": r"resolv(?:e|ed|ing|ution)",
    }
    lowered = text.lower()
    for token, form in predicate_forms.items():
        if token not in expected_tokens:
            continue
        if re.search(
            rf"\b(?:is|are|was|were|did|does|has|have|had)\s+not\s+"
            rf"(?:(?:actually|successfully|yet)\s+)?{form}\b",
            lowered,
        ):
            return True
        if re.search(rf"\b(?:not|never)\s+{form}\b", lowered):
            return True
    return False


def _semantic_expectation_met(expected: str, reply: str) -> bool:
    """Match contract meaning while retaining polarity and material terms."""

    expected_tokens = _canonical_tokens(expected)
    reply_tokens = _canonical_tokens(reply)
    if not expected_tokens:
        return expected.lower() in reply.lower()

    expected_is_uncertain = _states_uncertainty(expected)
    if expected_is_uncertain and not _states_uncertainty(reply):
        return False
    if not expected_is_uncertain and _locally_negates(reply, expected_tokens):
        return False

    overlap = len(expected_tokens & reply_tokens) / len(expected_tokens)
    # A short expected phrase needs all material terms; longer facts may omit an
    # actor or grammatical detail while preserving the same meaning.
    threshold = 1.0 if len(expected_tokens) <= 2 else 0.6
    return overlap >= threshold


def _formatting_checks(reply: str) -> dict[str, bool]:
    stripped = reply.strip()
    lowered = stripped.lower()
    return {
        "nonempty": bool(stripped),
        "no_reasoning_or_planning_text": not any(
            marker in lowered for marker in _REASONING_MARKERS
        ),
        "balanced_citation_brackets": stripped.count("[") == stripped.count("]"),
        "complete_ending": bool(stripped)
        and (stripped[-1] in ".!?]" or stripped.endswith(")")),
    }


def evaluate_live_quality(
    contract: dict[str, Any],
    reply: str,
    *,
    intent_correct: bool,
    evidence_status_correct: bool,
    citation_valid: bool,
    citation_relevance: bool,
    required_evidence_present: bool,
) -> dict[str, Any]:
    """Score natural LLM answers without changing deterministic contracts."""

    expected_contains = contract.get("expected_contains", [])
    expected_facts = contract.get("expected_facts", [])
    unsupported_claims = contract.get("unsupported_claims", [])
    expected_text_checks = {
        text: _semantic_expectation_met(text, reply) for text in expected_contains
    }
    expected_fact_checks = {
        fact: _semantic_expectation_met(fact, reply) for fact in expected_facts
    }
    expected_fact_contradictions = {
        fact: (
            not _states_uncertainty(fact)
            and _locally_negates(reply, _canonical_tokens(fact))
        )
        for fact in expected_facts
    }
    unsupported_claims_avoided = all(
        not asserts_unsupported_claim(reply, claim) for claim in unsupported_claims
    )
    quality_expectations = contract.get("answer_quality_checks", {})
    state_checks: dict[str, bool] = {}
    if quality_expectations.get("must_state_uncertainty"):
        state_checks["states_uncertainty"] = _states_uncertainty(reply)
    if quality_expectations.get("must_state_confirmed_completion"):
        completion_tokens = {"complete", "confirm", "resolve", "clear"}
        state_checks["states_confirmed_completion"] = bool(
            completion_tokens & _canonical_tokens(reply)
        ) and not _locally_negates(reply, completion_tokens)

    semantic_answer_correct = all(
        (
            intent_correct,
            evidence_status_correct,
            all(expected_text_checks.values()),
            all(state_checks.values()),
            not any(expected_fact_contradictions.values()),
            unsupported_claims_avoided,
        )
    )
    factual_completeness = all(
        (
            all(expected_text_checks.values()),
            all(expected_fact_checks.values()),
        )
    )
    claim_grounding = all(
        (
            required_evidence_present,
            citation_relevance,
            unsupported_claims_avoided,
        )
    )
    formatting_checks = _formatting_checks(reply)
    formatting_compliance = all(formatting_checks.values())
    dimensions = {
        "semantic_answer_correctness": semantic_answer_correct,
        "factual_completeness": factual_completeness,
        "claim_grounding": claim_grounding,
        "citation_validity": citation_valid,
        "formatting_compliance": formatting_compliance,
    }
    failures = [name for name, passed in dimensions.items() if not passed]
    return {
        **dimensions,
        "quality_pass": all(dimensions.values()),
        "quality_failure_dimensions": failures,
        "semantic_expected_text_checks": expected_text_checks,
        "semantic_expected_fact_checks": expected_fact_checks,
        "semantic_fact_contradictions": expected_fact_contradictions,
        "semantic_state_checks": state_checks,
        "formatting_checks": formatting_checks,
        "unsupported_claims_avoided": unsupported_claims_avoided,
    }


def evaluate_live_response(
    scenario: DemoScenario,
    contract: dict[str, Any],
    body: dict[str, Any],
    run_number: int,
) -> dict[str, Any]:
    reply = str(body.get("reply", ""))
    citations = [
        SourceCitation.model_validate(citation)
        for citation in body.get("citations", [])
    ]
    returned_sources = {citation.source_id for citation in citations}
    retrieval_debug = body.get("retrieval_debug") or {}
    generation_debug = body.get("generation_debug") or {}
    history_debug = body.get("history_debug") or {}
    selected_documents = set(retrieval_debug.get("selected_document_ids") or [])

    citation_labels_inline = bool(citations) and all(
        citation.citation_label in reply for citation in citations
    )
    citation_relevance = bool(citations) and all(
        citation_supports_claim(reply, citation) for citation in citations
    )
    llm_success = (
        body.get("generation_mode") == "llm"
        and generation_debug.get("fallback_reason") is None
    )
    citation_valid = llm_success and citation_labels_inline

    expected_contains = contract.get("expected_contains", [])
    expected_facts = contract.get("expected_facts", [])
    unsupported_claims = contract.get("unsupported_claims", [])
    expected_text_present = all(text in reply for text in expected_contains)
    expected_facts_present = all(text in reply for text in expected_facts)
    unsupported_claims_avoided = all(
        not asserts_unsupported_claim(reply, claim)
        for claim in unsupported_claims
    )
    required = contract.get("required_evidence", {})
    required_sources = set(required.get("source_ids", []))
    required_documents = set(required.get("document_ids", []))
    expected_sources = set(contract.get("expected_source_ids", []))
    expected_documents = set(contract.get("relevant_document_ids", []))
    required_evidence_present = (
        required_sources.issubset(returned_sources)
        and required_documents.issubset(selected_documents)
        and expected_sources.issubset(returned_sources)
        and expected_documents.issubset(selected_documents)
    )
    legacy_quality_results = _legacy_quality_results(
        contract, reply, citations, unsupported_claims_avoided
    )
    legacy_answer_quality_correct = all(legacy_quality_results.values())
    intent_correct = body.get("intent") == contract["expected_intent"]
    evidence_status_correct = (
        body.get("evidence_status") == contract["expected_evidence_status"]
    )
    legacy_answer_correct = all(
        (
            intent_correct,
            evidence_status_correct,
            expected_text_present,
            expected_facts_present,
            unsupported_claims_avoided,
            required_evidence_present,
            legacy_answer_quality_correct,
            citation_relevance,
        )
    )
    legacy_grounding_checks_passed = all(
        (
            required_evidence_present,
            expected_facts_present,
            unsupported_claims_avoided,
            citation_relevance,
        )
    )
    legacy_failures: list[str] = []
    legacy_checks = (
        (llm_success, "llm_fallback"),
        (citation_labels_inline, "citation_label_missing_inline"),
        (citation_relevance, "claim_citation_relevance"),
        (intent_correct, "intent"),
        (evidence_status_correct, "evidence_status"),
        (expected_text_present, "expected_text"),
        (expected_facts_present, "expected_facts"),
        (unsupported_claims_avoided, "unsupported_claim"),
        (required_evidence_present, "required_evidence"),
        (legacy_answer_quality_correct, "answer_quality"),
    )
    legacy_failures.extend(
        name for passed, name in legacy_checks if not passed
    )
    quality = evaluate_live_quality(
        contract,
        reply,
        intent_correct=intent_correct,
        evidence_status_correct=evidence_status_correct,
        citation_valid=citation_valid,
        citation_relevance=citation_relevance,
        required_evidence_present=required_evidence_present,
    )
    failures = list(quality["quality_failure_dimensions"])
    if not llm_success:
        failures.insert(0, "llm_fallback")
    return {
        "scenario_key": scenario.key,
        "scenario_label": scenario.label,
        "golden_contract_id": contract["id"],
        "run_number": run_number,
        "generation_mode": body.get("generation_mode"),
        "llm_success": llm_success,
        "fallback": not llm_success,
        "fallback_reason": generation_debug.get("fallback_reason"),
        "finish_reason": generation_debug.get("finish_reason"),
        "output_tokens": generation_debug.get("output_tokens"),
        "intent_correct": intent_correct,
        "evidence_status_correct": evidence_status_correct,
        "citation_valid": citation_valid,
        "citation_labels_inline": citation_labels_inline,
        "citation_relevance": citation_relevance,
        "semantic_answer_correct": quality["semantic_answer_correctness"],
        "factual_completeness": quality["factual_completeness"],
        "claim_grounding": quality["claim_grounding"],
        "formatting_compliance": quality["formatting_compliance"],
        "quality_pass": quality["quality_pass"],
        "quality_failure_dimensions": quality["quality_failure_dimensions"],
        "semantic_expected_text_checks": quality[
            "semantic_expected_text_checks"
        ],
        "semantic_expected_fact_checks": quality[
            "semantic_expected_fact_checks"
        ],
        "semantic_fact_contradictions": quality[
            "semantic_fact_contradictions"
        ],
        "semantic_state_checks": quality["semantic_state_checks"],
        "formatting_checks": quality["formatting_checks"],
        "answer_correct": legacy_answer_correct,
        "grounding_checks_passed": legacy_grounding_checks_passed,
        "legacy_exact_contract_pass": legacy_answer_correct,
        "legacy_grounding_contract_pass": legacy_grounding_checks_passed,
        "human_reviewed_answer_quality": None,
        "expected_text_present": expected_text_present,
        "expected_facts_present": expected_facts_present,
        "unsupported_claims_avoided": unsupported_claims_avoided,
        "required_evidence_present": required_evidence_present,
        "answer_quality_correct": legacy_answer_quality_correct,
        "answer_quality_checks": legacy_quality_results,
        "failure_categories": failures,
        "legacy_failure_categories": legacy_failures,
        "generation_latency_ms": generation_debug.get("latency_ms"),
        "total_latency_ms": body.get("total_latency_ms"),
        "retrieval_latency_ms": retrieval_debug.get("latency_ms"),
        "snowflake_latency_ms": history_debug.get("latency_ms"),
        "retrieval_mode": retrieval_debug.get("effective_mode"),
        "retrieval_fallback_reason": retrieval_debug.get("fallback_reason"),
        "history_status": history_debug.get("status"),
        "citation_source_ids": sorted(returned_sources),
        "selected_document_ids": sorted(selected_documents),
        "answer_text": reply,
        "answer_sha256": hashlib.sha256(reply.encode("utf-8")).hexdigest(),
    }


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    generation_latencies = [
        float(run["generation_latency_ms"])
        for run in runs
        if run.get("generation_latency_ms") is not None
    ]
    total_latencies = [
        float(run["total_latency_ms"])
        for run in runs
        if run.get("total_latency_ms") is not None
    ]
    output_tokens = [
        int(run["output_tokens"])
        for run in runs
        if run.get("output_tokens") is not None
    ]
    failures = [
        {
            "run_number": run["run_number"],
            "categories": run.get("quality_failure_dimensions", []),
            "fallback_reason": run.get("fallback_reason"),
        }
        for run in runs
        if run.get("quality_failure_dimensions")
    ]
    legacy_failures = [
        {
            "run_number": run["run_number"],
            "categories": run.get("legacy_failure_categories", []),
            "fallback_reason": run.get("fallback_reason"),
        }
        for run in runs
        if run.get("legacy_failure_categories")
    ]
    return {
        "run_count": len(runs),
        "llm_success_rate": _rate([bool(run["llm_success"]) for run in runs]),
        "fallback_rate": _rate([bool(run["fallback"]) for run in runs]),
        "citation_validity_rate": _rate(
            [bool(run["citation_valid"]) for run in runs]
        ),
        "citation_relevance_rate": _rate(
            [bool(run["citation_relevance"]) for run in runs]
        ),
        "semantic_answer_correctness_rate": _rate(
            [bool(run["semantic_answer_correct"]) for run in runs]
        ),
        "factual_completeness_rate": _rate(
            [bool(run["factual_completeness"]) for run in runs]
        ),
        "claim_grounding_rate": _rate(
            [bool(run["claim_grounding"]) for run in runs]
        ),
        "formatting_compliance_rate": _rate(
            [bool(run["formatting_compliance"]) for run in runs]
        ),
        "quality_pass_rate": _rate(
            [bool(run["quality_pass"]) for run in runs]
        ),
        "legacy_exact_contract_rate": _rate(
            [bool(run["legacy_exact_contract_pass"]) for run in runs]
        ),
        "legacy_grounding_contract_rate": _rate(
            [bool(run["legacy_grounding_contract_pass"]) for run in runs]
        ),
        # Backward-compatible aliases retain their original exact semantics.
        "answer_correctness_rate": _rate(
            [bool(run["legacy_exact_contract_pass"]) for run in runs]
        ),
        "grounding_checks_rate": _rate(
            [bool(run["legacy_grounding_contract_pass"]) for run in runs]
        ),
        "human_reviewed_answer_quality_rate": None,
        "generation_latency": _latency(generation_latencies),
        "total_latency": _latency(total_latencies),
        "output_tokens_p50": (
            round(statistics.median(output_tokens), 3)
            if output_tokens
            else None
        ),
        "finish_reasons": dict(
            Counter(str(run.get("finish_reason")) for run in runs)
        ),
        "fallback_reasons": dict(
            Counter(
                str(run["fallback_reason"])
                for run in runs
                if run.get("fallback_reason")
            )
        ),
        "failure_count": len(failures),
        "failures": failures,
        "legacy_failure_count": len(legacy_failures),
        "legacy_failures": legacy_failures,
    }


def summarize_live_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_scenario: dict[str, Any] = {}
    for scenario_key in LIVE_SCENARIO_GOLDEN_IDS:
        matching = [run for run in runs if run["scenario_key"] == scenario_key]
        by_scenario[scenario_key] = _summary(matching)
    return {"overall": _summary(runs), "scenarios": by_scenario}


def _reevaluate_saved_run(
    run: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    """Re-score stored text and evidence without calling external services."""

    rescored = dict(run)
    previous_failures = list(
        run.get("legacy_failure_categories", run.get("failure_categories", []))
    )
    intent_correct = bool(
        run.get("intent_correct", "intent" not in previous_failures)
    )
    evidence_status_correct = bool(
        run.get(
            "evidence_status_correct",
            "evidence_status" not in previous_failures,
        )
    )
    quality = evaluate_live_quality(
        contract,
        str(run.get("answer_text", "")),
        intent_correct=intent_correct,
        evidence_status_correct=evidence_status_correct,
        citation_valid=bool(run.get("citation_valid")),
        citation_relevance=bool(run.get("citation_relevance")),
        required_evidence_present=bool(run.get("required_evidence_present")),
    )
    legacy_exact = bool(run.get("legacy_exact_contract_pass", run.get("answer_correct")))
    legacy_grounding = bool(
        run.get(
            "legacy_grounding_contract_pass",
            run.get("grounding_checks_passed"),
        )
    )
    rescored.update(
        {
            "intent_correct": intent_correct,
            "evidence_status_correct": evidence_status_correct,
            "semantic_answer_correct": quality[
                "semantic_answer_correctness"
            ],
            "factual_completeness": quality["factual_completeness"],
            "claim_grounding": quality["claim_grounding"],
            "formatting_compliance": quality["formatting_compliance"],
            "quality_pass": quality["quality_pass"],
            "quality_failure_dimensions": quality[
                "quality_failure_dimensions"
            ],
            "semantic_expected_text_checks": quality[
                "semantic_expected_text_checks"
            ],
            "semantic_expected_fact_checks": quality[
                "semantic_expected_fact_checks"
            ],
            "semantic_fact_contradictions": quality[
                "semantic_fact_contradictions"
            ],
            "semantic_state_checks": quality["semantic_state_checks"],
            "formatting_checks": quality["formatting_checks"],
            "unsupported_claims_avoided": quality[
                "unsupported_claims_avoided"
            ],
            "legacy_exact_contract_pass": legacy_exact,
            "legacy_grounding_contract_pass": legacy_grounding,
            "legacy_failure_categories": previous_failures,
            "failure_categories": list(
                quality["quality_failure_dimensions"]
            ),
        }
    )
    return rescored


def _legacy_grounding_failure_reasons(run: dict[str, Any]) -> list[str]:
    checks = (
        (run.get("required_evidence_present", False), "required_evidence"),
        (run.get("expected_facts_present", False), "exact_expected_facts"),
        (run.get("unsupported_claims_avoided", False), "unsupported_claim"),
        (run.get("citation_relevance", False), "claim_citation_relevance"),
    )
    return [name for passed, name in checks if not passed]


def reevaluate_saved_report(
    report: dict[str, Any], dataset_path: str | Path
) -> dict[str, Any]:
    """Apply revised metrics to a saved benchmark; no integration calls occur."""

    if report.get("evaluation_type") != "live_ollama" or not report.get("runs"):
        raise ValueError("The saved report is not a live Ollama benchmark artifact.")
    contracts = {
        scenario.key: contract
        for scenario, contract in load_live_contracts(dataset_path)
    }
    previous_runs = list(report["runs"])
    rescored_runs = [
        _reevaluate_saved_run(run, contracts[run["scenario_key"]])
        for run in previous_runs
    ]
    grounding_failures = [
        {
            "scenario_key": run["scenario_key"],
            "run_number": run["run_number"],
            "answer_sha256": run.get("answer_sha256"),
            "legacy_failure_categories": run.get(
                "legacy_failure_categories", run.get("failure_categories", [])
            ),
            "legacy_grounding_failure_reasons": (
                _legacy_grounding_failure_reasons(run)
            ),
            "revised_claim_grounding": rescored["claim_grounding"],
            "revised_factual_completeness": rescored[
                "factual_completeness"
            ],
            "explanation": (
                "The legacy grounding aggregate included exact expected-fact "
                "wording; the revised claim-grounding dimension evaluates "
                "evidence coverage, claim-citation relevance, and unsupported "
                "claims independently."
            ),
        }
        for run, rescored in zip(previous_runs, rescored_runs, strict=True)
        if not run.get("grounding_checks_passed", True)
    ]
    formatting_failures = [
        {
            "scenario_key": run["scenario_key"],
            "run_number": run["run_number"],
            "answer_sha256": run.get("answer_sha256"),
            "legacy_check": "must_answer_directly",
            "legacy_reason": "Response did not begin with deterministic **Answer** markup.",
            "revised_formatting_compliance": rescored[
                "formatting_compliance"
            ],
        }
        for run, rescored in zip(previous_runs, rescored_runs, strict=True)
        if run.get("answer_quality_checks", {}).get("must_answer_directly")
        is False
    ]
    previous_overall = report.get("summary", {}).get("overall", {})
    rescored_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    revised = dict(report)
    revised.update(
        {
            "schema_version": LIVE_REPORT_SCHEMA_VERSION,
            "rescored_at_utc": rescored_at,
            "rescored_without_live_calls": True,
            "runs": rescored_runs,
            "summary": summarize_live_runs(rescored_runs),
            "scoring_methodology": {
                "version": "live-semantic-v2",
                "semantic_answer_correctness": (
                    "Expected meaning, requested answer state, intent, evidence "
                    "status, and unsupported-certainty checks."
                ),
                "factual_completeness": (
                    "Semantic coverage of every material expected statement and fact."
                ),
                "claim_grounding": (
                    "Required evidence, claim-citation relevance, and unsupported-claim avoidance."
                ),
                "citation_validity": (
                    "Application-accepted LLM mode with returned labels present inline."
                ),
                "formatting_compliance": (
                    "Finished customer-facing output with balanced citation markup; "
                    "deterministic Markdown headings are not required."
                ),
            },
            "score_revision": {
                "previous_schema_version": report.get("schema_version"),
                "previous_answer_correctness_rate": previous_overall.get(
                    "answer_correctness_rate"
                ),
                "previous_grounding_checks_rate": previous_overall.get(
                    "grounding_checks_rate"
                ),
                "legacy_exact_contract_preserved": True,
                "grounding_contract_failures": grounding_failures,
                "formatting_failures": formatting_failures,
            },
        }
    )
    revised["metric_notes"] = {
        **report.get("metric_notes", {}),
        "semantic_answer_correctness": (
            "Meaning-aware deterministic contract checks; not a human review."
        ),
        "factual_completeness": (
            "Every material expected fact must be present semantically; omissions fail."
        ),
        "claim_grounding": (
            "Required evidence and claim-citation support remain mandatory; invalid "
            "or unsupported claims fail."
        ),
        "formatting_compliance": (
            "Natural LLM answers need not use deterministic **Answer** markup."
        ),
        "legacy_exact_contract": (
            "Original exact-string and deterministic-format result retained for auditability."
        ),
    }
    return revised


async def run_live_evaluation(
    settings: Settings,
    dataset_path: str | Path,
    runs_per_scenario: int = 5,
) -> dict[str, Any]:
    if runs_per_scenario <= 0:
        raise ValueError("runs_per_scenario must be positive")
    if settings.history_backend != "snowflake":
        raise ConfigurationError(
            "Live Ollama evaluation requires HISTORY_BACKEND=snowflake."
        )
    if settings.retrieval_mode != "hybrid":
        raise ConfigurationError(
            "Live Ollama evaluation requires RETRIEVAL_MODE=hybrid."
        )
    if settings.llm_provider != "openai_compatible" or not settings.llm_api_key:
        raise ConfigurationError(
            "Live Ollama evaluation requires the configured OpenAI-compatible LLM."
        )
    if settings.llm_model != "qwen3:4b-instruct":
        raise ConfigurationError(
            "Live Ollama evaluation is scoped to qwen3:4b-instruct."
        )

    dataset = Path(dataset_path)
    contracts = load_live_contracts(dataset)
    with tempfile.TemporaryDirectory(prefix="servicing-agent-live-eval-") as temp:
        app = create_app(
            Path(temp) / "live-evaluation.db",
            configure_logs=False,
            settings_override=settings,
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            timeout = httpx.Timeout(settings.ui_api_read_timeout_seconds)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://live-evaluation",
                timeout=timeout,
            ) as client:
                warm_scenario, warm_contract = contracts[0]
                warm_response = await client.post(
                    "/chat",
                    headers={"X-Demo-User": warm_scenario.demo_user},
                    json={"message": warm_scenario.prompt},
                )
                warm_response.raise_for_status()
                warmup = evaluate_live_response(
                    warm_scenario, warm_contract, warm_response.json(), 0
                )

                measured: list[dict[str, Any]] = []
                for scenario, contract in contracts:
                    for run_number in range(1, runs_per_scenario + 1):
                        response = await client.post(
                            "/chat",
                            headers={"X-Demo-User": scenario.demo_user},
                            json={"message": scenario.prompt},
                        )
                        response.raise_for_status()
                        measured.append(
                            evaluate_live_response(
                                scenario,
                                contract,
                                response.json(),
                                run_number,
                            )
                        )

    summary = summarize_live_runs(measured)
    cold_total = warmup.get("total_latency_ms")
    warm_multi_p50 = summary["scenarios"]["multi_source"]["total_latency"][
        "p50_ms"
    ]
    cold_excess = (
        round(float(cold_total) - float(warm_multi_p50), 3)
        if cold_total is not None and warm_multi_p50 is not None
        else None
    )
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": LIVE_REPORT_SCHEMA_VERSION,
        "evaluation_type": "live_ollama",
        "generated_at_utc": generated_at,
        "model": settings.llm_model,
        "provider": settings.llm_provider,
        "history_backend": settings.history_backend,
        "retrieval_mode": settings.retrieval_mode,
        "runs_per_scenario": runs_per_scenario,
        "scenario_count": len(contracts),
        "measured_run_count": len(measured),
        "warmup_excluded": True,
        "dataset_path": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "max_output_tokens": settings.llm_max_tokens,
        "llm_timeout_seconds": settings.llm_timeout_seconds,
        "warmup": warmup,
        "cold_start": {
            "total_latency_ms": cold_total,
            "generation_latency_ms": warmup.get("generation_latency_ms"),
            "retrieval_latency_ms": warmup.get("retrieval_latency_ms"),
            "snowflake_latency_ms": warmup.get("snowflake_latency_ms"),
            "warm_multi_source_total_p50_ms": warm_multi_p50,
            "cold_start_excess_ms": cold_excess,
        },
        "summary": summary,
        "runs": measured,
        "scoring_methodology": {
            "version": "live-semantic-v2",
            "semantic_answer_correctness": (
                "Meaning-aware contract answer checks, including polarity and uncertainty."
            ),
            "factual_completeness": "Semantic coverage of every material expected fact.",
            "claim_grounding": (
                "Required evidence, claim-citation relevance, and unsupported-claim avoidance."
            ),
            "citation_validity": "Application-accepted inline source labels.",
            "formatting_compliance": (
                "Finished natural response; deterministic Markdown headings are optional."
            ),
        },
        "human_review": {
            "status": "not_measured",
            "score": None,
            "note": (
                "No human answer-quality review was performed. Deterministic "
                "golden-contract checks are reported separately."
            ),
        },
        "metric_notes": {
            "citation_validity": (
                "The application validator accepted LLM mode and every returned "
                "citation label appeared inline. This is not definitive factual "
                "correctness."
            ),
            "citation_relevance": (
                "Deterministic claim-to-source lexical relevance and polarity check."
            ),
            "semantic_answer_correctness": (
                "Meaning-aware deterministic contract checks; not a human review."
            ),
            "factual_completeness": (
                "All material expected statements and facts must be covered semantically."
            ),
            "claim_grounding": (
                "Required evidence, claim-citation relevance, and unsupported-claim avoidance."
            ),
            "formatting_compliance": (
                "Natural LLM answers need not use deterministic **Answer** markup."
            ),
            "legacy_exact_contract": (
                "Original exact wording and deterministic formatting retained separately."
            ),
            "latency": "Measured warm requests only; the warm-up is separate.",
        },
    }


def write_live_report(
    report: dict[str, Any], output_dir: str | Path
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = str(
        report.get("rescored_at_utc") or report["generated_at_utc"]
    )
    compact = timestamp.replace("-", "").replace(":", "").replace(".", "")
    compact = compact.replace("+0000", "Z")
    timestamped = directory / f"live_ollama_{compact}.json"
    latest = directory / "latest.json"
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    timestamped.write_text(rendered, encoding="utf-8")
    latest.write_text(rendered, encoding="utf-8")
    return timestamped, latest


def main() -> int:
    settings = Settings.from_environment()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate demo scenarios 2-6 using real Snowflake, hybrid RAG, and "
            "the configured qwen3:4b-instruct Ollama model."
        )
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=settings.evaluation_dataset_path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.evaluation_results_path.parent / "live_ollama",
    )
    parser.add_argument(
        "--reevaluate",
        type=Path,
        help=(
            "Re-score an existing live artifact locally without calling Snowflake or Ollama."
        ),
    )
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    configure_logging(
        settings.evaluation_log_level,
        json_logs=settings.log_json,
        trace_enabled=settings.trace_enabled,
    )
    if args.reevaluate:
        saved_report = json.loads(args.reevaluate.read_text(encoding="utf-8"))
        report = reevaluate_saved_report(saved_report, args.dataset)
    else:
        report = asyncio.run(
            run_live_evaluation(settings, args.dataset, args.runs)
        )
    timestamped, latest = write_live_report(report, args.output_dir)
    overall = report["summary"]["overall"]
    print(
        json.dumps(
            {
                "timestamped_report": str(timestamped),
                "latest_report": str(latest),
                "measured_runs": report["measured_run_count"],
                "llm_success_rate": overall["llm_success_rate"],
                "fallback_rate": overall["fallback_rate"],
                "semantic_answer_correctness_rate": overall[
                    "semantic_answer_correctness_rate"
                ],
                "factual_completeness_rate": overall[
                    "factual_completeness_rate"
                ],
                "claim_grounding_rate": overall["claim_grounding_rate"],
                "citation_validity_rate": overall[
                    "citation_validity_rate"
                ],
                "formatting_compliance_rate": overall[
                    "formatting_compliance_rate"
                ],
                "legacy_exact_contract_rate": overall[
                    "legacy_exact_contract_rate"
                ],
                "generation_latency": overall["generation_latency"],
                "total_latency": overall["total_latency"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
