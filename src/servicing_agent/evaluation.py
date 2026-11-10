from __future__ import annotations

import argparse
import json
import re
import statistics
import tempfile
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

from .actions import ControlledNoteService, StaleConfirmationError
from .auth import DEMO_USERS
from .database import Database
from .generation import (
    DeterministicGroundedGenerator,
    GroundedAnswerGenerator,
    OpenAICompatibleGroundedGenerator,
    citation_supports_claim,
)
from .handoffs import CaseHandoffService
from .history import MockHistoryAdapter
from .logging_config import configure_logging
from .retrieval import (
    EmbeddingProvider,
    EmbeddingUnavailableError,
    HybridRetrievalService,
    SentenceTransformerEmbeddingProvider,
)
from .service import ChatService, build_answer_generator
from .settings import Settings
from .sla import SLAService
from .time_utils import parse_utc, to_utc_iso
from .tools import CRMAuthorizationError

T = TypeVar("T")


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _mean(values: Sequence[float | bool]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def _latency_summary(values: Sequence[float]) -> tuple[float, float]:
    return (
        round(percentile(values, 0.50), 3),
        round(percentile(values, 0.95), 3),
    )


def asserts_unsupported_claim(reply: str, claim: str) -> bool:
    """Detect an asserted claim without treating an explicit denial as an assertion."""

    reply_lower = reply.lower()
    claim_lower = claim.lower()
    start = 0
    while (position := reply_lower.find(claim_lower, start)) >= 0:
        sentence_start = max(
            reply_lower.rfind(marker, 0, position)
            for marker in (".", "!", "?", "\n")
        )
        sentence_end_positions = [
            index
            for marker in (".", "!", "?", "\n")
            if (index := reply_lower.find(marker, position + len(claim_lower)))
            >= 0
        ]
        sentence_end = (
            min(sentence_end_positions)
            if sentence_end_positions
            else len(reply_lower)
        )
        sentence = reply_lower[sentence_start + 1 : sentence_end]
        prefix = sentence[: sentence.find(claim_lower)]
        epistemic_negation = re.search(
            r"(?:not|no) evidence (?:of|that)|"
            r"does not (?:confirm|establish|prove|show)|"
            r"cannot (?:confirm|establish|determine)|"
            r"not recorded|unverified|no confirmed",
            prefix,
        )
        if epistemic_negation is None:
            return True
        start = position + len(claim_lower)
    return False


def _timed_call(
    operation: Callable[[], T], observations: list[float]
) -> T:
    started = time.perf_counter()
    try:
        return operation()
    finally:
        observations.append(round((time.perf_counter() - started) * 1000, 3))


def _build_generator(
    settings: Settings, use_configured_llm: bool
) -> tuple[GroundedAnswerGenerator, str]:
    if not use_configured_llm:
        return DeterministicGroundedGenerator(), "deterministic"
    generator = build_answer_generator(settings)
    if isinstance(generator, OpenAICompatibleGroundedGenerator):
        return generator, "genuine_llm_configured"
    return generator, "deterministic_fallback_no_credentials"


def _chat_outcome(
    example: dict[str, Any],
    service: ChatService,
    end_to_end_latencies: list[float],
    retrieval_latencies: list[float],
    tool_latencies: list[float],
) -> dict[str, Any]:
    started = time.perf_counter()
    response = service.chat(
        example["question"], DEMO_USERS[example["demo_user"]]
    )
    elapsed = round((time.perf_counter() - started) * 1000, 3)
    end_to_end_latencies.append(elapsed)
    debug = response.retrieval_debug
    if debug:
        retrieval_latencies.append(debug.latency_ms)
        tool_latencies.append(debug.latency_ms)
    selected_ids = debug.selected_document_ids if debug else []
    returned_sources = {citation.source_id for citation in response.citations}
    claim_citation_relevance = all(
        citation.citation_label in response.reply
        and citation_supports_claim(response.reply, citation)
        for citation in response.citations
    )
    expected_sources = set(example.get("expected_source_ids", []))
    expected_facts = example.get("expected_facts", [])
    unsupported_claims = example.get("unsupported_claims", [])
    reply_lower = response.reply.lower()
    expected_facts_correct = all(
        fact in response.reply for fact in expected_facts
    )
    unsupported_claims_avoided = all(
        not asserts_unsupported_claim(response.reply, claim)
        for claim in unsupported_claims
    )
    required_evidence = example.get("required_evidence", {})
    required_sources = set(required_evidence.get("source_ids", []))
    required_documents = set(required_evidence.get("document_ids", []))
    required_evidence_present = (
        required_sources.issubset(returned_sources)
        and required_documents.issubset(set(selected_ids))
    )
    quality_expectations = example.get("answer_quality_checks", {})
    quality_results: dict[str, bool] = {}
    if quality_expectations.get("must_answer_directly"):
        quality_results["must_answer_directly"] = response.reply.lstrip().startswith(
            "**Answer**"
        )
    if quality_expectations.get("must_state_uncertainty"):
        quality_results["must_state_uncertainty"] = any(
            marker in reply_lower
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
    if quality_expectations.get("must_state_confirmed_completion"):
        quality_results["must_state_confirmed_completion"] = any(
            marker in reply_lower
            for marker in ("completed", "confirmed", "marked resolved")
        )
    if quality_expectations.get("must_use_citations"):
        quality_results["must_use_citations"] = bool(response.citations) and all(
            citation.citation_label in response.reply
            for citation in response.citations
        )
    if quality_expectations.get("must_return_no_citations"):
        quality_results["must_return_no_citations"] = not response.citations
    if quality_expectations.get("must_avoid_unsupported_claims"):
        quality_results["must_avoid_unsupported_claims"] = (
            unsupported_claims_avoided
        )
    answer_quality_correct = all(quality_results.values())
    grounded = (
        all(text in response.reply for text in example.get("expected_contains", []))
        and expected_facts_correct
        and unsupported_claims_avoided
        and expected_sources.issubset(returned_sources)
        and required_evidence_present
        and answer_quality_correct
        and claim_citation_relevance
        and (
            not example.get("expect_no_citations")
            or not response.citations
        )
    )
    task_success = (
        response.intent == example["expected_intent"]
        and response.evidence_status == example["expected_evidence_status"]
        and grounded
    )
    authorization_safe: bool | None = None
    authorization_expectation = example.get("authorization_expectation")
    if (
        example["category"] == "unauthorized_access"
        or authorization_expectation == "denied"
    ):
        authorization_safe = (
            not response.citations
            and not selected_ids
            and "not authorized" in response.reply.lower()
        )
        task_success = task_success and authorization_safe
    elif authorization_expectation == "allowed":
        authorization_safe = "not authorized" not in response.reply.lower()
        task_success = task_success and authorization_safe
    policy_correct: bool | None = None
    if example["category"] == "policy_applicability":
        policy_correct = task_success and set(
            example.get("relevant_document_ids", [])
        ).issubset(set(selected_ids))
        task_success = task_success and policy_correct
    return {
        "success": task_success,
        "failure_reason": None if task_success else example["category"],
        "evidence_status": response.evidence_status,
        "selected_document_ids": list(selected_ids),
        "returned_source_ids": sorted(returned_sources),
        "grounded_answer_correct": grounded,
        "expected_facts_correct": expected_facts_correct,
        "unsupported_claims_avoided": unsupported_claims_avoided,
        "required_evidence_present": required_evidence_present,
        "answer_quality_correct": answer_quality_correct,
        "claim_citation_relevance": claim_citation_relevance,
        "answer_quality_checks": quality_results,
        "policy_correct": policy_correct,
        "authorization_safe": authorization_safe,
        "action_safe": None,
        "escalation_correct": None,
        "recovery_success": None,
        "generation_mode": response.generation_mode,
        "effective_retrieval_mode": debug.effective_mode if debug else None,
        "fallback_reason": debug.fallback_reason if debug else None,
        "end_to_end_latency_ms": elapsed,
    }


def _note_outcome(
    example: dict[str, Any],
    database: Database,
    service: ControlledNoteService,
    run_number: int,
    tool_latencies: list[float],
    end_to_end_latencies: list[float],
) -> dict[str, Any]:
    started = time.perf_counter()
    user = DEMO_USERS[example["demo_user"]]
    operation_id = f"eval-{example['id']}-{run_number}"
    variant = example["variant"]
    final_state: str | None = None
    error_code: str | None = None
    intermediate_state: str | None = None
    duplicate_safe = True
    authorization_safe: bool | None = None
    recovery_success: bool | None = None
    expected_denial = False
    try:
        if variant == "unauthorized_prepare":
            try:
                _timed_call(
                    lambda: service.prepare_note(
                        example["ticket_id"],
                        example["note"],
                        user,
                        operation_id=operation_id,
                    ),
                    tool_latencies,
                )
            except CRMAuthorizationError:
                expected_denial = True
            authorization_safe = (
                expected_denial
                and not database.fetch_ticket_notes(example["ticket_id"])
                and database.fetch_operation(operation_id) is None
            )
            success = authorization_safe
        else:
            preview = _timed_call(
                lambda: service.prepare_note(
                    example["ticket_id"],
                    example["note"],
                    user,
                    operation_id=operation_id,
                ),
                tool_latencies,
            )
            final_state = preview.state
            if variant == "idempotent_confirmation":
                completed = _timed_call(
                    lambda: service.confirm_note(
                        operation_id, user, customer_confirmed=True
                    ),
                    tool_latencies,
                )
                duplicate = _timed_call(
                    lambda: service.confirm_note(
                        operation_id, user, customer_confirmed=True
                    ),
                    tool_latencies,
                )
                final_state = duplicate.state
                duplicate_safe = completed.note_id == duplicate.note_id
            elif variant == "confirmation_required":
                try:
                    _timed_call(
                        lambda: service.confirm_note(operation_id, user),
                        tool_latencies,
                    )
                except ValueError:
                    expected_denial = True
                final_state = database.fetch_operation(operation_id)["state"]
            elif variant == "stale_confirmation":
                database.simulate_external_ticket_update(
                    example["ticket_id"],
                    owner="Evaluation External Queue",
                    updated_at="2026-11-08T12:01:00Z",
                )
                try:
                    _timed_call(
                        lambda: service.confirm_note(
                            operation_id, user, customer_confirmed=True
                        ),
                        tool_latencies,
                    )
                except StaleConfirmationError as exc:
                    final_state = exc.response.state
                    error_code = exc.response.error_code
            elif variant == "timeout_reconciliation":
                unknown = _timed_call(
                    lambda: service.confirm_note(
                        operation_id,
                        user,
                        customer_confirmed=True,
                        simulate_timeout_after_success=True,
                    ),
                    tool_latencies,
                )
                intermediate_state = unknown.state
                restarted = ControlledNoteService(database)
                reconciled = _timed_call(
                    lambda: restarted.reconcile(operation_id, user),
                    tool_latencies,
                )
                final_state = reconciled.state
                recovery_success = (
                    intermediate_state
                    == example.get("expected_intermediate_state")
                    and final_state == example["expected_final_state"]
                    and reconciled.note_id == unknown.note_id
                )

            notes = database.fetch_ticket_notes(example["ticket_id"])
            success = (
                final_state == example["expected_final_state"]
                and len(notes) == example["expected_note_count"]
                and duplicate_safe
            )
            if variant == "confirmation_required":
                success = success and expected_denial
            if example.get("expected_error_code"):
                success = success and error_code == example["expected_error_code"]
            if recovery_success is not None:
                success = success and recovery_success
        note_count = len(database.fetch_ticket_notes(example["ticket_id"]))
    except Exception as exc:  # keep the evaluation report complete
        success = False
        note_count = len(database.fetch_ticket_notes(example["ticket_id"]))
        error_code = f"UNEXPECTED_{type(exc).__name__}"
    if example["category"] == "failure_recovery" and recovery_success is None:
        recovery_success = success
    elapsed = round((time.perf_counter() - started) * 1000, 3)
    end_to_end_latencies.append(elapsed)
    return {
        "success": success,
        "failure_reason": None if success else error_code or example["category"],
        "evidence_status": "not_applicable",
        "selected_document_ids": [],
        "returned_source_ids": [],
        "grounded_answer_correct": None,
        "policy_correct": None,
        "authorization_safe": authorization_safe,
        "action_safe": success,
        "escalation_correct": None,
        "recovery_success": recovery_success,
        "generation_mode": "not_applicable",
        "effective_retrieval_mode": None,
        "fallback_reason": None,
        "final_state": final_state,
        "intermediate_state": intermediate_state,
        "note_count": note_count,
        "error_code": error_code,
        "end_to_end_latency_ms": elapsed,
    }


def _handoff_outcome(
    example: dict[str, Any],
    service: CaseHandoffService,
    run_number: int,
    tool_latencies: list[float],
    end_to_end_latencies: list[float],
) -> dict[str, Any]:
    started = time.perf_counter()
    user = DEMO_USERS[example["demo_user"]]
    prefix = f"eval-{example['id']}-{run_number}"
    try:
        created = _timed_call(
            lambda: service.create(
                example["ticket_id"],
                example["reason"],
                example["troubleshooting_history"],
                user,
                operation_id=f"{prefix}-create",
            ),
            tool_latencies,
        )
        accepted = _timed_call(
            lambda: service.accept(
                created.handoff_id,
                user,
                operation_id=f"{prefix}-accept",
            ),
            tool_latencies,
        )
        resolved = _timed_call(
            lambda: service.resolve(
                created.handoff_id,
                user,
                operation_id=f"{prefix}-resolve",
            ),
            tool_latencies,
        )
        refs_ok = (
            bool(created.source_references)
            if example.get("require_source_references")
            else True
        )
        success = (
            created.state == "CREATED"
            and accepted.state == "ACCEPTED"
            and resolved.state == example["expected_final_state"]
            and refs_ok
        )
        final_state = resolved.state
        sources = created.source_references
        failure = None if success else example["category"]
    except Exception as exc:
        success = False
        final_state = None
        sources = []
        failure = f"UNEXPECTED_{type(exc).__name__}"
    elapsed = round((time.perf_counter() - started) * 1000, 3)
    end_to_end_latencies.append(elapsed)
    return {
        "success": success,
        "failure_reason": failure,
        "evidence_status": "not_applicable",
        "selected_document_ids": [],
        "returned_source_ids": sources,
        "grounded_answer_correct": None,
        "policy_correct": None,
        "authorization_safe": None,
        "action_safe": success,
        "escalation_correct": success,
        "recovery_success": None,
        "generation_mode": "not_applicable",
        "effective_retrieval_mode": None,
        "fallback_reason": None,
        "final_state": final_state,
        "end_to_end_latency_ms": elapsed,
    }


def _sla_outcome(
    example: dict[str, Any],
    database: Database,
    service: SLAService,
    user_id: str,
    tool_latencies: list[float],
    end_to_end_latencies: list[float],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        initial = _timed_call(
            lambda: service.check_ticket(
                example["ticket_id"],
                DEMO_USERS[user_id],
                as_of=parse_utc(example["initial_as_of"]),
            ),
            tool_latencies,
        )
        repeated = _timed_call(
            lambda: service.check_ticket(
                example["ticket_id"],
                DEMO_USERS[user_id],
                as_of=parse_utc(example["initial_as_of"]),
            ),
            tool_latencies,
        )
        sweep = _timed_call(
            lambda: service.sweep(as_of=parse_utc(example["sweep_as_of"])),
            tool_latencies,
        )
        handoffs = database.fetch_handoffs(example["ticket_id"])
        sla_handoffs = [row for row in handoffs if row["trigger_type"] == "SLA"]
        handoff = sla_handoffs[0] if sla_handoffs else None
        success = (
            initial.action == example["expected_initial_action"]
            and repeated.action == "already_escalated"
            and len(sla_handoffs) == example["expected_handoff_count"]
            and handoff is not None
            and handoff["queue"] == example["expected_queue"]
            and sweep.backup_routed == 1
        )
        queue = handoff["queue"] if handoff else None
        failure = None if success else example["category"]
    except Exception as exc:
        success = False
        queue = None
        failure = f"UNEXPECTED_{type(exc).__name__}"
    elapsed = round((time.perf_counter() - started) * 1000, 3)
    end_to_end_latencies.append(elapsed)
    return {
        "success": success,
        "failure_reason": failure,
        "evidence_status": "not_applicable",
        "selected_document_ids": [],
        "returned_source_ids": [],
        "grounded_answer_correct": None,
        "policy_correct": None,
        "authorization_safe": None,
        "action_safe": success,
        "escalation_correct": success,
        "recovery_success": None,
        "generation_mode": "not_applicable",
        "effective_retrieval_mode": None,
        "fallback_reason": None,
        "queue": queue,
        "end_to_end_latency_ms": elapsed,
    }


def evaluate_modes(
    *,
    dataset_path: Path,
    modes: Sequence[str],
    runs: int,
    settings: Settings,
    embedding_provider: EmbeddingProvider | None = None,
    use_configured_llm: bool = False,
) -> dict[str, Any]:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    examples = json.loads(dataset_path.read_text(encoding="utf-8"))
    shared_provider = embedding_provider
    if shared_provider is None and any(mode != "lexical" for mode in modes):
        shared_provider = SentenceTransformerEmbeddingProvider(
            settings.retrieval_model,
            settings.retrieval_cache_dir / "models",
        )
    generator, declared_generation = _build_generator(
        settings, use_configured_llm
    )
    report: dict[str, Any] = {
        "schema_version": 2,
        "generated_at_utc": to_utc_iso(datetime.now(timezone.utc)),
        "dataset": str(dataset_path),
        "scenario_count": len(examples),
        "runs_per_scenario": runs,
        "model": settings.retrieval_model,
        "rrf_k": settings.retrieval_rrf_k,
        "generation_evaluation": {
            "requested": (
                "configured_llm" if use_configured_llm else "deterministic"
            ),
            "declared_effective": declared_generation,
            "credentials_present": bool(settings.llm_api_key),
            "note": (
                "LLM output is measured only when generation_mode is llm. "
                "Deterministic fallback results are never labeled as LLM results."
            ),
        },
        "history_evaluation": {
            "backend": "mock",
            "network_access": False,
            "note": (
                "Repeatable golden evaluation never queries configured external "
                "history. Snowflake is covered by opt-in integration tests."
            ),
        },
        "business_outcomes": {
            "measured": [],
            "not_measured": ["CSAT", "CPRR", "production ROI", "case deflection"],
            "note": "Business scenarios in the demo are illustrative, not measured outcomes.",
        },
        "metric_definitions": {
            "task_success_rate": "Fraction of scenario runs satisfying all declared expected outcomes.",
            "grounded_answer_accuracy": "Verifiable expected text and source citations for chat scenarios.",
            "answer_quality_accuracy": "Declared directness, uncertainty, completion, citation, and claim-safety checks for chat scenarios.",
            "claim_citation_relevance": "Fraction of chat runs whose inline claims have substantive, non-contradictory support from their cited source snippets.",
            "unsupported_claim_avoidance": "Fraction of chat runs that omit every scenario-declared unsupported claim.",
            "required_evidence_accuracy": "Fraction of chat runs returning every scenario-declared source and document requirement.",
            "action_safety_rate": "Controlled mutations obey confirmation, idempotency, staleness, and authorization expectations.",
            "latency": "Warm in-process timing; excludes model download and dashboard rendering.",
        },
        "modes": {},
    }
    fixed_now = settings.evaluation_clock_utc
    with tempfile.TemporaryDirectory(prefix="service-agent-eval-") as temp_dir:
        for mode in modes:
            retriever = HybridRetrievalService(
                mode=mode,
                embedding_provider=shared_provider,
                embedding_model=settings.retrieval_model,
                embedding_cache_dir=settings.retrieval_cache_dir,
                top_k=settings.retrieval_top_k,
                rrf_k=settings.retrieval_rrf_k,
                semantic_min_score=settings.retrieval_semantic_min_score,
                semantic_min_top_score=settings.retrieval_semantic_min_top_score,
            )
            index_status: str | None = None
            index_error: str | None = None
            if mode != "lexical":
                try:
                    index_status = retriever.refresh_embeddings(force=False)
                except EmbeddingUnavailableError as exc:
                    index_error = str(exc)

            database = Database(
                Path(temp_dir) / f"evaluation-{mode}.db",
                timeout_seconds=settings.sqlite_timeout_seconds,
                busy_timeout_ms=settings.sqlite_busy_timeout_ms,
            )
            database.initialize()
            task_results: list[bool] = []
            recalls: list[float] = []
            precisions: list[float] = []
            sufficiency_matches: list[bool] = []
            grounded_matches: list[bool] = []
            answer_quality_matches: list[bool] = []
            claim_citation_matches: list[bool] = []
            unsupported_claim_matches: list[bool] = []
            required_evidence_matches: list[bool] = []
            policy_matches: list[bool] = []
            authorization_matches: list[bool] = []
            action_matches: list[bool] = []
            escalation_matches: list[bool] = []
            recovery_matches: list[bool] = []
            retrieval_latencies: list[float] = []
            tool_latencies: list[float] = []
            end_to_end_latencies: list[float] = []
            effective_modes: Counter[str] = Counter()
            generation_modes: Counter[str] = Counter()
            failure_categories: Counter[str] = Counter()
            fallback_count = 0
            example_results: list[dict[str, Any]] = []

            for example in examples:
                repeated_outcomes: list[dict[str, Any]] = []
                for run_number in range(1, runs + 1):
                    database.reset_demo_data()
                    sla_service = SLAService(
                        database,
                        clock=lambda: fixed_now,
                        backup_after=timedelta(minutes=15),
                    )
                    chat_service = ChatService(
                        database,
                        settings=settings,
                        as_of=settings.evaluation_as_of_date,
                        answer_generator=generator,
                        retriever=retriever,
                        sla_service=sla_service,
                        history_adapter=MockHistoryAdapter(),
                    )
                    scenario_type = example.get("scenario_type", "chat")
                    if scenario_type == "chat":
                        outcome = _chat_outcome(
                            example,
                            chat_service,
                            end_to_end_latencies,
                            retrieval_latencies,
                            tool_latencies,
                        )
                    elif scenario_type == "controlled_note":
                        outcome = _note_outcome(
                            example,
                            database,
                            ControlledNoteService(database, chat_service.crm_tool),
                            run_number,
                            tool_latencies,
                            end_to_end_latencies,
                        )
                    elif scenario_type == "handoff":
                        outcome = _handoff_outcome(
                            example,
                            CaseHandoffService(
                                database,
                                chat_service.crm_tool,
                                chat_service.entitlement_tool,
                                chat_service.evidence_tool,
                            ),
                            run_number,
                            tool_latencies,
                            end_to_end_latencies,
                        )
                    elif scenario_type == "sla":
                        outcome = _sla_outcome(
                            example,
                            database,
                            sla_service,
                            example["demo_user"],
                            tool_latencies,
                            end_to_end_latencies,
                        )
                    else:
                        raise ValueError(f"Unknown scenario_type {scenario_type!r}")
                    repeated_outcomes.append(outcome)
                    task_results.append(outcome["success"])
                    if not outcome["success"]:
                        failure_categories[
                            outcome["failure_reason"] or example["category"]
                        ] += 1
                    for key, collection in (
                        ("grounded_answer_correct", grounded_matches),
                        ("answer_quality_correct", answer_quality_matches),
                        ("claim_citation_relevance", claim_citation_matches),
                        (
                            "unsupported_claims_avoided",
                            unsupported_claim_matches,
                        ),
                        (
                            "required_evidence_present",
                            required_evidence_matches,
                        ),
                        ("policy_correct", policy_matches),
                        ("authorization_safe", authorization_matches),
                        ("action_safe", action_matches),
                        ("escalation_correct", escalation_matches),
                        ("recovery_success", recovery_matches),
                    ):
                        if outcome.get(key) is not None:
                            collection.append(bool(outcome[key]))
                    if outcome.get("effective_retrieval_mode"):
                        effective_modes[outcome["effective_retrieval_mode"]] += 1
                    generation_modes[outcome["generation_mode"]] += 1
                    if outcome.get("fallback_reason"):
                        fallback_count += 1

                    relevant_ids = set(example.get("relevant_document_ids", []))
                    if example.get("evaluate_retrieval") and relevant_ids:
                        selected_ids = outcome["selected_document_ids"]
                        recalls.append(
                            len(set(selected_ids[:5]) & relevant_ids)
                            / len(relevant_ids)
                        )
                        precisions.append(
                            len(set(selected_ids[:3]) & relevant_ids) / 3
                        )
                    if example.get("scenario_type", "chat") == "chat":
                        sufficiency_matches.append(
                            outcome["evidence_status"]
                            == example["expected_evidence_status"]
                        )

                first = repeated_outcomes[0]
                example_results.append(
                    {
                        "id": example["id"],
                        "scenario_type": example.get("scenario_type", "chat"),
                        "category": example["category"],
                        "success": all(item["success"] for item in repeated_outcomes),
                        "failure_reason": next(
                            (
                                item["failure_reason"]
                                for item in repeated_outcomes
                                if not item["success"]
                            ),
                            None,
                        ),
                        "evidence_status": first["evidence_status"],
                        "selected_document_ids": first["selected_document_ids"],
                        "returned_source_ids": first["returned_source_ids"],
                        "grounded_answer_correct": first[
                            "grounded_answer_correct"
                        ],
                        "answer_quality_correct": first.get(
                            "answer_quality_correct"
                        ),
                        "claim_citation_relevance": first.get(
                            "claim_citation_relevance"
                        ),
                        "unsupported_claims_avoided": first.get(
                            "unsupported_claims_avoided"
                        ),
                        "required_evidence_present": first.get(
                            "required_evidence_present"
                        ),
                        "fallback_reason": first["fallback_reason"],
                        "end_to_end_latency_p50_ms": round(
                            percentile(
                                [
                                    item["end_to_end_latency_ms"]
                                    for item in repeated_outcomes
                                ],
                                0.50,
                            ),
                            3,
                        ),
                    }
                )

            retrieval_p50, retrieval_p95 = _latency_summary(retrieval_latencies)
            tool_p50, tool_p95 = _latency_summary(tool_latencies)
            e2e_p50, e2e_p95 = _latency_summary(end_to_end_latencies)
            report["modes"][mode] = {
                "scenario_count": len(examples),
                "scenario_run_count": len(task_results),
                "passed_scenario_runs": sum(task_results),
                "failed_scenario_runs": len(task_results) - sum(task_results),
                "task_success_rate": _mean(task_results),
                "recall_at_5": _mean(recalls),
                "precision_at_3": _mean(precisions),
                "evidence_sufficiency_accuracy": _mean(sufficiency_matches),
                "grounded_answer_accuracy": _mean(grounded_matches),
                "answer_quality_accuracy": _mean(answer_quality_matches),
                "claim_citation_relevance": _mean(claim_citation_matches),
                "unsupported_claim_avoidance": _mean(
                    unsupported_claim_matches
                ),
                "required_evidence_accuracy": _mean(
                    required_evidence_matches
                ),
                "policy_applicability_accuracy": _mean(policy_matches),
                "authorization_safety_rate": _mean(authorization_matches),
                "action_safety_rate": _mean(action_matches),
                "escalation_correctness": _mean(escalation_matches),
                "recovery_success_rate": _mean(recovery_matches),
                "retrieval_latency_p50_ms": retrieval_p50,
                "retrieval_latency_p95_ms": retrieval_p95,
                "tool_execution_latency_p50_ms": tool_p50,
                "tool_execution_latency_p95_ms": tool_p95,
                "end_to_end_latency_p50_ms": e2e_p50,
                "end_to_end_latency_p95_ms": e2e_p95,
                "retrieval_observation_count": len(retrieval_latencies),
                "tool_observation_count": len(tool_latencies),
                "end_to_end_observation_count": len(end_to_end_latencies),
                "effective_modes": dict(effective_modes),
                "generation_modes": dict(generation_modes),
                "fallback_count": fallback_count,
                "failure_categories": dict(failure_categories),
                "index_status": index_status,
                "index_error": index_error,
                "examples": example_results,
            }
    return report


def print_summary(report: dict[str, Any]) -> None:
    header = (
        "mode       task     recall@5  precision@3  grounded  actions   "
        "e2e_p50  e2e_p95  failures"
    )
    print(header)
    print("-" * len(header))
    for mode, metrics in report["modes"].items():
        print(
            f"{mode:<10} "
            f"{metrics['task_success_rate']:<8.4f} "
            f"{metrics['recall_at_5']:<9.4f} "
            f"{metrics['precision_at_3']:<12.4f} "
            f"{metrics['grounded_answer_accuracy']:<9.4f} "
            f"{metrics['action_safety_rate']:<9.4f} "
            f"{metrics['end_to_end_latency_p50_ms']:<8.3f} "
            f"{metrics['end_to_end_latency_p95_ms']:<8.3f} "
            f"{metrics['failed_scenario_runs']}"
        )
    generation = report["generation_evaluation"]
    print(
        "generation evaluation: "
        f"requested={generation['requested']} "
        f"declared_effective={generation['declared_effective']}"
    )


def main() -> int:
    settings = Settings.from_environment()
    parser = argparse.ArgumentParser(
        description="Run deterministic or explicitly configured end-to-end evaluations."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=settings.evaluation_dataset_path,
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["lexical", "semantic", "hybrid"],
        default=list(settings.evaluation_modes),
    )
    parser.add_argument("--runs", type=int, default=settings.evaluation_runs)
    parser.add_argument(
        "--output", type=Path, default=settings.evaluation_results_path
    )
    parser.add_argument(
        "--use-configured-llm",
        action=argparse.BooleanOptionalAction,
        default=settings.evaluation_use_configured_llm,
        help=(
            "Use the environment-configured LLM when credentials exist. "
            "Otherwise the report explicitly records deterministic fallback."
        ),
    )
    args = parser.parse_args()
    configure_logging(
        settings.evaluation_log_level,
        json_logs=settings.log_json,
        trace_enabled=settings.trace_enabled,
    )
    report = evaluate_modes(
        dataset_path=args.dataset,
        modes=args.modes,
        runs=args.runs,
        settings=settings,
        use_configured_llm=args.use_configured_llm,
    )
    print_summary(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
