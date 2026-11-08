from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .history import HistoryRecord, HistoryResult
from .retrieval import RetrievalResult, RetrievedEvidence
from .schemas import SourceCitation


@dataclass(frozen=True)
class GenerationOutcome:
    reply: str
    mode: str
    citations: tuple[SourceCitation, ...]
    warnings: tuple[str, ...]


class GroundedAnswerGenerator(Protocol):
    def generate(
        self,
        *,
        question: str,
        ticket: dict[str, Any] | None,
        entitlements: list[dict[str, Any]],
        retrieval: RetrievalResult,
        history: HistoryResult,
    ) -> GenerationOutcome: ...


def _snippet(value: str, limit: int = 320) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit].rsplit(' ', 1)[0]}…"


def document_citation(item: RetrievedEvidence) -> SourceCitation:
    document = item.document
    return SourceCitation(
        source_id=document.source_id,
        document_version=document.version,
        title=document.title,
        document_type=document.document_type,
        effective_date=document.effective_date.isoformat(),
        snippet=_snippet(document.text),
        citation_label=document.citation_label,
    )


def entitlement_citation(entitlement: dict[str, Any]) -> SourceCitation:
    label = f"[{entitlement['entitlement_id']} record]"
    return SourceCitation(
        source_id=entitlement["entitlement_id"],
        document_version="record",
        title=f"Verified {entitlement['product_id']} entitlement",
        document_type="entitlement",
        effective_date=entitlement["start_date"],
        snippet=(
            f"Active {entitlement['subscription_tier']} entitlement for "
            f"{entitlement['product_id']} beginning {entitlement['start_date']}."
        ),
        citation_label=label,
    )


def history_citation(
    record: HistoryRecord, retrieved_at: str | None
) -> SourceCitation:
    return SourceCitation(
        source_id=record.source_id,
        document_version="record",
        title=record.title,
        document_type=f"snowflake_{record.source_table.lower()}",
        effective_date=record.event_at or retrieved_at or "timestamp unavailable",
        snippet=_snippet(record.summary),
        citation_label=record.citation_label,
    )


class DeterministicGroundedGenerator:
    """Evidence-only local fallback. This class does not call an LLM."""

    def generate(
        self,
        *,
        question: str,
        ticket: dict[str, Any] | None,
        entitlements: list[dict[str, Any]],
        retrieval: RetrievalResult,
        history: HistoryResult,
    ) -> GenerationOutcome:
        del question
        warnings = list(retrieval.warnings) + list(history.warnings)
        history_records = list(history.records[:3])
        if history_records:
            warnings = [
                warning
                for warning in warnings
                if warning != "No relevant, current, authorized evidence was found."
            ]
        if retrieval.status == "conflicting":
            citations = tuple(document_citation(item) for item in retrieval.evidence)
            return GenerationOutcome(
                reply=(
                    "I found conflicting applicable evidence and cannot give a "
                    "reliable answer without human review. No production action "
                    "is allowed from this local demo."
                ),
                mode="deterministic",
                citations=citations,
                warnings=tuple(warnings),
            )

        if retrieval.status == "insufficient" and not history_records:
            context = (
                f" for {ticket['ticket_id']}" if ticket else " for this request"
            )
            return GenerationOutcome(
                reply=(
                    f"I found insufficient current, approved, applicable evidence{context}. "
                    "I will not infer an answer. Include an authorized case ID or "
                    "ask a servicing specialist to review the missing evidence. "
                    "No production action is allowed from this local demo."
                ),
                mode="deterministic",
                citations=(),
                warnings=tuple(warnings),
            )

        selected = list(retrieval.evidence[:3])
        citations: list[SourceCitation] = []
        answer_parts: list[str] = []

        policy_evidence = next(
            (
                item
                for item in selected
                if item.document.document_type == "support_policy"
            ),
            None,
        )
        if policy_evidence:
            matching_entitlement = next(
                (
                    entitlement
                    for entitlement in entitlements
                    if entitlement["product_id"]
                    in policy_evidence.document.product_ids
                    and entitlement["subscription_tier"]
                    in policy_evidence.document.subscription_tiers
                ),
                None,
            )
            if matching_entitlement:
                entitlement_source = entitlement_citation(matching_entitlement)
                citations.append(entitlement_source)
                answer_parts.append(
                    f"The customer has a verified "
                    f"{matching_entitlement['subscription_tier']} entitlement for "
                    f"{matching_entitlement['product_id']} "
                    f"{entitlement_source.citation_label}."
                )

        for item in selected:
            citation = document_citation(item)
            citations.append(citation)
            answer_parts.append(f"{item.document.text} {citation.citation_label}")

        for record in history_records:
            citation = history_citation(record, history.retrieved_at)
            citations.append(citation)
            event_context = f" at {record.event_at}" if record.event_at else ""
            answer_parts.append(
                f"Snowflake history records {record.title}{event_context}: "
                f"{record.summary} {citation.citation_label}."
            )

        answer_parts.append(
            "This answer was assembled by the deterministic evidence fallback; "
            "it was not generated by an LLM. No production action is allowed "
            "from this local demo."
        )
        return GenerationOutcome(
            reply=" ".join(answer_parts),
            mode="deterministic",
            citations=tuple(citations),
            warnings=tuple(warnings),
        )


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    api_key: str
    model: str
    api_base: str
    timeout_seconds: float = 20.0


class OpenAICompatibleGroundedGenerator:
    """Optional OpenAI-compatible generator with citation validation."""

    _CITATION_PATTERN = re.compile(r"\[[A-Z0-9-]+\s+(?:v[0-9.]+|record)\]")

    def __init__(self, config: OpenAICompatibleConfig):
        self.config = config
        self.fallback = DeterministicGroundedGenerator()

    def generate(
        self,
        *,
        question: str,
        ticket: dict[str, Any] | None,
        entitlements: list[dict[str, Any]],
        retrieval: RetrievalResult,
        history: HistoryResult,
    ) -> GenerationOutcome:
        if retrieval.status == "conflicting" or (
            retrieval.status != "sufficient" and not history.records
        ):
            return self.fallback.generate(
                question=question,
                ticket=ticket,
                entitlements=entitlements,
                retrieval=retrieval,
                history=history,
            )

        evidence_citations = [
            document_citation(item) for item in retrieval.evidence
        ]
        entitlement_citations = [
            entitlement_citation(entitlement) for entitlement in entitlements
        ]
        history_citations = [
            history_citation(record, history.retrieved_at)
            for record in history.records[:5]
        ]
        allowed = {
            citation.citation_label: citation
            for citation in [
                *evidence_citations,
                *entitlement_citations,
                *history_citations,
            ]
        }
        evidence_lines = [
            f"{citation.citation_label}: {citation.snippet}"
            for citation in allowed.values()
        ]
        prompt = (
            f"Question: {question}\n"
            f"Authorized case: {ticket['ticket_id'] if ticket else 'none'}\n"
            "Evidence:\n"
            + "\n".join(evidence_lines)
        )
        try:
            response = httpx.post(
                f"{self.config.api_base.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Answer only from the supplied evidence. Cite every "
                                "material claim with the exact bracketed citation. "
                                "State uncertainty when evidence is insufficient or "
                                "conflicting. Do not propose data mutations."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            return self._fallback_with_reason(
                question,
                ticket,
                entitlements,
                retrieval,
                history,
                f"LLM call failed: {type(exc).__name__}",
            )

        found_labels = set(self._CITATION_PATTERN.findall(content))
        if not found_labels or not found_labels.issubset(allowed):
            return self._fallback_with_reason(
                question,
                ticket,
                entitlements,
                retrieval,
                history,
                "LLM output failed citation validation.",
            )
        return GenerationOutcome(
            reply=content,
            mode="llm",
            citations=tuple(allowed[label] for label in sorted(found_labels)),
            warnings=(*retrieval.warnings, *history.warnings),
        )

    def _fallback_with_reason(
        self,
        question: str,
        ticket: dict[str, Any] | None,
        entitlements: list[dict[str, Any]],
        retrieval: RetrievalResult,
        history: HistoryResult,
        reason: str,
    ) -> GenerationOutcome:
        fallback = self.fallback.generate(
            question=question,
            ticket=ticket,
            entitlements=entitlements,
            retrieval=retrieval,
            history=history,
        )
        return GenerationOutcome(
            reply=fallback.reply,
            mode=fallback.mode,
            citations=fallback.citations,
            warnings=(*fallback.warnings, reason),
        )
