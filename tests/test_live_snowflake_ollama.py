from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from servicing_agent.api import create_app
from servicing_agent.settings import (
    Settings,
    ollama_integration_requested,
    snowflake_integration_requested,
)


@pytest.mark.snowflake_integration
@pytest.mark.ollama_integration
def test_real_snowflake_and_ollama_end_to_end(tmp_path: Path) -> None:
    if not snowflake_integration_requested():
        pytest.skip("Set RUN_SNOWFLAKE_INTEGRATION=true for live Snowflake.")
    if not ollama_integration_requested():
        pytest.skip("Set RUN_OLLAMA_INTEGRATION=true for local Ollama.")

    settings = Settings.from_environment()
    assert settings.history_backend == "snowflake"
    assert settings.llm_provider == "openai_compatible"
    assert settings.retrieval_mode == "hybrid"
    assert settings.llm_model == "qwen3:4b-instruct"

    app = create_app(
        tmp_path / "snowflake-ollama-integration.db",
        configure_logs=False,
        settings_override=settings,
    )

    questions = (
        "Why is CASE-4821 still under investigation?",
        "Why was CASE-4821 escalated?",
    )

    async def invoke() -> list[httpx.Response]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
                timeout=settings.ui_api_read_timeout_seconds,
            ) as client:
                return [
                    await client.post(
                        "/chat",
                        headers={"X-Demo-User": "demo-agent-alex"},
                        json={"message": question},
                    )
                    for question in questions
                ]

    responses = asyncio.run(invoke())
    metrics = []
    for question, response in zip(questions, responses, strict=True):
        assert response.status_code == 200
        body = response.json()

        assert body["history_debug"]["status"] == "available"
        assert body["history_debug"]["record_count"] >= 3
        assert all(
            body["history_debug"]["table_counts"].get(table, 0) >= 1
            for table in (
                "CUSTOMER_ACTIVITY",
                "CASE_HISTORY",
                "SUPPORT_INTERACTIONS",
            )
        )
        assert body["retrieval_debug"]["effective_mode"] == "hybrid"
        assert body["retrieval_debug"]["fallback_reason"] is None
        # A safe deterministic fallback is not a successful live LLM test.
        assert body["generation_mode"] == "llm", body["generation_debug"]
        assert body["generation_debug"]["fallback_reason"] is None
        assert body["generation_debug"]["finish_reason"] == "stop"
        assert 0 < body["generation_debug"]["output_tokens"] <= 512
        assert body["generation_debug"]["latency_ms"] > 0
        assert body["total_latency_ms"] >= body["generation_debug"]["latency_ms"]
        assert any(
            citation["source_id"] == "CASE-NOTE-4821"
            for citation in body["citations"]
        )
        assert any(
            citation["source_id"].startswith("SF-")
            for citation in body["citations"]
        )
        assert all(
            citation["citation_label"] in body["reply"]
            for citation in body["citations"]
        )
        assert "We are given" not in body["reply"]
        metrics.append(
            {
                "question": question,
                "generation_mode": body["generation_mode"],
                "generation_ms": body["generation_debug"]["latency_ms"],
                "finish_reason": body["generation_debug"]["finish_reason"],
                "output_tokens": body["generation_debug"]["output_tokens"],
                "snowflake_ms": body["history_debug"]["latency_ms"],
                "retrieval_ms": body["retrieval_debug"]["latency_ms"],
                "total_ms": body["total_latency_ms"],
            }
        )

    print(
        json.dumps(
            {
                "model": settings.llm_model,
                "results": metrics,
            },
            sort_keys=True,
        )
    )


@pytest.mark.snowflake_integration
@pytest.mark.ollama_integration
def test_expanded_cases_use_grounded_llm_without_false_fallbacks(
    tmp_path: Path,
) -> None:
    if not snowflake_integration_requested():
        pytest.skip("Set RUN_SNOWFLAKE_INTEGRATION=true for live Snowflake.")
    if not ollama_integration_requested():
        pytest.skip("Set RUN_OLLAMA_INTEGRATION=true for local Ollama.")

    settings = Settings.from_environment()
    app = create_app(
        tmp_path / "expanded-live-grounding.db",
        configure_logs=False,
        settings_override=settings,
    )
    scenarios = (
        {
            "case_id": "CASE-4824",
            "user": "demo-agent-alex",
            "question": "What initial response target applies to CASE-4824?",
            "required_sources": {
                "POL-TELEMETRY-RESPONSE",
                "ENT-1002-TELEMETRY",
            },
            "required_text": ("eight business hours", "standard"),
        },
        {
            "case_id": "CASE-4823",
            "user": "demo-agent-jordan",
            "question": (
                "Does CASE-4823 not being reproduced prove the issue did not occur?"
            ),
            "required_sources": {"CASE-NOTE-4823"},
            "required_text": ("does not prove",),
        },
        {
            "case_id": "CASE-4825",
            "user": "demo-agent-alex",
            "question": (
                "What was the confirmed root cause and completed resolution for "
                "CASE-4825?"
            ),
            "required_sources": {"CASE-NOTE-4825"},
            "required_text": ("expired connector certificate", "completed"),
        },
    )

    async def invoke() -> list[httpx.Response]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
                timeout=settings.ui_api_read_timeout_seconds,
            ) as client:
                return [
                    await client.post(
                        "/chat",
                        headers={"X-Demo-User": scenario["user"]},
                        json={"message": scenario["question"]},
                    )
                    for scenario in scenarios
                ]

    responses = asyncio.run(invoke())
    metrics = []
    for scenario, response in zip(scenarios, responses, strict=True):
        assert response.status_code == 200
        body = response.json()
        assert body["history_debug"]["status"] == "available"
        assert body["retrieval_debug"]["effective_mode"] == "hybrid"
        assert body["retrieval_debug"]["fallback_reason"] is None
        assert body["generation_mode"] == "llm", body["generation_debug"]
        assert body["generation_debug"]["fallback_reason"] is None
        assert body["generation_debug"]["finish_reason"] == "stop"
        assert 0 < body["generation_debug"]["output_tokens"] <= 512
        reply_lower = body["reply"].lower()
        assert all(text in reply_lower for text in scenario["required_text"])
        returned_sources = {
            citation["source_id"] for citation in body["citations"]
        }
        assert scenario["required_sources"].issubset(returned_sources)
        assert all(
            citation["citation_label"] in body["reply"]
            for citation in body["citations"]
        )
        metrics.append(
            {
                "case_id": scenario["case_id"],
                "generation_mode": body["generation_mode"],
                "fallback_reason": body["generation_debug"]["fallback_reason"],
                "generation_ms": body["generation_debug"]["latency_ms"],
                "snowflake_ms": body["history_debug"]["latency_ms"],
                "retrieval_ms": body["retrieval_debug"]["latency_ms"],
                "total_ms": body["total_latency_ms"],
                "citation_ids": sorted(returned_sources),
            }
        )

    print(json.dumps({"model": settings.llm_model, "results": metrics}, sort_keys=True))


@pytest.mark.snowflake_integration
@pytest.mark.ollama_integration
def test_demo_investigation_diagnostics_and_resolution_use_llm(
    tmp_path: Path,
) -> None:
    if not snowflake_integration_requested():
        pytest.skip("Set RUN_SNOWFLAKE_INTEGRATION=true for live Snowflake.")
    if not ollama_integration_requested():
        pytest.skip("Set RUN_OLLAMA_INTEGRATION=true for local Ollama.")

    settings = Settings.from_environment()
    app = create_app(
        tmp_path / "demo-scenarios-live.db",
        configure_logs=False,
        settings_override=settings,
    )
    scenarios = (
        {
            "name": "multi_source",
            "question": (
                "Using the case notes and phone conversation, summarize why "
                "CASE-4821 was escalated and what delay the customer observed."
            ),
            "required_text": ("escalat", "forty to sixty"),
            "required_sources": {"CASE-NOTE-4821"},
        },
        {
            "name": "unknown_diagnostics",
            "question": (
                "Were the requested connector diagnostic logs received for "
                "CASE-4821?"
            ),
            "required_text": ("does not confirm",),
            "required_sources": {"CASE-DIAG-4821"},
        },
        {
            "name": "resolved_incident",
            "question": "What was confirmed and completed to resolve CASE-4825?",
            "required_text": ("expired connector certificate", "completed"),
            "required_sources": {"CASE-NOTE-4825"},
        },
    )

    async def invoke() -> list[httpx.Response]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
                timeout=settings.ui_api_read_timeout_seconds,
            ) as client:
                return [
                    await client.post(
                        "/chat",
                        headers={"X-Demo-User": "demo-agent-alex"},
                        json={"message": scenario["question"]},
                    )
                    for scenario in scenarios
                ]

    responses = asyncio.run(invoke())
    metrics = []
    for scenario, response in zip(scenarios, responses, strict=True):
        assert response.status_code == 200
        body = response.json()
        assert body["history_debug"]["status"] == "available"
        assert body["retrieval_debug"]["effective_mode"] == "hybrid"
        assert body["retrieval_debug"]["fallback_reason"] is None
        assert body["generation_mode"] == "llm", body["generation_debug"]
        assert body["generation_debug"]["fallback_reason"] is None
        reply_lower = body["reply"].lower()
        assert all(text in reply_lower for text in scenario["required_text"])
        if scenario["name"] == "unknown_diagnostics":
            assert "logs were not received" not in reply_lower
        source_ids = {citation["source_id"] for citation in body["citations"]}
        assert scenario["required_sources"].issubset(source_ids)
        assert all(
            citation["citation_label"] in body["reply"]
            for citation in body["citations"]
        )
        if scenario["name"] == "multi_source":
            assert source_ids.intersection(
                {"CALL-4821-20261105", "SF-SUPPORT-INTERACTIONS-INT-001"}
            )
        metrics.append(
            {
                "scenario": scenario["name"],
                "generation_ms": body["generation_debug"]["latency_ms"],
                "total_ms": body["total_latency_ms"],
                "citations": sorted(source_ids),
            }
        )

    print(json.dumps({"model": settings.llm_model, "results": metrics}, sort_keys=True))
