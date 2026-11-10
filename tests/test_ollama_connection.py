from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from servicing_agent.api import create_app
from servicing_agent.settings import Settings, ollama_integration_requested


@pytest.mark.ollama_integration
def test_local_ollama_generates_grounded_case_answer(
    tmp_path: Path, monkeypatch
) -> None:
    if not ollama_integration_requested():
        pytest.skip(
            "Set RUN_OLLAMA_INTEGRATION=true to query local Ollama."
        )

    # Keep this smoke test focused on local generation. Snowflake has its own
    # opt-in integration test and the application never sends unauthorized data.
    monkeypatch.setenv("HISTORY_BACKEND", "mock")
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")
    settings = Settings.from_environment()
    assert settings.llm_provider == "openai_compatible"
    assert settings.llm_model == "qwen3:4b-instruct"
    assert settings.llm_api_base == "http://localhost:11434/v1"

    app = create_app(
        tmp_path / "ollama-integration.db",
        configure_logs=False,
        settings_override=settings,
    )

    async def invoke() -> tuple[httpx.Response, httpx.Response]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                investigation = await client.post(
                    "/chat",
                    headers={"X-Demo-User": "demo-agent-alex"},
                    json={
                        "message": "Why is CASE-4821 still under investigation?"
                    },
                )
                guarantee = await client.post(
                    "/chat",
                    headers={"X-Demo-User": "demo-agent-alex"},
                    json={
                        "message": (
                            "Can you guarantee CASE-4821 will be resolved tomorrow?"
                        )
                    },
                )
                return investigation, guarantee

    response, guarantee_response = asyncio.run(invoke())

    assert response.status_code == 200
    body = response.json()
    assert body["generation_mode"] in {"llm", "deterministic"}
    if body["generation_mode"] == "deterministic":
        assert body["generation_debug"]["fallback_reason"] in {
            "LLM output was incomplete (finish_reason=length).",
            "LLM output exposed reasoning or planning text.",
        }
    assert body["evidence_status"] == "sufficient"
    assert "investigat" in body["reply"].lower()
    assert "We are given" not in body["reply"]
    assert body["reply"].rstrip()[-1] in ".!?])"
    assert any(
        citation["source_id"] == "CASE-NOTE-4821"
        for citation in body["citations"]
    )
    assert not any("LLM call failed" in warning for warning in body["warnings"])
    assert not any(
        "citation validation" in warning for warning in body["warnings"]
    )

    assert guarantee_response.status_code == 200
    guarantee = guarantee_response.json()
    assert any(
        marker in guarantee["reply"].lower()
        for marker in ("cannot guarantee", "can't guarantee", "no guarantee")
    )
    assert "cannot guarantee" in guarantee["reply"].lower()
    assert "three consecutive synchronization windows" not in guarantee["reply"]
    assert {
        citation["source_id"] for citation in guarantee["citations"]
    }.issubset({"CRM-CASE-4821"})

    print(
        json.dumps(
            {
                "investigation": {
                    "mode": body["generation_mode"],
                    "fallback_reason": body["generation_debug"][
                        "fallback_reason"
                    ],
                    "latency_ms": body["generation_debug"]["latency_ms"],
                    "citation_ids": [
                        citation["source_id"] for citation in body["citations"]
                    ],
                    "answer_relevant": "investigat" in body["reply"].lower(),
                },
                "guarantee": {
                    "mode": guarantee["generation_mode"],
                    "fallback_reason": guarantee["generation_debug"][
                        "fallback_reason"
                    ],
                    "latency_ms": guarantee["generation_debug"]["latency_ms"],
                    "citation_ids": [
                        citation["source_id"]
                        for citation in guarantee["citations"]
                    ],
                    "answer_relevant": "guarantee"
                    in guarantee["reply"].lower(),
                },
            },
            sort_keys=True,
        )
    )
