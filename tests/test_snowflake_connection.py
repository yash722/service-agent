from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from servicing_agent.api import create_app
from servicing_agent.settings import Settings, snowflake_integration_requested


@pytest.mark.snowflake_integration
def test_real_snowflake_history_through_fastapi_and_langgraph(
    tmp_path: Path, monkeypatch
) -> None:
    if not snowflake_integration_requested():
        pytest.skip(
            "Set RUN_SNOWFLAKE_INTEGRATION=1 to query real Snowflake history."
        )
    monkeypatch.setenv("HISTORY_BACKEND", "snowflake")
    monkeypatch.setenv("RETRIEVAL_MODE", "lexical")
    settings = Settings.from_environment()
    app = create_app(
        tmp_path / "snowflake-integration.db",
        configure_logs=False,
        settings_override=settings,
    )

    async def invoke() -> httpx.Response:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                return await client.post(
                    "/chat",
                    headers={"X-Demo-User": "demo-agent-alex"},
                    json={
                        "message": (
                            "Using customer activity, case changes, and support "
                            "interactions, summarize the history of CASE-4821."
                        )
                    },
                )

    response = asyncio.run(invoke())

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "complex_request"
    assert body["history_debug"]["backend"] == "snowflake"
    assert body["history_debug"]["status"] == "available"
    assert body["history_debug"]["record_count"] > 0
    assert body["history_debug"]["fallback_used"] is False
    assert any(
        citation["source_id"].startswith("SF-")
        for citation in body["citations"]
    )
    history_citations = [
        citation
        for citation in body["citations"]
        if citation["source_id"].startswith("SF-")
    ]
    assert any(
        citation["effective_date"].startswith("2026-11-05")
        for citation in history_citations
    )
    assert all(
        not citation["effective_date"].startswith("2026-11-09")
        for citation in history_citations
    )
