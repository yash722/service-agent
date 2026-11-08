from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
import pytest

from servicing_agent.api import create_app


REQUIRED_ENVIRONMENT = (
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_ROLE",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
    "SNOWFLAKE_PRIVATE_KEY_FILE",
    "SNOWFLAKE_PRIVATE_KEY_PASSWORD",
)


@pytest.mark.snowflake_integration
@pytest.mark.skipif(
    os.getenv("RUN_SNOWFLAKE_INTEGRATION") != "1",
    reason="Set RUN_SNOWFLAKE_INTEGRATION=1 to query real Snowflake history.",
)
def test_real_snowflake_history_through_fastapi_and_langgraph(
    tmp_path: Path, monkeypatch
) -> None:
    missing = [name for name in REQUIRED_ENVIRONMENT if not os.getenv(name)]
    assert not missing, f"Missing required Snowflake environment names: {missing}"
    monkeypatch.setenv("HISTORY_BACKEND", "snowflake")
    monkeypatch.setenv("RETRIEVAL_MODE", "lexical")
    app = create_app(tmp_path / "snowflake-integration.db", configure_logs=False)

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
