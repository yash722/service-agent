from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_optional_integrations(request, monkeypatch) -> None:
    """Keep ordinary tests offline regardless of a developer's root .env."""

    snowflake_marker = request.node.get_closest_marker(
        "snowflake_integration"
    )
    ollama_marker = request.node.get_closest_marker("ollama_integration")
    if snowflake_marker is None:
        monkeypatch.setenv("HISTORY_BACKEND", "mock")
        monkeypatch.setenv("RUN_SNOWFLAKE_INTEGRATION", "false")
    if ollama_marker is None:
        monkeypatch.setenv("SERVICE_AGENT_LLM_PROVIDER", "none")
        monkeypatch.delenv("SERVICE_AGENT_LLM_API_KEY", raising=False)
        monkeypatch.setenv("RUN_OLLAMA_INTEGRATION", "false")
    if snowflake_marker is None and ollama_marker is None:
        monkeypatch.setenv("RETRIEVAL_MODE", "lexical")
