from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from servicing_agent.run_streamlit import command as streamlit_command
from servicing_agent.settings import (
    PROJECT_ROOT,
    ROOT_ENV_FILE,
    SUPPORTED_ENVIRONMENT_VARIABLES,
    ConfigurationError,
    Settings,
    snowflake_integration_requested,
)


def _keys(path: Path) -> set[str]:
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        keys.add(stripped.split("=", 1)[0].strip())
    return keys


def test_env_and_example_cover_every_supported_variable() -> None:
    expected = set(SUPPORTED_ENVIRONMENT_VARIABLES)

    assert expected == _keys(ROOT_ENV_FILE)
    assert expected == _keys(PROJECT_ROOT / ".env.example")


def test_process_environment_overrides_dotenv(monkeypatch) -> None:
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")
    monkeypatch.setenv("RETRIEVAL_MODEL", "test-embedding-model")
    monkeypatch.setenv("RETRIEVAL_CACHE_DIR", "data/test-cache")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "7")
    monkeypatch.setenv("RETRIEVAL_RRF_K", "42")
    monkeypatch.setenv("SERVICE_AGENT_LLM_MODEL", "test-generation-model")
    monkeypatch.setenv("SERVICE_AGENT_LLM_API_BASE", "https://llm.example.test/v1")
    monkeypatch.setenv("SERVICE_AGENT_LLM_TIMEOUT_SECONDS", "9")
    monkeypatch.setenv("SERVICE_AGENT_LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("SERVICE_AGENT_LLM_MAX_TOKENS", "333")
    monkeypatch.setenv("SERVICE_AGENT_LLM_REASONING_EFFORT", "none")
    monkeypatch.setenv("SERVICE_AGENT_API_PORT", "8123")
    monkeypatch.setenv("SERVICE_AGENT_API_URL", "http://127.0.0.1:8123")
    monkeypatch.setenv("SERVICE_AGENT_UI_API_READ_TIMEOUT_SECONDS", "222")

    settings = Settings.from_environment()

    assert settings.retrieval_mode == "hybrid"
    assert settings.retrieval_model == "test-embedding-model"
    assert settings.retrieval_cache_dir == PROJECT_ROOT / "data" / "test-cache"
    assert settings.retrieval_top_k == 7
    assert settings.retrieval_rrf_k == 42
    assert settings.llm_model == "test-generation-model"
    assert settings.llm_api_base == "https://llm.example.test/v1"
    assert settings.llm_timeout_seconds == 9
    assert settings.llm_temperature == 0.2
    assert settings.llm_max_tokens == 333
    assert settings.llm_reasoning_effort == "none"
    assert settings.api_port == 8123
    assert settings.api_url == "http://127.0.0.1:8123"
    assert settings.ui_api_read_timeout_seconds == 222


def test_root_dotenv_loads_from_a_different_working_directory(tmp_path) -> None:
    expected_mode = Settings.from_environment().retrieval_mode
    code = (
        "import json; from servicing_agent.settings import Settings; "
        "s=Settings.from_environment(); "
        "print(json.dumps({'env': s.env_file.name, "
        "'rooted': s.database_path.is_absolute(), "
        "'mode': s.retrieval_mode}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload == {
        "env": ".env",
        "rooted": True,
        "mode": expected_mode,
    }


def test_cli_uses_central_settings_from_a_different_directory(tmp_path) -> None:
    database_path = tmp_path / "nested" / "cli-reset.db"
    environment = os.environ.copy()
    environment["SERVICE_AGENT_DB_PATH"] = str(database_path)
    result = subprocess.run(
        [sys.executable, "-m", "servicing_agent.demo_reset"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "reset"
    assert database_path.is_file()


def test_optional_integrations_can_be_empty_for_offline_mode(monkeypatch) -> None:
    monkeypatch.setenv("HISTORY_BACKEND", "mock")
    monkeypatch.setenv("SERVICE_AGENT_LLM_PROVIDER", "none")
    monkeypatch.setenv("RUN_SNOWFLAKE_INTEGRATION", "false")
    for name in (
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_ROLE",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
        "SNOWFLAKE_PRIVATE_KEY_FILE",
        "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
        "SNOWFLAKE_PRIVATE_KEY_PASSWORD",
        "SERVICE_AGENT_LLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_environment()

    assert settings.history_backend == "mock"
    assert settings.llm_api_key is None
    assert settings.snowflake_account is None


def test_disabled_live_test_flag_does_not_validate_snowflake(monkeypatch) -> None:
    monkeypatch.setenv("RUN_SNOWFLAKE_INTEGRATION", "false")
    monkeypatch.setenv("HISTORY_BACKEND", "snowflake")
    for name in (
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_ROLE",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
        "SNOWFLAKE_PRIVATE_KEY_FILE",
        "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
        "SNOWFLAKE_PRIVATE_KEY_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)

    assert snowflake_integration_requested() is False


def test_secrets_are_redacted_from_settings_repr(monkeypatch) -> None:
    monkeypatch.setenv("SERVICE_AGENT_LLM_API_KEY", "unit-test-llm-secret")
    monkeypatch.setenv(
        "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "unit-test-key-passphrase"
    )

    rendered = repr(Settings.from_environment())

    assert "unit-test-llm-secret" not in rendered
    assert "unit-test-key-passphrase" not in rendered


def test_invalid_configuration_has_clear_variable_name(monkeypatch) -> None:
    monkeypatch.setenv("RETRIEVAL_MODE", "unsupported")

    with pytest.raises(ConfigurationError, match="RETRIEVAL_MODE"):
        Settings.from_environment()


def test_streamlit_launcher_uses_central_ui_settings(monkeypatch) -> None:
    monkeypatch.setenv("SERVICE_AGENT_UI_HOST", "0.0.0.0")
    monkeypatch.setenv("SERVICE_AGENT_UI_PORT", "8600")
    settings = Settings.from_environment()

    launch = streamlit_command(settings)

    assert launch[launch.index("--server.address") + 1] == "0.0.0.0"
    assert launch[launch.index("--server.port") + 1] == "8600"
