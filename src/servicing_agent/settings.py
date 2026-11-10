from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT_ENV_FILE = PROJECT_ROOT / ".env"

SUPPORTED_ENVIRONMENT_VARIABLES = (
    "SERVICE_AGENT_DB_PATH",
    "SERVICE_AGENT_SQLITE_TIMEOUT_SECONDS",
    "SERVICE_AGENT_SQLITE_BUSY_TIMEOUT_MS",
    "SERVICE_AGENT_LOG_LEVEL",
    "SERVICE_AGENT_LOG_JSON",
    "SERVICE_AGENT_TRACE_ENABLED",
    "SERVICE_AGENT_API_HOST",
    "SERVICE_AGENT_API_PORT",
    "SERVICE_AGENT_API_RELOAD",
    "SERVICE_AGENT_API_URL",
    "SERVICE_AGENT_API_REQUEST_TIMEOUT_SECONDS",
    "SERVICE_AGENT_UI_HOST",
    "SERVICE_AGENT_UI_PORT",
    "SERVICE_AGENT_UI_HEADLESS",
    "SERVICE_AGENT_UI_PAGE_TITLE",
    "SERVICE_AGENT_UI_PAGE_ICON",
    "SERVICE_AGENT_UI_LAYOUT",
    "SERVICE_AGENT_UI_GATHER_USAGE_STATS",
    "SERVICE_AGENT_UI_API_READ_TIMEOUT_SECONDS",
    "SERVICE_AGENT_LLM_PROVIDER",
    "SERVICE_AGENT_LLM_API_KEY",
    "SERVICE_AGENT_LLM_MODEL",
    "SERVICE_AGENT_LLM_API_BASE",
    "SERVICE_AGENT_LLM_TIMEOUT_SECONDS",
    "SERVICE_AGENT_LLM_TEMPERATURE",
    "SERVICE_AGENT_LLM_MAX_TOKENS",
    "SERVICE_AGENT_LLM_REASONING_EFFORT",
    "RETRIEVAL_MODE",
    "RETRIEVAL_MODEL",
    "RETRIEVAL_CACHE_DIR",
    "RETRIEVAL_TOP_K",
    "RETRIEVAL_RRF_K",
    "RETRIEVAL_SEMANTIC_MIN_SCORE",
    "RETRIEVAL_SEMANTIC_MIN_TOP_SCORE",
    "HISTORY_BACKEND",
    "HISTORY_ANSWER_RECORD_LIMIT",
    "HISTORY_LLM_RECORD_LIMIT",
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_ROLE",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
    "SNOWFLAKE_PRIVATE_KEY_FILE",
    "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
    "SNOWFLAKE_LOGIN_TIMEOUT_SECONDS",
    "SNOWFLAKE_NETWORK_TIMEOUT_SECONDS",
    "SNOWFLAKE_QUERY_TIMEOUT_SECONDS",
    "SNOWFLAKE_HISTORY_ROW_LIMIT",
    "SNOWFLAKE_QUERY_TAG",
    "SNOWFLAKE_APPLICATION",
    "SLA_BACKUP_AFTER_MINUTES",
    "EVALUATION_DATASET_PATH",
    "EVALUATION_RESULTS_PATH",
    "EVALUATION_MODES",
    "EVALUATION_RUNS",
    "EVALUATION_USE_CONFIGURED_LLM",
    "EVALUATION_LOG_LEVEL",
    "EVALUATION_AS_OF_DATE",
    "EVALUATION_CLOCK_UTC",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "RUN_SNOWFLAKE_INTEGRATION",
    "RUN_OLLAMA_INTEGRATION",
)

# Resolve the one local configuration file from this module rather than the
# working directory. Existing process variables win for deployment and tests.
load_dotenv(ROOT_ENV_FILE, override=False)


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    project_root: Path
    env_file: Path

    database_path: Path
    sqlite_timeout_seconds: float
    sqlite_busy_timeout_ms: int

    log_level: str
    log_json: bool
    trace_enabled: bool

    api_host: str
    api_port: int
    api_reload: bool
    api_url: str
    api_request_timeout_seconds: float

    ui_host: str
    ui_port: int
    ui_headless: bool
    ui_page_title: str
    ui_page_icon: str
    ui_layout: str
    ui_gather_usage_stats: bool
    ui_api_read_timeout_seconds: float

    llm_provider: str
    llm_model: str
    llm_api_base: str
    llm_api_key: str | None = field(repr=False)
    llm_timeout_seconds: float
    llm_temperature: float
    llm_max_tokens: int
    llm_reasoning_effort: str | None

    retrieval_mode: str
    retrieval_model: str
    retrieval_cache_dir: Path
    retrieval_top_k: int
    retrieval_rrf_k: int
    retrieval_semantic_min_score: float
    retrieval_semantic_min_top_score: float

    history_backend: str
    history_answer_record_limit: int
    history_llm_record_limit: int

    snowflake_account: str | None
    snowflake_user: str | None
    snowflake_role: str | None
    snowflake_warehouse: str | None
    snowflake_database: str | None
    snowflake_schema: str | None
    snowflake_private_key_file: Path | None
    snowflake_private_key_password: str | None = field(repr=False)
    snowflake_login_timeout_seconds: int
    snowflake_network_timeout_seconds: int
    snowflake_query_timeout_seconds: int
    snowflake_history_row_limit: int
    snowflake_query_tag: str
    snowflake_application: str

    sla_backup_after_minutes: int

    evaluation_dataset_path: Path
    evaluation_results_path: Path
    evaluation_modes: tuple[str, ...]
    evaluation_runs: int
    evaluation_use_configured_llm: bool
    evaluation_log_level: str
    evaluation_as_of_date: date
    evaluation_clock_utc: datetime

    hf_hub_offline: bool
    transformers_offline: bool
    run_snowflake_integration: bool
    run_ollama_integration: bool

    @classmethod
    def from_environment(cls) -> "Settings":
        api_host = _text("SERVICE_AGENT_API_HOST", "127.0.0.1")
        api_port = _integer("SERVICE_AGENT_API_PORT", 8000)
        api_url = _optional_text("SERVICE_AGENT_API_URL") or (
            f"http://127.0.0.1:{api_port}"
        )
        settings = cls(
            project_root=PROJECT_ROOT,
            env_file=ROOT_ENV_FILE,
            database_path=_path(
                "SERVICE_AGENT_DB_PATH", "data/service_agent.db"
            ),
            sqlite_timeout_seconds=_floating(
                "SERVICE_AGENT_SQLITE_TIMEOUT_SECONDS", 5.0
            ),
            sqlite_busy_timeout_ms=_integer(
                "SERVICE_AGENT_SQLITE_BUSY_TIMEOUT_MS", 5000
            ),
            log_level=_text("SERVICE_AGENT_LOG_LEVEL", "INFO").upper(),
            log_json=_boolean("SERVICE_AGENT_LOG_JSON", True),
            trace_enabled=_boolean("SERVICE_AGENT_TRACE_ENABLED", True),
            api_host=api_host,
            api_port=api_port,
            api_reload=_boolean("SERVICE_AGENT_API_RELOAD", False),
            api_url=api_url.rstrip("/"),
            api_request_timeout_seconds=_floating(
                "SERVICE_AGENT_API_REQUEST_TIMEOUT_SECONDS", 20.0
            ),
            ui_host=_text("SERVICE_AGENT_UI_HOST", "127.0.0.1"),
            ui_port=_integer("SERVICE_AGENT_UI_PORT", 8501),
            ui_headless=_boolean("SERVICE_AGENT_UI_HEADLESS", True),
            ui_page_title=_text(
                "SERVICE_AGENT_UI_PAGE_TITLE", "FDE Servicing Agent Demo"
            ),
            ui_page_icon=_text("SERVICE_AGENT_UI_PAGE_ICON", "⚡"),
            ui_layout=_text("SERVICE_AGENT_UI_LAYOUT", "wide").lower(),
            ui_gather_usage_stats=_boolean(
                "SERVICE_AGENT_UI_GATHER_USAGE_STATS", False
            ),
            ui_api_read_timeout_seconds=_floating(
                "SERVICE_AGENT_UI_API_READ_TIMEOUT_SECONDS", 240.0
            ),
            llm_provider=_text(
                "SERVICE_AGENT_LLM_PROVIDER", "none"
            ).lower(),
            llm_model=_text(
                "SERVICE_AGENT_LLM_MODEL", "gpt-4.1-mini"
            ),
            llm_api_base=_text(
                "SERVICE_AGENT_LLM_API_BASE", "https://api.openai.com/v1"
            ).rstrip("/"),
            llm_api_key=_optional_text("SERVICE_AGENT_LLM_API_KEY"),
            llm_timeout_seconds=_floating(
                "SERVICE_AGENT_LLM_TIMEOUT_SECONDS", 20.0
            ),
            llm_temperature=_floating(
                "SERVICE_AGENT_LLM_TEMPERATURE", 0.0
            ),
            llm_max_tokens=_integer("SERVICE_AGENT_LLM_MAX_TOKENS", 512),
            llm_reasoning_effort=_optional_text(
                "SERVICE_AGENT_LLM_REASONING_EFFORT"
            ),
            retrieval_mode=_text("RETRIEVAL_MODE", "lexical").lower(),
            retrieval_model=_text(
                "RETRIEVAL_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            retrieval_cache_dir=_path(
                "RETRIEVAL_CACHE_DIR", "data/embedding_cache"
            ),
            retrieval_top_k=_integer("RETRIEVAL_TOP_K", 5),
            retrieval_rrf_k=_integer("RETRIEVAL_RRF_K", 60),
            retrieval_semantic_min_score=_floating(
                "RETRIEVAL_SEMANTIC_MIN_SCORE", 0.25
            ),
            retrieval_semantic_min_top_score=_floating(
                "RETRIEVAL_SEMANTIC_MIN_TOP_SCORE", 0.45
            ),
            history_backend=_text("HISTORY_BACKEND", "mock").lower(),
            history_answer_record_limit=_integer(
                "HISTORY_ANSWER_RECORD_LIMIT", 3
            ),
            history_llm_record_limit=_integer(
                "HISTORY_LLM_RECORD_LIMIT", 5
            ),
            snowflake_account=_optional_text("SNOWFLAKE_ACCOUNT"),
            snowflake_user=_optional_text("SNOWFLAKE_USER"),
            snowflake_role=_optional_text("SNOWFLAKE_ROLE"),
            snowflake_warehouse=_optional_text("SNOWFLAKE_WAREHOUSE"),
            snowflake_database=_optional_text("SNOWFLAKE_DATABASE"),
            snowflake_schema=_optional_text("SNOWFLAKE_SCHEMA"),
            snowflake_private_key_file=_optional_path(
                "SNOWFLAKE_PRIVATE_KEY_FILE"
            ),
            snowflake_private_key_password=(
                _optional_text("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
                # Backward compatibility for configurations created before the
                # passphrase variable name was standardized.
                or _optional_text("SNOWFLAKE_PRIVATE_KEY_PASSWORD")
            ),
            snowflake_login_timeout_seconds=_integer(
                "SNOWFLAKE_LOGIN_TIMEOUT_SECONDS", 20
            ),
            snowflake_network_timeout_seconds=_integer(
                "SNOWFLAKE_NETWORK_TIMEOUT_SECONDS", 30
            ),
            snowflake_query_timeout_seconds=_integer(
                "SNOWFLAKE_QUERY_TIMEOUT_SECONDS", 20
            ),
            snowflake_history_row_limit=_integer(
                "SNOWFLAKE_HISTORY_ROW_LIMIT", 25
            ),
            snowflake_query_tag=_text(
                "SNOWFLAKE_QUERY_TAG", "fde_servicing_agent_history_read"
            ),
            snowflake_application=_text(
                "SNOWFLAKE_APPLICATION", "FDE_SERVICING_AGENT_POC"
            ),
            sla_backup_after_minutes=_integer(
                "SLA_BACKUP_AFTER_MINUTES", 15
            ),
            evaluation_dataset_path=_path(
                "EVALUATION_DATASET_PATH", "evaluations/golden_questions.json"
            ),
            evaluation_results_path=_path(
                "EVALUATION_RESULTS_PATH", "evaluations/latest_results.json"
            ),
            evaluation_modes=_csv(
                "EVALUATION_MODES", ("lexical", "semantic", "hybrid")
            ),
            evaluation_runs=_integer("EVALUATION_RUNS", 3),
            evaluation_use_configured_llm=_boolean(
                "EVALUATION_USE_CONFIGURED_LLM", False
            ),
            evaluation_log_level=_text(
                "EVALUATION_LOG_LEVEL", "WARNING"
            ).upper(),
            evaluation_as_of_date=_date(
                "EVALUATION_AS_OF_DATE", "2026-11-07"
            ),
            evaluation_clock_utc=_datetime(
                "EVALUATION_CLOCK_UTC", "2026-11-08T12:00:00Z"
            ),
            hf_hub_offline=_boolean("HF_HUB_OFFLINE", False),
            transformers_offline=_boolean("TRANSFORMERS_OFFLINE", False),
            run_snowflake_integration=_boolean(
                "RUN_SNOWFLAKE_INTEGRATION", False
            ),
            run_ollama_integration=_boolean("RUN_OLLAMA_INTEGRATION", False),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.log_level not in {
            "CRITICAL",
            "ERROR",
            "WARNING",
            "INFO",
            "DEBUG",
        }:
            raise ConfigurationError("SERVICE_AGENT_LOG_LEVEL is invalid.")
        if self.evaluation_log_level not in {
            "CRITICAL",
            "ERROR",
            "WARNING",
            "INFO",
            "DEBUG",
        }:
            raise ConfigurationError("EVALUATION_LOG_LEVEL is invalid.")
        if self.llm_provider not in {"none", "openai_compatible"}:
            raise ConfigurationError(
                "SERVICE_AGENT_LLM_PROVIDER must be none or openai_compatible."
            )
        if self.retrieval_mode not in {"lexical", "semantic", "hybrid"}:
            raise ConfigurationError(
                "RETRIEVAL_MODE must be lexical, semantic, or hybrid."
            )
        if self.history_backend not in {"mock", "snowflake"}:
            raise ConfigurationError(
                "HISTORY_BACKEND must be mock or snowflake."
            )
        if self.ui_layout not in {"centered", "wide"}:
            raise ConfigurationError(
                "SERVICE_AGENT_UI_LAYOUT must be centered or wide."
            )
        for name, value in (
            ("SERVICE_AGENT_API_PORT", self.api_port),
            ("SERVICE_AGENT_UI_PORT", self.ui_port),
        ):
            if not 1 <= value <= 65535:
                raise ConfigurationError(f"{name} must be between 1 and 65535.")
        for name, value in (
            ("SERVICE_AGENT_SQLITE_TIMEOUT_SECONDS", self.sqlite_timeout_seconds),
            (
                "SERVICE_AGENT_API_REQUEST_TIMEOUT_SECONDS",
                self.api_request_timeout_seconds,
            ),
            (
                "SERVICE_AGENT_UI_API_READ_TIMEOUT_SECONDS",
                self.ui_api_read_timeout_seconds,
            ),
            ("SERVICE_AGENT_LLM_TIMEOUT_SECONDS", self.llm_timeout_seconds),
        ):
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive.")
        for name, value in (
            ("SERVICE_AGENT_SQLITE_BUSY_TIMEOUT_MS", self.sqlite_busy_timeout_ms),
            ("RETRIEVAL_TOP_K", self.retrieval_top_k),
            ("RETRIEVAL_RRF_K", self.retrieval_rrf_k),
            ("HISTORY_ANSWER_RECORD_LIMIT", self.history_answer_record_limit),
            ("HISTORY_LLM_RECORD_LIMIT", self.history_llm_record_limit),
            ("SERVICE_AGENT_LLM_MAX_TOKENS", self.llm_max_tokens),
            ("SNOWFLAKE_LOGIN_TIMEOUT_SECONDS", self.snowflake_login_timeout_seconds),
            (
                "SNOWFLAKE_NETWORK_TIMEOUT_SECONDS",
                self.snowflake_network_timeout_seconds,
            ),
            ("SNOWFLAKE_QUERY_TIMEOUT_SECONDS", self.snowflake_query_timeout_seconds),
            ("SNOWFLAKE_HISTORY_ROW_LIMIT", self.snowflake_history_row_limit),
            ("SLA_BACKUP_AFTER_MINUTES", self.sla_backup_after_minutes),
            ("EVALUATION_RUNS", self.evaluation_runs),
        ):
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive.")
        if not 0 <= self.llm_temperature <= 2:
            raise ConfigurationError(
                "SERVICE_AGENT_LLM_TEMPERATURE must be between 0 and 2."
            )
        for name, value in (
            ("RETRIEVAL_SEMANTIC_MIN_SCORE", self.retrieval_semantic_min_score),
            (
                "RETRIEVAL_SEMANTIC_MIN_TOP_SCORE",
                self.retrieval_semantic_min_top_score,
            ),
        ):
            if not -1 <= value <= 1:
                raise ConfigurationError(f"{name} must be between -1 and 1.")
        invalid_modes = set(self.evaluation_modes) - {
            "lexical",
            "semantic",
            "hybrid",
        }
        if not self.evaluation_modes or invalid_modes:
            raise ConfigurationError(
                "EVALUATION_MODES must contain lexical, semantic, and/or hybrid."
            )
        _validate_url("SERVICE_AGENT_API_URL", self.api_url)
        _validate_url("SERVICE_AGENT_LLM_API_BASE", self.llm_api_base)
        if self.history_backend == "snowflake" or self.run_snowflake_integration:
            missing = self.missing_snowflake_variables()
            if missing:
                raise ConfigurationError(
                    "Snowflake is enabled but required configuration is missing: "
                    + ", ".join(missing)
                    + "."
                )
            private_key_file = self.snowflake_private_key_file
            if private_key_file is None or not private_key_file.is_file():
                raise ConfigurationError(
                    "SNOWFLAKE_PRIVATE_KEY_FILE does not reference a readable file."
                )

    def missing_snowflake_variables(self) -> list[str]:
        fields = (
            ("SNOWFLAKE_ACCOUNT", self.snowflake_account),
            ("SNOWFLAKE_USER", self.snowflake_user),
            ("SNOWFLAKE_ROLE", self.snowflake_role),
            ("SNOWFLAKE_WAREHOUSE", self.snowflake_warehouse),
            ("SNOWFLAKE_DATABASE", self.snowflake_database),
            ("SNOWFLAKE_SCHEMA", self.snowflake_schema),
            ("SNOWFLAKE_PRIVATE_KEY_FILE", self.snowflake_private_key_file),
            (
                "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
                self.snowflake_private_key_password,
            ),
        )
        return [name for name, value in fields if value in {None, ""}]


def snowflake_integration_requested() -> bool:
    """Read only the opt-in flag without validating optional integrations."""

    return _boolean("RUN_SNOWFLAKE_INTEGRATION", False)


def ollama_integration_requested() -> bool:
    """Read only the local Ollama smoke-test opt-in flag."""

    return _boolean("RUN_OLLAMA_INTEGRATION", False)


def _raw(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _text(name: str, default: str) -> str:
    value = _raw(name, default).strip()
    if not value:
        raise ConfigurationError(f"{name} cannot be empty.")
    return value


def _optional_text(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _boolean(name: str, default: bool) -> bool:
    raw = _raw(name, "true" if default else "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false.")


def _integer(name: str, default: int) -> int:
    try:
        return int(_raw(name, str(default)).strip())
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc


def _floating(name: str, default: float) -> float:
    try:
        return float(_raw(name, str(default)).strip())
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number.") from exc


def _path(name: str, default: str) -> Path:
    raw = _raw(name, default).strip()
    if not raw:
        raise ConfigurationError(f"{name} cannot be empty.")
    return _resolve_path(raw)


def _optional_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return _resolve_path(raw) if raw else None


def _resolve_path(raw: str) -> Path:
    expanded = Path(os.path.expandvars(raw)).expanduser()
    if expanded.is_absolute():
        return expanded
    return (PROJECT_ROOT / expanded).resolve()


def _csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _raw(name, ",".join(default))
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


def _date(name: str, default: str) -> date:
    try:
        return date.fromisoformat(_raw(name, default).strip())
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be YYYY-MM-DD.") from exc


def _datetime(name: str, default: str) -> datetime:
    raw = _raw(name, default).strip().replace("Z", "+00:00")
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an ISO-8601 timestamp.") from exc
    if value.tzinfo is None:
        raise ConfigurationError(f"{name} must include a UTC offset or Z.")
    return value.astimezone(UTC)


def _validate_url(name: str, value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError(f"{name} must be an absolute HTTP(S) URL.")
