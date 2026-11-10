from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Protocol, Sequence

from .auth import DemoUserContext

if TYPE_CHECKING:
    from .settings import Settings


HistoryStatus = Literal["available", "empty", "unavailable", "not_applicable"]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_TABLES = ("CUSTOMER_ACTIVITY", "CASE_HISTORY", "SUPPORT_INTERACTIONS")
_IDENTIFIER_COLUMNS = (
    "ACTIVITY_ID",
    "HISTORY_ID",
    "INTERACTION_ID",
    "EVENT_ID",
    "RECORD_ID",
    "ID",
)
_TIMESTAMP_COLUMNS = (
    "EVENT_TIME",
    "ACTIVITY_TIMESTAMP",
    "ACTIVITY_TS",
    "EVENT_TIMESTAMP",
    "EVENT_TS",
    "INTERACTION_TIMESTAMP",
    "INTERACTION_TS",
    "OCCURRED_AT",
    "CREATED_AT",
    "UPDATED_AT",
    "STATUS_DATE",
)
_TITLE_COLUMNS = (
    "ACTIVITY_TYPE",
    "EVENT_TYPE",
    "INTERACTION_TYPE",
    "CHANNEL",
    "SUBJECT",
    "STATUS",
)
_SUMMARY_COLUMNS = (
    "DESCRIPTION",
    "SUMMARY",
    "DETAILS",
    "NOTES",
    "OUTCOME",
    "STATUS",
)
_SELECTABLE_COLUMNS = frozenset(
    (
        "CUSTOMER_ID",
        "CASE_ID",
        *_IDENTIFIER_COLUMNS,
        *_TIMESTAMP_COLUMNS,
        *_TITLE_COLUMNS,
        *_SUMMARY_COLUMNS,
    )
)


class HistoryError(RuntimeError):
    """Base exception whose message is safe to return to application code."""

    category = "history_error"


class HistoryConfigurationError(HistoryError):
    category = "configuration"


class HistoryAuthenticationError(HistoryError):
    category = "authentication"


class HistoryAuthorizationError(PermissionError, HistoryError):
    category = "authorization"


class HistoryQueryError(HistoryError):
    category = "query"


@dataclass(frozen=True)
class HistoryRecord:
    source_id: str
    source_table: str
    customer_id: str
    case_id: str | None
    event_at: str | None
    title: str
    summary: str

    @property
    def citation_label(self) -> str:
        return f"[{self.source_id} record]"


@dataclass(frozen=True)
class HistoryResult:
    records: tuple[HistoryRecord, ...] = ()
    backend: str = "mock"
    status: HistoryStatus = "empty"
    retrieved_at: str | None = None
    freshness_at: str | None = None
    latency_ms: float = 0.0
    table_counts: Mapping[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    failure_category: str | None = None
    fallback_used: bool = False

    @classmethod
    def not_applicable(cls, backend: str) -> "HistoryResult":
        return cls(backend=backend, status="not_applicable")

    @classmethod
    def unavailable(
        cls, backend: str, category: str, latency_ms: float = 0.0
    ) -> "HistoryResult":
        return cls(
            backend=backend,
            status="unavailable",
            warnings=(
                "The configured Snowflake history source is unavailable. "
                "No mock history was substituted.",
            ),
            failure_category=category,
            fallback_used=False,
            latency_ms=latency_ms,
        )


class HistoryAdapter(Protocol):
    backend: str

    def fetch_case_history(
        self,
        ticket: Mapping[str, Any],
        user_context: DemoUserContext,
    ) -> HistoryResult: ...


@dataclass(frozen=True)
class SnowflakeHistoryConfig:
    account: str
    user: str
    role: str
    warehouse: str
    database: str
    schema: str
    private_key_file: Path
    private_key_password: str = field(repr=False)
    login_timeout_seconds: int = 20
    network_timeout_seconds: int = 30
    query_timeout_seconds: int = 20
    row_limit_per_table: int = 25
    query_tag: str = "fde_servicing_agent_history_read"
    application: str = "FDE_SERVICING_AGENT_POC"

    @classmethod
    def from_settings(cls, settings: "Settings") -> "SnowflakeHistoryConfig":
        missing = settings.missing_snowflake_variables()
        if missing:
            raise HistoryConfigurationError(
                "Snowflake history is enabled but required environment variables "
                f"are missing: {', '.join(sorted(missing))}."
            )
        key_path = settings.snowflake_private_key_file
        if key_path is None or not key_path.is_file():
            raise HistoryConfigurationError(
                "SNOWFLAKE_PRIVATE_KEY_FILE does not reference a readable file."
            )
        config = cls(
            account=str(settings.snowflake_account),
            user=str(settings.snowflake_user),
            role=str(settings.snowflake_role),
            warehouse=str(settings.snowflake_warehouse),
            database=str(settings.snowflake_database),
            schema=str(settings.snowflake_schema),
            private_key_file=key_path,
            private_key_password=str(settings.snowflake_private_key_password),
            login_timeout_seconds=settings.snowflake_login_timeout_seconds,
            network_timeout_seconds=settings.snowflake_network_timeout_seconds,
            query_timeout_seconds=settings.snowflake_query_timeout_seconds,
            row_limit_per_table=settings.snowflake_history_row_limit,
            query_tag=settings.snowflake_query_tag,
            application=settings.snowflake_application,
        )
        for name in ("role", "warehouse", "database", "schema"):
            _validate_identifier(getattr(config, name), name)
        return config


class MockHistoryAdapter:
    """Network-free adapter; records are injectable and empty by default."""

    backend = "mock"

    def __init__(self, records: Sequence[HistoryRecord] = ()):
        self.records = tuple(records)

    def fetch_case_history(
        self,
        ticket: Mapping[str, Any],
        user_context: DemoUserContext,
    ) -> HistoryResult:
        started = time.perf_counter()
        _authorize_ticket(ticket, user_context)
        customer_id = str(ticket["customer_id"])
        case_id = str(ticket["ticket_id"])
        selected = tuple(
            record
            for record in self.records
            if record.customer_id == customer_id
            and record.case_id in {None, case_id}
        )
        retrieved_at = _utc_now()
        table_counts = {
            table: sum(record.source_table == table for record in selected)
            for table in _TABLES
        }
        return HistoryResult(
            records=selected,
            backend=self.backend,
            status="available" if selected else "empty",
            retrieved_at=retrieved_at,
            freshness_at=_freshness(selected),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            table_counts=table_counts,
        )


class SnowflakeHistoryAdapter:
    """Read-only, case-scoped Snowflake customer-history adapter."""

    backend = "snowflake"

    def __init__(
        self,
        config: SnowflakeHistoryConfig,
        *,
        connector_factory: Callable[..., Any] | None = None,
    ):
        self.config = config
        self._connector_factory = connector_factory

    def fetch_case_history(
        self,
        ticket: Mapping[str, Any],
        user_context: DemoUserContext,
    ) -> HistoryResult:
        started = time.perf_counter()
        _authorize_ticket(ticket, user_context)
        customer_id = str(ticket["customer_id"])
        case_id = str(ticket["ticket_id"])
        connection: Any | None = None
        cursor: Any | None = None
        try:
            connection = self._connect()
            cursor = connection.cursor()
            self._verify_session(cursor)
            records: list[HistoryRecord] = []
            table_counts: dict[str, int] = {}
            for table in _TABLES:
                table_records = self._query_table(
                    cursor,
                    table=table,
                    customer_id=customer_id,
                    case_id=case_id,
                )
                records.extend(table_records)
                table_counts[table] = len(table_records)
        except HistoryError:
            raise
        except Exception as exc:  # connector/runtime boundary
            raise _safe_connector_error(exc) from exc
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

        records.sort(
            key=lambda item: (item.event_at or "", item.source_id),
            reverse=True,
        )
        retrieved_at = _utc_now()
        return HistoryResult(
            records=tuple(records),
            backend=self.backend,
            status="available" if records else "empty",
            retrieved_at=retrieved_at,
            freshness_at=_freshness(records),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            table_counts=table_counts,
            fallback_used=False,
        )

    def _connect(self) -> Any:
        factory = self._connector_factory
        if factory is None:
            try:
                import snowflake.connector
            except ImportError as exc:
                raise HistoryConfigurationError(
                    "Snowflake history requires the snowflake-connector-python package."
                ) from exc
            factory = snowflake.connector.connect
        try:
            return factory(
                account=self.config.account,
                user=self.config.user,
                authenticator="SNOWFLAKE_JWT",
                private_key_file=str(self.config.private_key_file),
                private_key_password=self.config.private_key_password,
                role=self.config.role,
                warehouse=self.config.warehouse,
                database=self.config.database,
                schema=self.config.schema,
                login_timeout=self.config.login_timeout_seconds,
                network_timeout=self.config.network_timeout_seconds,
                client_session_keep_alive=False,
                application=self.config.application,
                session_parameters={
                    "QUERY_TAG": self.config.query_tag,
                    "STATEMENT_TIMEOUT_IN_SECONDS": self.config.query_timeout_seconds,
                },
            )
        except Exception as exc:
            raise _safe_connector_error(exc) from exc

    def _verify_session(self, cursor: Any) -> None:
        cursor.execute(
            "SELECT CURRENT_ROLE(), CURRENT_DATABASE(), CURRENT_SCHEMA()",
            timeout=self.config.query_timeout_seconds,
        )
        row = cursor.fetchone()
        if not row or len(row) < 3:
            raise HistoryAuthorizationError(
                "Snowflake did not return a verifiable read context."
            )
        expected = (
            self.config.role.upper(),
            self.config.database.upper(),
            self.config.schema.upper(),
        )
        actual = tuple(str(value).upper() for value in row[:3])
        if actual != expected:
            raise HistoryAuthorizationError(
                "Snowflake session context does not match the configured read context."
            )

    def _query_table(
        self,
        cursor: Any,
        *,
        table: str,
        customer_id: str,
        case_id: str,
    ) -> list[HistoryRecord]:
        columns = self._table_columns(cursor, table)
        if "CUSTOMER_ID" not in columns:
            raise HistoryQueryError(
                f"{table} does not expose the required customer authorization key."
            )
        requires_case = table in {"CASE_HISTORY", "SUPPORT_INTERACTIONS"}
        if requires_case and "CASE_ID" not in columns:
            raise HistoryQueryError(
                f"{table} does not expose the required case authorization key."
            )

        selected = [column for column in columns if column in _SELECTABLE_COLUMNS]
        if not selected:
            raise HistoryQueryError(f"{table} has no approved evidence columns.")
        order_column = next(
            (column for column in _TIMESTAMP_COLUMNS if column in selected),
            None,
        )
        where = '"CUSTOMER_ID" = %s'
        params: list[Any] = [customer_id]
        if requires_case:
            where += ' AND "CASE_ID" = %s'
            params.append(case_id)
        order = f' ORDER BY "{order_column}" DESC' if order_column else ""
        select_list = ", ".join(f'"{column}"' for column in selected)
        sql = (
            f'SELECT {select_list} FROM "{table}" WHERE {where}'
            f"{order} LIMIT %s"
        )
        params.append(self.config.row_limit_per_table)
        cursor.execute(
            sql,
            tuple(params),
            timeout=self.config.query_timeout_seconds,
        )
        rows = cursor.fetchall()
        description = _description_names(cursor.description)
        records = [
            _history_record(table, dict(zip(description, row, strict=True)))
            for row in rows
        ]
        if any(record.customer_id != customer_id for record in records):
            raise HistoryAuthorizationError(
                "Snowflake returned history outside the authorized customer scope."
            )
        if requires_case and any(record.case_id != case_id for record in records):
            raise HistoryAuthorizationError(
                "Snowflake returned history outside the authorized case scope."
            )
        return records

    def _table_columns(self, cursor: Any, table: str) -> tuple[str, ...]:
        if table not in _TABLES:
            raise HistoryQueryError(
                "An unsupported Snowflake history table was requested."
            )
        database = self.config.database.upper()
        _validate_identifier(database, "database")
        sql = (
            f'SELECT COLUMN_NAME FROM "{database}".INFORMATION_SCHEMA.COLUMNS '
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s ORDER BY ORDINAL_POSITION"
        )
        cursor.execute(
            sql,
            (self.config.schema.upper(), table),
            timeout=self.config.query_timeout_seconds,
        )
        return tuple(str(row[0]).upper() for row in cursor.fetchall())


def _validate_identifier(value: str, label: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise HistoryConfigurationError(
            f"SNOWFLAKE_{label.upper()} is not a valid Snowflake identifier."
        )


def _authorize_ticket(
    ticket: Mapping[str, Any], user_context: DemoUserContext
) -> None:
    customer_id = ticket.get("customer_id")
    case_id = ticket.get("ticket_id")
    if not isinstance(customer_id, str) or not isinstance(case_id, str):
        raise HistoryAuthorizationError("Untrusted case context was rejected.")
    if not user_context.can_access_customer(customer_id):
        raise HistoryAuthorizationError(
            "The selected demo user is not authorized for this history record."
        )


def _description_names(description: Any) -> tuple[str, ...]:
    names: list[str] = []
    for item in description or ():
        name = getattr(item, "name", None)
        if name is None and isinstance(item, (tuple, list)) and item:
            name = item[0]
        if name is None:
            raise HistoryQueryError("Snowflake returned an invalid result schema.")
        names.append(str(name).upper())
    return tuple(names)


def _history_record(table: str, row: Mapping[str, Any]) -> HistoryRecord:
    customer_id = str(row.get("CUSTOMER_ID") or "")
    case_value = row.get("CASE_ID")
    case_id = str(case_value) if case_value is not None else None
    event_at = _first_value(row, _TIMESTAMP_COLUMNS)
    title_value = _first_value(row, _TITLE_COLUMNS)
    summary_values = [
        str(row[column]).strip()
        for column in _SUMMARY_COLUMNS
        if row.get(column) is not None and str(row[column]).strip()
    ]
    summary = " · ".join(dict.fromkeys(summary_values))
    if not summary:
        summary = "Authorized historical event with no narrative summary."
    identifier = _first_value(row, _IDENTIFIER_COLUMNS)
    if identifier is None:
        digest_input = json.dumps(
            {key: str(value) for key, value in sorted(row.items())},
            sort_keys=True,
            separators=(",", ":"),
        )
        identifier = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:12]
    safe_identifier = re.sub(r"[^A-Za-z0-9]+", "-", str(identifier)).strip("-")
    source_id = f"SF-{table.replace('_', '-')}-{safe_identifier}".upper()
    return HistoryRecord(
        source_id=source_id,
        source_table=table,
        customer_id=customer_id,
        case_id=case_id,
        event_at=_normalize_timestamp(event_at),
        title=str(title_value or table.replace("_", " ").title()),
        summary=summary,
    )


def _first_value(row: Mapping[str, Any], names: Sequence[str]) -> Any | None:
    return next((row[name] for name in names if row.get(name) is not None), None)


def _normalize_timestamp(value: Any | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _freshness(records: Sequence[HistoryRecord]) -> str | None:
    timestamps = [record.event_at for record in records if record.event_at]
    return max(timestamps) if timestamps else None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_connector_error(exc: Exception) -> HistoryError:
    code = str(getattr(exc, "errno", ""))
    lowered = str(exc).lower()
    if code in {"250001", "390100", "390144", "390318"} or any(
        marker in lowered
        for marker in ("authentication", "jwt", "private key", "passphrase")
    ):
        return HistoryAuthenticationError(
            "Snowflake authentication failed; no mock history was substituted."
        )
    if any(
        marker in lowered
        for marker in ("not authorized", "insufficient privileges", "current role")
    ):
        return HistoryAuthorizationError(
            "Snowflake authorization failed; no mock history was substituted."
        )
    return HistoryQueryError(
        "Snowflake history query failed; no mock history was substituted."
    )
