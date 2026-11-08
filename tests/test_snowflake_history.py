from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from servicing_agent.auth import DEMO_USERS
from servicing_agent.database import Database
from servicing_agent.history import (
    HistoryAuthenticationError,
    HistoryAuthorizationError,
    SnowflakeHistoryAdapter,
    SnowflakeHistoryConfig,
)
from servicing_agent.service import ChatService


TABLE_COLUMNS = {
    "CUSTOMER_ACTIVITY": (
        "ACTIVITY_ID",
        "CUSTOMER_ID",
        "ACTIVITY_TYPE",
        "ACTIVITY_TS",
        "DESCRIPTION",
    ),
    "CASE_HISTORY": (
        "HISTORY_ID",
        "CASE_ID",
        "CUSTOMER_ID",
        "STATUS",
        "EVENT_TS",
        "DESCRIPTION",
    ),
    "SUPPORT_INTERACTIONS": (
        "INTERACTION_ID",
        "CASE_ID",
        "CUSTOMER_ID",
        "CHANNEL",
        "INTERACTION_TS",
        "SUMMARY",
    ),
}

TABLE_ROWS = {
    "CUSTOMER_ACTIVITY": [
        (
            "ACT-100",
            "CUST-1001",
            "Telemetry alert",
            datetime(2026, 11, 5, 13, tzinfo=timezone.utc),
            "Repeated delayed telemetry windows were recorded.",
        )
    ],
    "CASE_HISTORY": [
        (
            "HIST-100",
            "CASE-4821",
            "CUST-1001",
            "Escalated",
            datetime(2026, 11, 5, 14, tzinfo=timezone.utc),
            "Tier 2 escalation followed repeated delays.",
        )
    ],
    "SUPPORT_INTERACTIONS": [
        (
            "INT-100",
            "CASE-4821",
            "CUST-1001",
            "Phone",
            datetime(2026, 11, 5, 15, tzinfo=timezone.utc),
            "Customer reported a delay near twenty minutes.",
        )
    ],
}


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.description = None
        self.rows = []
        self.closed = False
        self.session_row = (
            "SERVICING_HISTORY_READER",
            "SERVICING_AGENT_DEMO",
            "HISTORY",
        )

    def execute(self, sql, params=None, timeout=None):
        self.connection.executions.append((sql, params, timeout))
        if "CURRENT_ROLE()" in sql:
            self.rows = [self.session_row]
            self.description = (
                ("CURRENT_ROLE",),
                ("CURRENT_DATABASE",),
                ("CURRENT_SCHEMA",),
            )
            return self
        if "INFORMATION_SCHEMA.COLUMNS" in sql:
            table = params[1]
            self.rows = [(column,) for column in TABLE_COLUMNS[table]]
            self.description = (("COLUMN_NAME",),)
            return self
        table = next(
            table for table in TABLE_COLUMNS if f'FROM "{table}"' in sql
        )
        self.rows = list(TABLE_ROWS[table])
        self.description = tuple((column,) for column in TABLE_COLUMNS[table])
        return self

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self):
        self.executions = []
        self.cursor_instance = FakeCursor(self)
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def close(self):
        self.closed = True


class FakeConnector:
    def __init__(self):
        self.connection = FakeConnection()
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.connection


def config(tmp_path: Path) -> SnowflakeHistoryConfig:
    key_path = tmp_path / "test-key.p8"
    key_path.write_text("encrypted-test-placeholder", encoding="utf-8")
    return SnowflakeHistoryConfig(
        account="TEST_ACCOUNT",
        user="TEST_USER",
        role="SERVICING_HISTORY_READER",
        warehouse="SERVICING_POC_WH",
        database="SERVICING_AGENT_DEMO",
        schema="HISTORY",
        private_key_file=key_path,
        private_key_password="test-passphrase",
        query_timeout_seconds=7,
    )


def ticket() -> dict[str, str]:
    return {"ticket_id": "CASE-4821", "customer_id": "CUST-1001"}


def test_snowflake_adapter_uses_parameterized_read_only_queries_and_cleanup(
    tmp_path,
) -> None:
    connector = FakeConnector()
    adapter = SnowflakeHistoryAdapter(
        config(tmp_path), connector_factory=connector
    )

    result = adapter.fetch_case_history(
        ticket(), DEMO_USERS["demo-agent-alex"]
    )

    assert result.status == "available"
    assert len(result.records) == 3
    assert result.freshness_at == "2026-11-05T15:00:00Z"
    assert result.table_counts == {
        "CUSTOMER_ACTIVITY": 1,
        "CASE_HISTORY": 1,
        "SUPPORT_INTERACTIONS": 1,
    }
    assert all(record.source_id.startswith("SF-") for record in result.records)
    assert connector.connection.cursor_instance.closed is True
    assert connector.connection.closed is True
    assert connector.kwargs["authenticator"] == "SNOWFLAKE_JWT"
    assert connector.kwargs["client_session_keep_alive"] is False

    for sql, params, timeout in connector.connection.executions:
        assert sql.lstrip().upper().startswith("SELECT")
        assert timeout == 7
        if 'FROM "CUSTOMER_ACTIVITY"' in sql:
            assert params == ("CUST-1001", 25)
        elif 'FROM "CASE_HISTORY"' in sql or 'FROM "SUPPORT_INTERACTIONS"' in sql:
            assert params == ("CUST-1001", "CASE-4821", 25)
        if 'FROM "' in sql and "INFORMATION_SCHEMA" not in sql:
            assert "CUST-1001" not in sql
            assert "CASE-4821" not in sql


def test_unauthorized_context_never_opens_snowflake_connection(tmp_path) -> None:
    called = False

    def connector(**kwargs):
        nonlocal called
        called = True
        raise AssertionError((kwargs, "connector should not be called"))

    adapter = SnowflakeHistoryAdapter(
        config(tmp_path), connector_factory=connector
    )

    with pytest.raises(HistoryAuthorizationError):
        adapter.fetch_case_history(ticket(), DEMO_USERS["demo-agent-jordan"])

    assert called is False


def test_server_context_mismatch_is_rejected_and_connection_is_closed(
    tmp_path,
) -> None:
    connector = FakeConnector()
    connector.connection.cursor_instance.session_row = (
        "ACCOUNTADMIN",
        "SERVICING_AGENT_DEMO",
        "HISTORY",
    )
    adapter = SnowflakeHistoryAdapter(
        config(tmp_path), connector_factory=connector
    )

    with pytest.raises(HistoryAuthorizationError):
        adapter.fetch_case_history(ticket(), DEMO_USERS["demo-agent-alex"])

    assert connector.connection.cursor_instance.closed is True
    assert connector.connection.closed is True
    assert all(
        "CUSTOMER_ACTIVITY" not in sql
        for sql, _params, _timeout in connector.connection.executions
    )


def test_langgraph_combines_authorized_history_with_local_rag(tmp_path) -> None:
    database = Database(tmp_path / "history-workflow.db")
    database.initialize()
    connector = FakeConnector()
    history = SnowflakeHistoryAdapter(
        config(tmp_path), connector_factory=connector
    )
    service = ChatService(
        database,
        as_of=date(2026, 11, 7),
        history_adapter=history,
    )

    response = service.chat(
        "Why was CASE-4821 escalated?", DEMO_USERS["demo-agent-alex"]
    )
    query_count = len(connector.connection.executions)
    status_response = service.chat(
        "What is the status of CASE-4821?", DEMO_USERS["demo-agent-alex"]
    )

    returned_ids = {citation.source_id for citation in response.citations}
    assert "CASE-NOTE-4821" in returned_ids
    assert any(source_id.startswith("SF-") for source_id in returned_ids)
    assert "Snowflake history records" in response.reply
    assert response.history_debug is not None
    assert response.history_debug.backend == "snowflake"
    assert response.history_debug.status == "available"
    assert response.history_debug.record_count == 3
    assert status_response.history_debug is None
    assert status_response.citations == []
    assert len(connector.connection.executions) == query_count


def test_snowflake_failure_is_visible_and_never_falls_back_to_mock(tmp_path) -> None:
    class FailingAdapter:
        backend = "snowflake"

        def fetch_case_history(self, ticket, user_context):
            raise HistoryAuthenticationError("safe test failure")

    database = Database(tmp_path / "history-failure.db")
    database.initialize()
    service = ChatService(
        database,
        as_of=date(2026, 11, 7),
        history_adapter=FailingAdapter(),
    )

    response = service.chat(
        "Why was CASE-4821 escalated?", DEMO_USERS["demo-agent-alex"]
    )

    assert response.history_debug is not None
    assert response.history_debug.backend == "snowflake"
    assert response.history_debug.status == "unavailable"
    assert response.history_debug.failure_category == "authentication"
    assert response.history_debug.fallback_used is False
    assert all(not item.source_id.startswith("SF-") for item in response.citations)
    assert any("No mock history was substituted" in item for item in response.warnings)


def test_unauthorized_langgraph_request_never_invokes_history_adapter(
    tmp_path,
) -> None:
    class GuardedAdapter:
        backend = "snowflake"

        def __init__(self):
            self.called = False

        def fetch_case_history(self, ticket, user_context):
            self.called = True
            raise AssertionError("unauthorized request reached history adapter")

    database = Database(tmp_path / "history-authorization.db")
    database.initialize()
    adapter = GuardedAdapter()
    service = ChatService(database, history_adapter=adapter)

    response = service.chat(
        "Why was CASE-4821 escalated?", DEMO_USERS["demo-agent-jordan"]
    )

    assert adapter.called is False
    assert response.history_debug is None
    assert response.citations == []


def test_snowflake_config_does_not_reveal_passphrase(tmp_path, monkeypatch) -> None:
    key_path = tmp_path / "encrypted-key.p8"
    key_path.write_text("encrypted-test-placeholder", encoding="utf-8")
    values = {
        "SNOWFLAKE_ACCOUNT": "TEST_ACCOUNT",
        "SNOWFLAKE_USER": "TEST_USER",
        "SNOWFLAKE_ROLE": "SERVICING_HISTORY_READER",
        "SNOWFLAKE_WAREHOUSE": "SERVICING_POC_WH",
        "SNOWFLAKE_DATABASE": "SERVICING_AGENT_DEMO",
        "SNOWFLAKE_SCHEMA": "HISTORY",
        "SNOWFLAKE_PRIVATE_KEY_FILE": str(key_path),
        "SNOWFLAKE_PRIVATE_KEY_PASSWORD": "do-not-display",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    loaded = SnowflakeHistoryConfig.from_environment()

    assert loaded.private_key_password == "do-not-display"
    assert "do-not-display" not in repr(loaded)
