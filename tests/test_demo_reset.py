from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from servicing_agent.actions import ControlledNoteService
from servicing_agent.auth import DEMO_USERS
from servicing_agent.database import Database
from servicing_agent.demo_reset import main
from servicing_agent.sla import SLAService


def test_demo_reset_restores_canonical_state_and_is_repeatable(tmp_path) -> None:
    database = Database(tmp_path / "reset.db")
    database.initialize()
    user = DEMO_USERS["demo-agent-alex"]
    note_service = ControlledNoteService(database)
    preview = note_service.prepare_note(
        "CASE-4821",
        "Temporary demo note.",
        user,
        operation_id="op-reset-test-note-001",
    )
    note_service.confirm_note(
        preview.operation_id, user, customer_confirmed=True
    )
    SLAService(
        database,
        clock=lambda: datetime(2026, 11, 8, 12, tzinfo=timezone.utc),
    ).check_ticket("CASE-4821", user)
    database.simulate_external_ticket_update(
        "CASE-4821",
        owner="Temporary Demo Queue",
        updated_at="2026-11-08T13:00:00Z",
    )

    first = database.reset_demo_data()
    second = database.reset_demo_data()

    ticket = database.fetch_ticket("CASE-4821")
    assert first["ticket_notes"] == 1
    assert first["case_handoffs"] == 1
    assert first["operations"] == 2
    assert second["ticket_notes"] == 0
    assert second["case_handoffs"] == 0
    assert second["operations"] == 0
    assert ticket["owner"] == "Tier 2 Support"
    assert ticket["record_version"] == 1
    assert ticket["sla_deadline"] == "2026-11-06T21:30:00Z"
    assert database.fetch_ticket_notes("CASE-4821") == []
    assert database.fetch_handoffs("CASE-4821") == []


def test_demo_reset_cli_reports_target_database(
    tmp_path, monkeypatch, capsys
) -> None:
    database_path = tmp_path / "cli-reset.db"
    monkeypatch.setattr(
        sys,
        "argv",
        ["servicing_agent.demo_reset", "--database", str(database_path)],
    )

    assert main() == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "reset"
    assert result["database"] == str(database_path)
    assert result["synthetic_data_only"] is True
    assert Database(database_path).fetch_ticket("CASE-4821") is not None
