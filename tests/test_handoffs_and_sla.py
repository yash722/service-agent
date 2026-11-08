from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from servicing_agent.auth import DEMO_USERS
from servicing_agent.database import Database
from servicing_agent.handoffs import (
    CaseHandoffService,
    InvalidHandoffTransitionError,
)
from servicing_agent.service import ChatService
from servicing_agent.sla import SLAService
from servicing_agent.tools import CRMAuthorizationError


def build_services(tmp_path):
    database = Database(tmp_path / "handoffs.db")
    database.initialize()
    chat = ChatService(database)
    handoffs = CaseHandoffService(
        database,
        chat.crm_tool,
        chat.entitlement_tool,
        chat.evidence_tool,
    )
    return database, chat, handoffs


def test_authorized_handoff_contains_permitted_context_and_evidence(tmp_path) -> None:
    _, _, handoffs = build_services(tmp_path)
    alex = DEMO_USERS["demo-agent-alex"]

    handoff = handoffs.create(
        "CASE-4821",
        "Repeated telemetry delays require specialist review.",
        ["Restarted the connector", "Observed three delayed sync windows"],
        alex,
        operation_id="op-handoff-create-001",
    )

    assert handoff.state == "CREATED"
    assert handoff.queue == "Primary Escalation Queue (simulated)"
    assert set(handoff.case_context) == {
        "ticket_id",
        "subject",
        "status",
        "priority",
        "owner",
        "updated_at",
    }
    assert handoff.source_references
    assert handoff.troubleshooting_history == [
        "Restarted the connector",
        "Observed three delayed sync windows",
    ]

    with pytest.raises(CRMAuthorizationError):
        handoffs.get(handoff.handoff_id, DEMO_USERS["demo-agent-jordan"])


def test_unauthorized_user_cannot_create_handoff(tmp_path) -> None:
    _, _, handoffs = build_services(tmp_path)

    with pytest.raises(CRMAuthorizationError):
        handoffs.create(
            "CASE-4821",
            "Attempted unauthorized escalation.",
            [],
            DEMO_USERS["demo-agent-jordan"],
        )


def test_handoff_state_machine_and_idempotent_transition(tmp_path) -> None:
    _, _, handoffs = build_services(tmp_path)
    user = DEMO_USERS["demo-agent-alex"]
    created = handoffs.create(
        "CASE-4821",
        "Human review required.",
        [],
        user,
        operation_id="op-handoff-state-create-001",
    )

    with pytest.raises(InvalidHandoffTransitionError):
        handoffs.resolve(created.handoff_id, user)

    accepted = handoffs.accept(
        created.handoff_id,
        user,
        operation_id="op-handoff-state-accept-001",
    )
    accepted_again = handoffs.accept(
        created.handoff_id,
        user,
        operation_id="op-handoff-state-accept-001",
    )
    resolved = handoffs.resolve(
        created.handoff_id,
        user,
        operation_id="op-handoff-state-resolve-001",
    )

    assert accepted.state == "ACCEPTED"
    assert accepted_again.state == "ACCEPTED"
    assert resolved.state == "RESOLVED"
    assert resolved.assigned_to == user.user_id


def test_sla_trigger_is_deterministic_and_idempotent(tmp_path) -> None:
    database = Database(tmp_path / "sla.db")
    database.initialize()
    now = datetime(2026, 11, 8, 12, tzinfo=timezone.utc)
    sla = SLAService(database, clock=lambda: now)

    first = sla.check_ticket("CASE-4821", DEMO_USERS["demo-agent-alex"])
    second = sla.check_ticket("CASE-4821", DEMO_USERS["demo-agent-alex"])

    assert first.action == "created"
    assert first.overdue is True
    assert second.action == "already_escalated"
    assert second.handoff_id == first.handoff_id
    assert len(database.fetch_handoffs("CASE-4821")) == 1


def test_sla_sweep_routes_unaccepted_handoff_to_backup(tmp_path) -> None:
    database = Database(tmp_path / "sla-backup.db")
    database.initialize()
    created_at = datetime(2026, 11, 8, 12, tzinfo=timezone.utc)
    sla = SLAService(
        database,
        clock=lambda: created_at,
        backup_after=timedelta(minutes=15),
    )
    initial = sla.check_ticket("CASE-4821", DEMO_USERS["demo-agent-alex"])

    sweep = sla.sweep(as_of=created_at + timedelta(minutes=16))
    handoff = database.fetch_handoff(initial.handoff_id)

    assert sweep.backup_routed == 1
    assert handoff["state"] == "CREATED"
    assert handoff["queue"] == "Backup Escalation Queue (simulated)"
    assert handoff["backup_routed_at"] is not None


def test_sla_sweep_does_not_backup_route_accepted_handoff(tmp_path) -> None:
    database, chat, handoffs = build_services(tmp_path)
    created_at = datetime(2026, 11, 8, 12, tzinfo=timezone.utc)
    sla = SLAService(
        database,
        chat.crm_tool,
        clock=lambda: created_at,
        backup_after=timedelta(minutes=15),
    )
    sla_result = sla.check_ticket(
        "CASE-4821", DEMO_USERS["demo-agent-alex"]
    )
    handoffs.accept(
        sla_result.handoff_id,
        DEMO_USERS["demo-agent-alex"],
        operation_id="op-handoff-accept-sla-001",
    )

    routed = sla.route_unaccepted_to_backup(
        as_of=created_at + timedelta(hours=1)
    )
    handoff = database.fetch_handoff(sla_result.handoff_id)

    assert routed == 0
    assert handoff["state"] == "ACCEPTED"
    assert handoff["queue"] == "High Priority SLA Queue (simulated)"


def test_request_time_ticket_status_runs_sla_rule(tmp_path) -> None:
    database = Database(tmp_path / "sla-chat.db")
    database.initialize()
    now = datetime(2026, 11, 8, 12, tzinfo=timezone.utc)
    sla = SLAService(database, clock=lambda: now)
    chat = ChatService(database, sla_service=sla)

    response = chat.chat(
        "What is the status of CASE-4821?",
        DEMO_USERS["demo-agent-alex"],
    )

    assert response.sla is not None
    assert response.sla.overdue is True
    assert response.sla.action == "created"
    assert "deterministic escalation" in response.reply
