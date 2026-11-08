from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from servicing_agent.actions import (
    ControlledNoteService,
    IdempotencyConflictError,
    OperationAuthorizationError,
    StaleConfirmationError,
)
from servicing_agent.auth import DEMO_USERS
from servicing_agent.database import Database


@pytest.fixture()
def action_context(tmp_path):
    database = Database(tmp_path / "actions.db")
    database.initialize()
    return database, ControlledNoteService(database)


def test_note_preview_requires_confirmation_and_writes_once(action_context) -> None:
    database, service = action_context
    user = DEMO_USERS["demo-agent-alex"]

    preview = service.prepare_note(
        "CASE-4821",
        "Customer confirmed that telemetry is current.",
        user,
        operation_id="op-note-preview-001",
    )

    assert preview.state == "PENDING"
    assert preview.preview is not None
    assert preview.preview.status == "In Progress"
    assert database.fetch_ticket_notes("CASE-4821") == []

    completed = service.confirm_note(
        preview.operation_id, user, customer_confirmed=True
    )
    duplicate = service.confirm_note(
        preview.operation_id, user, customer_confirmed=True
    )

    assert completed.state == "COMPLETED"
    assert duplicate.state == "COMPLETED"
    assert duplicate.note_id == completed.note_id
    assert len(database.fetch_ticket_notes("CASE-4821")) == 1


def test_note_authorization_is_revalidated_at_confirmation(action_context) -> None:
    _, service = action_context
    preview = service.prepare_note(
        "CASE-4821",
        "Authorized preview.",
        DEMO_USERS["demo-agent-alex"],
        operation_id="op-note-authorization-001",
    )

    with pytest.raises(OperationAuthorizationError):
        service.confirm_note(
            preview.operation_id,
            DEMO_USERS["demo-agent-jordan"],
            customer_confirmed=True,
        )


def test_note_cannot_save_without_explicit_customer_confirmation(
    action_context,
) -> None:
    database, service = action_context
    user = DEMO_USERS["demo-agent-alex"]
    preview = service.prepare_note(
        "CASE-4821",
        "Awaiting customer approval.",
        user,
        operation_id="op-note-customer-confirmation-001",
    )

    with pytest.raises(ValueError, match="customer confirmation"):
        service.confirm_note(preview.operation_id, user)

    assert database.fetch_operation(preview.operation_id)["state"] == "PENDING"
    assert database.fetch_ticket_notes("CASE-4821") == []


def test_materially_stale_confirmation_fails_without_writing(action_context) -> None:
    database, service = action_context
    user = DEMO_USERS["demo-agent-alex"]
    preview = service.prepare_note(
        "CASE-4821",
        "This preview will become stale.",
        user,
        operation_id="op-note-stale-001",
    )
    database.simulate_external_ticket_update(
        "CASE-4821",
        owner="Emergency Queue",
        updated_at="2026-11-08T12:00:00Z",
    )

    with pytest.raises(StaleConfirmationError) as caught:
        service.confirm_note(
            preview.operation_id, user, customer_confirmed=True
        )

    assert caught.value.response.state == "FAILED"
    assert caught.value.response.error_code == "STALE_CONFIRMATION"
    assert database.fetch_operation(preview.operation_id)["state"] == "FAILED"
    assert database.fetch_ticket_notes("CASE-4821") == []


def test_operation_id_cannot_be_reused_for_different_payload(action_context) -> None:
    _, service = action_context
    user = DEMO_USERS["demo-agent-alex"]
    service.prepare_note(
        "CASE-4821",
        "First payload.",
        user,
        operation_id="op-note-idempotency-001",
    )

    with pytest.raises(IdempotencyConflictError):
        service.prepare_note(
            "CASE-4821",
            "Different payload.",
            user,
            operation_id="op-note-idempotency-001",
        )


def test_timeout_after_success_reconciles_by_operation_id(action_context) -> None:
    database, service = action_context
    user = DEMO_USERS["demo-agent-alex"]
    preview = service.prepare_note(
        "CASE-4821",
        "Save despite a simulated lost acknowledgement.",
        user,
        operation_id="op-note-reconcile-001",
    )

    unknown = service.confirm_note(
        preview.operation_id,
        user,
        customer_confirmed=True,
        simulate_timeout_after_success=True,
    )
    restarted_service = ControlledNoteService(database)
    reconciled = restarted_service.reconcile(preview.operation_id, user)

    assert unknown.state == "UNKNOWN"
    assert reconciled.state == "COMPLETED"
    assert reconciled.note_id == unknown.note_id
    assert len(database.fetch_ticket_notes("CASE-4821")) == 1


def test_concurrent_confirmations_do_not_duplicate_note(action_context) -> None:
    database, service = action_context
    user = DEMO_USERS["demo-agent-alex"]
    preview = service.prepare_note(
        "CASE-4821",
        "Confirm concurrently.",
        user,
        operation_id="op-note-concurrency-001",
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(
            executor.map(
                lambda _: service.confirm_note(
                    preview.operation_id, user, customer_confirmed=True
                ),
                range(2),
            )
        )

    assert {response.state for response in responses} == {"COMPLETED"}
    assert len({response.note_id for response in responses}) == 1
    assert len(database.fetch_ticket_notes("CASE-4821")) == 1


def test_cancelled_preview_cannot_write(action_context) -> None:
    database, service = action_context
    user = DEMO_USERS["demo-agent-alex"]
    preview = service.prepare_note(
        "CASE-4821",
        "Cancel this note.",
        user,
        operation_id="op-note-cancel-001",
    )

    cancelled = service.cancel_note(preview.operation_id, user)
    after_confirm = service.confirm_note(
        preview.operation_id, user, customer_confirmed=True
    )

    assert cancelled.state == "FAILED"
    assert cancelled.error_code == "CANCELLED"
    assert after_confirm.state == "FAILED"
    assert database.fetch_ticket_notes("CASE-4821") == []


def test_operation_logs_state_and_id_without_note_content(
    action_context, caplog
) -> None:
    _, service = action_context
    user = DEMO_USERS["demo-agent-alex"]
    sensitive_note = "Synthetic private troubleshooting detail"

    with caplog.at_level(logging.INFO):
        preview = service.prepare_note(
            "CASE-4821",
            sensitive_note,
            user,
            operation_id="op-note-logging-001",
        )
        service.confirm_note(
            preview.operation_id, user, customer_confirmed=True
        )

    events = [
        record.msg for record in caplog.records if isinstance(record.msg, dict)
    ]
    transitions = [
        event for event in events if event.get("event") == "operation.transition"
    ]
    assert any(
        event.get("operation_id") == preview.operation_id
        and event.get("state") == "COMPLETED"
        and "latency_ms" in event
        for event in transitions
    )
    assert sensitive_note not in json.dumps(events)
