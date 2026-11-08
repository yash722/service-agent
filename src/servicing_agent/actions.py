from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any
from uuid import uuid4

from .action_workflow import NoteActionState, build_note_action_workflow
from .auth import DemoUserContext
from .database import Database
from .logging_config import log_event
from .schemas import NotePreview, OperationResponse
from .time_utils import to_utc_iso, utc_now
from .tools import CRMAuthorizationError, MockCRMTool, TicketNotFoundError

logger = logging.getLogger(__name__)


class OperationNotFoundError(LookupError):
    pass


class OperationAuthorizationError(PermissionError):
    pass


class IdempotencyConflictError(RuntimeError):
    pass


class StaleConfirmationError(RuntimeError):
    def __init__(self, response: OperationResponse):
        super().__init__(response.message)
        self.response = response


def _request_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ControlledNoteService:
    """Prepare/confirm CRM-note writes against the local simulated CRM."""

    def __init__(self, database: Database, crm_tool: MockCRMTool | None = None):
        self.database = database
        self.crm_tool = crm_tool or MockCRMTool(database)
        self.workflow = build_note_action_workflow(self._execute_workflow_command)

    def prepare_note(
        self,
        ticket_id: str,
        note: str,
        user_context: DemoUserContext,
        *,
        operation_id: str | None = None,
    ) -> OperationResponse:
        return self._invoke(
            command="prepare",
            user_context=user_context,
            ticket_id=ticket_id.upper(),
            note=note.strip(),
            operation_id=operation_id,
        )

    def confirm_note(
        self,
        operation_id: str,
        user_context: DemoUserContext,
        *,
        customer_confirmed: bool = False,
        simulate_timeout_after_success: bool = False,
    ) -> OperationResponse:
        return self._invoke(
            command="confirm",
            user_context=user_context,
            operation_id=operation_id,
            customer_confirmed=customer_confirmed,
            simulate_timeout_after_success=simulate_timeout_after_success,
        )

    def cancel_note(
        self, operation_id: str, user_context: DemoUserContext
    ) -> OperationResponse:
        return self._invoke(
            command="cancel",
            user_context=user_context,
            operation_id=operation_id,
        )

    def reconcile(
        self, operation_id: str, user_context: DemoUserContext
    ) -> OperationResponse:
        return self._invoke(
            command="reconcile",
            user_context=user_context,
            operation_id=operation_id,
        )

    def get_operation(
        self, operation_id: str, user_context: DemoUserContext
    ) -> OperationResponse:
        return self._invoke(
            command="status",
            user_context=user_context,
            operation_id=operation_id,
        )

    def _invoke(self, **state: Any) -> OperationResponse:
        result = self.workflow.invoke({**state, "steps": []})
        return result["result"]

    def _execute_workflow_command(
        self, state: NoteActionState
    ) -> OperationResponse:
        command = state["command"]
        if command == "prepare":
            return self._prepare(
                state["ticket_id"] or "",
                state["note"] or "",
                state["user_context"],
                state.get("operation_id"),
            )
        operation_id = state.get("operation_id") or ""
        if command == "confirm":
            return self._confirm(
                operation_id,
                state["user_context"],
                state.get("customer_confirmed", False),
                state.get("simulate_timeout_after_success", False),
            )
        if command == "cancel":
            return self._cancel(operation_id, state["user_context"])
        if command == "reconcile":
            return self._reconcile(operation_id, state["user_context"])
        return self._status(operation_id, state["user_context"])

    def _prepare(
        self,
        ticket_id: str,
        note: str,
        user_context: DemoUserContext,
        requested_operation_id: str | None,
    ) -> OperationResponse:
        started = time.perf_counter()
        if not note:
            raise ValueError("A non-empty note is required")
        ticket = self.crm_tool.get_ticket(ticket_id, user_context)
        if requested_operation_id and requested_operation_id.startswith("system-"):
            raise ValueError("Operation IDs beginning with 'system-' are reserved")
        operation_id = requested_operation_id or f"op-note-{uuid4()}"
        payload = {
            "ticket_id": ticket["ticket_id"],
            "note": note,
            "preview": {
                "ticket_id": ticket["ticket_id"],
                "subject": ticket["subject"],
                "status": ticket["status"],
                "owner": ticket["owner"],
                "note": note,
                "expected_ticket_version": ticket["record_version"],
            },
        }
        request_hash = _request_hash(
            {"ticket_id": ticket["ticket_id"], "note": note}
        )
        now = to_utc_iso(utc_now())
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if existing:
                self._validate_idempotent_reuse(
                    dict(existing), user_context, "TICKET_NOTE", request_hash
                )
                row = dict(existing)
            else:
                connection.execute(
                    """
                    INSERT INTO operations (
                        operation_id, operation_type, ticket_id, requested_by,
                        state, request_hash, payload_json, expected_ticket_version,
                        created_at, updated_at
                    ) VALUES (?, 'TICKET_NOTE', ?, ?, 'PENDING', ?, ?, ?, ?, ?)
                    """,
                    (
                        operation_id,
                        ticket["ticket_id"],
                        user_context.user_id,
                        request_hash,
                        json.dumps(payload, separators=(",", ":")),
                        ticket["record_version"],
                        now,
                        now,
                    ),
                )
                row = dict(
                    connection.execute(
                        "SELECT * FROM operations WHERE operation_id = ?",
                        (operation_id,),
                    ).fetchone()
                )
        response = self._operation_response(
            row,
            "Review the simulated ticket-note preview, then explicitly confirm or cancel.",
        )
        self._log_transition(
            operation_id,
            None if not existing else row["state"],
            row["state"],
            started,
        )
        return response

    def _confirm(
        self,
        operation_id: str,
        user_context: DemoUserContext,
        customer_confirmed: bool,
        simulate_timeout_after_success: bool,
    ) -> OperationResponse:
        started = time.perf_counter()
        if not customer_confirmed:
            log_event(
                logger,
                "operation.rejected",
                operation_id=operation_id,
                operation_type="TICKET_NOTE",
                error_code="CUSTOMER_CONFIRMATION_REQUIRED",
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            raise ValueError(
                "Explicit customer confirmation is required before saving a note."
            )
        operation = self._authorized_operation(operation_id, user_context)
        self.crm_tool.get_ticket(operation["ticket_id"], user_context)
        previous_state = operation["state"]
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise OperationNotFoundError(operation_id)
            operation = dict(row)
            if operation["state"] != "PENDING":
                response = self._operation_response(
                    operation,
                    self._existing_state_message(operation["state"]),
                )
                self._log_transition(
                    operation_id,
                    operation["state"],
                    operation["state"],
                    started,
                )
                return response

            payload = json.loads(operation["payload_json"])
            if operation["request_hash"] != _request_hash(
                {"ticket_id": payload["ticket_id"], "note": payload["note"]}
            ):
                raise IdempotencyConflictError("Stored operation payload is invalid")
            ticket_row = connection.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?",
                (operation["ticket_id"],),
            ).fetchone()
            if ticket_row is None:
                raise TicketNotFoundError(operation["ticket_id"])
            ticket = dict(ticket_row)
            if ticket["record_version"] != operation["expected_ticket_version"]:
                now = to_utc_iso(utc_now())
                connection.execute(
                    """
                    UPDATE operations
                    SET state = 'FAILED', error_code = 'STALE_CONFIRMATION',
                        error_message = ?, updated_at = ?
                    WHERE operation_id = ? AND state = 'PENDING'
                    """,
                    (
                        "The ticket changed after the preview was prepared.",
                        now,
                        operation_id,
                    ),
                )
                failed = dict(
                    connection.execute(
                        "SELECT * FROM operations WHERE operation_id = ?",
                        (operation_id,),
                    ).fetchone()
                )
                response = self._operation_response(
                    failed,
                    "Confirmation rejected because the ticket changed materially; prepare a new preview.",
                )
                self._log_transition(
                    operation_id, previous_state, "FAILED", started, "STALE_CONFIRMATION"
                )
                # Persist the terminal FAILED state before surfacing the conflict.
                connection.commit()
                raise StaleConfirmationError(response)

            note_id = f"NOTE-{uuid4()}"
            now = to_utc_iso(utc_now())
            final_state = "UNKNOWN" if simulate_timeout_after_success else "COMPLETED"
            connection.execute(
                """
                INSERT INTO ticket_notes (
                    note_id, operation_id, ticket_id, body, created_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    note_id,
                    operation_id,
                    operation["ticket_id"],
                    payload["note"],
                    user_context.user_id,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE tickets
                SET updated_at = ?, record_version = record_version + 1
                WHERE ticket_id = ? AND record_version = ?
                """,
                (
                    now,
                    operation["ticket_id"],
                    operation["expected_ticket_version"],
                ),
            )
            connection.execute(
                """
                UPDATE operations
                SET state = ?, result_json = ?, updated_at = ?
                WHERE operation_id = ? AND state = 'PENDING'
                """,
                (
                    final_state,
                    json.dumps({"note_id": note_id}, separators=(",", ":")),
                    now,
                    operation_id,
                ),
            )
            completed = dict(
                connection.execute(
                    "SELECT * FROM operations WHERE operation_id = ?",
                    (operation_id,),
                ).fetchone()
            )
        message = (
            "The simulated CRM note was saved, but acknowledgement was lost; reconcile before retrying."
            if final_state == "UNKNOWN"
            else "The simulated CRM ticket note was saved."
        )
        self._log_transition(
            operation_id, previous_state, final_state, started
        )
        return self._operation_response(completed, message)

    def _cancel(
        self, operation_id: str, user_context: DemoUserContext
    ) -> OperationResponse:
        started = time.perf_counter()
        operation = self._authorized_operation(operation_id, user_context)
        self.crm_tool.get_ticket(operation["ticket_id"], user_context)
        previous_state = operation["state"]
        if previous_state == "PENDING":
            now = to_utc_iso(utc_now())
            with self.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    UPDATE operations
                    SET state = 'FAILED', error_code = 'CANCELLED',
                        error_message = 'Cancelled by the requesting user', updated_at = ?
                    WHERE operation_id = ? AND state = 'PENDING'
                    """,
                    (now, operation_id),
                )
                operation = dict(
                    connection.execute(
                        "SELECT * FROM operations WHERE operation_id = ?",
                        (operation_id,),
                    ).fetchone()
                )
        self._log_transition(
            operation_id, previous_state, operation["state"], started
        )
        return self._operation_response(
            operation,
            "The pending simulated note was cancelled."
            if operation.get("error_code") == "CANCELLED"
            else self._existing_state_message(operation["state"]),
        )

    def _reconcile(
        self, operation_id: str, user_context: DemoUserContext
    ) -> OperationResponse:
        started = time.perf_counter()
        operation = self._authorized_operation(operation_id, user_context)
        self.crm_tool.get_ticket(operation["ticket_id"], user_context)
        previous_state = operation["state"]
        if previous_state == "UNKNOWN":
            with self.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                note = connection.execute(
                    "SELECT note_id FROM ticket_notes WHERE operation_id = ?",
                    (operation_id,),
                ).fetchone()
                if note:
                    now = to_utc_iso(utc_now())
                    connection.execute(
                        """
                        UPDATE operations
                        SET state = 'COMPLETED', result_json = ?, updated_at = ?
                        WHERE operation_id = ? AND state = 'UNKNOWN'
                        """,
                        (
                            json.dumps(
                                {"note_id": note["note_id"]},
                                separators=(",", ":"),
                            ),
                            now,
                            operation_id,
                        ),
                    )
                operation = dict(
                    connection.execute(
                        "SELECT * FROM operations WHERE operation_id = ?",
                        (operation_id,),
                    ).fetchone()
                )
        message = (
            "Reconciliation found the note by operation ID and confirmed completion."
            if previous_state == "UNKNOWN" and operation["state"] == "COMPLETED"
            else self._existing_state_message(operation["state"])
        )
        self._log_transition(
            operation_id, previous_state, operation["state"], started
        )
        return self._operation_response(operation, message)

    def _status(
        self, operation_id: str, user_context: DemoUserContext
    ) -> OperationResponse:
        operation = self._authorized_operation(operation_id, user_context)
        self.crm_tool.get_ticket(operation["ticket_id"], user_context)
        return self._operation_response(
            operation, self._existing_state_message(operation["state"])
        )

    def _authorized_operation(
        self, operation_id: str, user_context: DemoUserContext
    ) -> dict[str, Any]:
        operation = self.database.fetch_operation(operation_id)
        if operation is None:
            raise OperationNotFoundError(operation_id)
        if operation["requested_by"] != user_context.user_id:
            raise OperationAuthorizationError(
                "The operation is not available to this demo user."
            )
        return operation

    @staticmethod
    def _validate_idempotent_reuse(
        operation: dict[str, Any],
        user_context: DemoUserContext,
        operation_type: str,
        request_hash: str,
    ) -> None:
        if operation["requested_by"] != user_context.user_id:
            raise OperationAuthorizationError(
                "The operation is not available to this demo user."
            )
        if (
            operation["operation_type"] != operation_type
            or operation["request_hash"] != request_hash
        ):
            raise IdempotencyConflictError(
                "The operation ID was already used for a different request."
            )

    @staticmethod
    def _existing_state_message(state: str) -> str:
        return {
            "PENDING": "The operation is awaiting explicit confirmation.",
            "COMPLETED": "The operation was already completed; no duplicate write occurred.",
            "FAILED": "The operation failed or was cancelled and cannot be confirmed.",
            "UNKNOWN": "The operation outcome is unknown; reconcile before retrying.",
        }[state]

    @staticmethod
    def _operation_response(
        operation: dict[str, Any], message: str
    ) -> OperationResponse:
        payload = json.loads(operation["payload_json"])
        result = (
            json.loads(operation["result_json"])
            if operation.get("result_json")
            else {}
        )
        preview = payload.get("preview")
        return OperationResponse(
            operation_id=operation["operation_id"],
            operation_type=operation["operation_type"],
            state=operation["state"],
            ticket_id=operation["ticket_id"],
            message=message,
            preview=NotePreview(**preview) if preview else None,
            note_id=result.get("note_id"),
            error_code=operation.get("error_code"),
            created_at=operation["created_at"],
            updated_at=operation["updated_at"],
        )

    @staticmethod
    def _log_transition(
        operation_id: str,
        previous_state: str | None,
        state: str,
        started: float,
        error_code: str | None = None,
    ) -> None:
        log_event(
            logger,
            "operation.transition",
            operation_id=operation_id,
            operation_type="TICKET_NOTE",
            previous_state=previous_state,
            state=state,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            error_code=error_code,
        )
