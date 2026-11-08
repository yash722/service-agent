from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any
from uuid import uuid4

from .actions import (
    IdempotencyConflictError,
    OperationAuthorizationError,
    OperationNotFoundError,
)
from .auth import DemoUserContext
from .database import Database
from .logging_config import log_event
from .schemas import HandoffResponse
from .time_utils import to_utc_iso, utc_now
from .tools import EntitlementTool, EvidenceRetrievalTool, MockCRMTool

logger = logging.getLogger(__name__)


class InvalidHandoffTransitionError(RuntimeError):
    pass


def _hash_request(payload: dict[str, Any]) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class CaseHandoffService:
    """Authorized, idempotent handoffs for the simulated servicing queues."""

    def __init__(
        self,
        database: Database,
        crm_tool: MockCRMTool,
        entitlement_tool: EntitlementTool,
        evidence_tool: EvidenceRetrievalTool,
    ):
        self.database = database
        self.crm_tool = crm_tool
        self.entitlement_tool = entitlement_tool
        self.evidence_tool = evidence_tool

    def create(
        self,
        ticket_id: str,
        reason: str,
        troubleshooting_history: list[str],
        user_context: DemoUserContext,
        *,
        operation_id: str | None = None,
    ) -> HandoffResponse:
        started = time.perf_counter()
        ticket = self.crm_tool.get_ticket(ticket_id, user_context)
        history = [item.strip() for item in troubleshooting_history if item.strip()]
        entitlements = self.entitlement_tool.get_active_entitlements(
            ticket["customer_id"], user_context
        )
        retrieval = self.evidence_tool.search(
            " ".join([reason, *history]),
            user_context=user_context,
            ticket=ticket,
            entitlements=entitlements,
        )
        source_references = list(
            dict.fromkeys(
                item.document.citation_label for item in retrieval.evidence
            )
        )
        case_context = self._permitted_case_context(ticket)
        request_payload = {
            "ticket_id": ticket["ticket_id"],
            "reason": reason.strip(),
            "troubleshooting_history": history,
        }
        request_hash = _hash_request(request_payload)
        if operation_id and operation_id.startswith("system-"):
            raise ValueError("Operation IDs beginning with 'system-' are reserved")
        operation_id = operation_id or f"op-handoff-{uuid4()}"
        now = to_utc_iso(utc_now())
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if existing:
                operation = dict(existing)
                self._validate_operation(
                    operation,
                    user_context,
                    "HANDOFF_CREATE",
                    request_hash,
                )
                handoff = connection.execute(
                    "SELECT * FROM case_handoffs WHERE operation_id = ?",
                    (operation_id,),
                ).fetchone()
                if handoff is None:
                    raise OperationNotFoundError(operation_id)
                handoff_row = dict(handoff)
            else:
                handoff_id = f"HANDOFF-{uuid4()}"
                payload = {
                    **request_payload,
                    "case_context": case_context,
                    "source_references": source_references,
                }
                connection.execute(
                    """
                    INSERT INTO operations (
                        operation_id, operation_type, ticket_id, requested_by,
                        state, request_hash, payload_json, expected_ticket_version,
                        result_json, created_at, updated_at
                    ) VALUES (?, 'HANDOFF_CREATE', ?, ?, 'COMPLETED', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        operation_id,
                        ticket["ticket_id"],
                        user_context.user_id,
                        request_hash,
                        json.dumps(payload, separators=(",", ":")),
                        ticket["record_version"],
                        json.dumps({"handoff_id": handoff_id}),
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO case_handoffs (
                        handoff_id, operation_id, ticket_id, customer_id,
                        trigger_type, state, queue, reason, context_json,
                        source_references_json, troubleshooting_json, created_by,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'USER', 'CREATED', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        handoff_id,
                        operation_id,
                        ticket["ticket_id"],
                        ticket["customer_id"],
                        "Primary Escalation Queue (simulated)",
                        reason.strip(),
                        json.dumps(case_context, separators=(",", ":")),
                        json.dumps(source_references, separators=(",", ":")),
                        json.dumps(history, separators=(",", ":")),
                        user_context.user_id,
                        now,
                        now,
                    ),
                )
                handoff_row = dict(
                    connection.execute(
                        "SELECT * FROM case_handoffs WHERE handoff_id = ?",
                        (handoff_id,),
                    ).fetchone()
                )
        self._log(
            handoff_row["handoff_id"],
            operation_id,
            "CREATED",
            started,
            outcome="existing" if existing else "created",
        )
        return self.to_response(handoff_row)

    def accept(
        self,
        handoff_id: str,
        user_context: DemoUserContext,
        *,
        operation_id: str | None = None,
    ) -> HandoffResponse:
        return self._transition(
            handoff_id,
            "ACCEPTED",
            user_context,
            operation_id=operation_id,
        )

    def resolve(
        self,
        handoff_id: str,
        user_context: DemoUserContext,
        *,
        operation_id: str | None = None,
    ) -> HandoffResponse:
        return self._transition(
            handoff_id,
            "RESOLVED",
            user_context,
            operation_id=operation_id,
        )

    def get(
        self, handoff_id: str, user_context: DemoUserContext
    ) -> HandoffResponse:
        handoff = self.database.fetch_handoff(handoff_id)
        if handoff is None:
            raise OperationNotFoundError(handoff_id)
        self.crm_tool.get_ticket(handoff["ticket_id"], user_context)
        return self.to_response(handoff)

    def list_for_ticket(
        self, ticket_id: str, user_context: DemoUserContext
    ) -> list[HandoffResponse]:
        self.crm_tool.get_ticket(ticket_id, user_context)
        return [
            self.to_response(row)
            for row in self.database.fetch_handoffs(ticket_id)
        ]

    def _transition(
        self,
        handoff_id: str,
        target_state: str,
        user_context: DemoUserContext,
        *,
        operation_id: str | None,
    ) -> HandoffResponse:
        started = time.perf_counter()
        handoff = self.database.fetch_handoff(handoff_id)
        if handoff is None:
            raise OperationNotFoundError(handoff_id)
        self.crm_tool.get_ticket(handoff["ticket_id"], user_context)
        operation_type = f"HANDOFF_{target_state}"
        if operation_id and operation_id.startswith("system-"):
            raise ValueError("Operation IDs beginning with 'system-' are reserved")
        operation_id = operation_id or f"op-handoff-{target_state.lower()}-{uuid4()}"
        request_hash = _hash_request(
            {"handoff_id": handoff_id, "target_state": target_state}
        )
        now = to_utc_iso(utc_now())
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if existing:
                self._validate_operation(
                    dict(existing),
                    user_context,
                    operation_type,
                    request_hash,
                )
                current = connection.execute(
                    "SELECT * FROM case_handoffs WHERE handoff_id = ?",
                    (handoff_id,),
                ).fetchone()
                if current is None:
                    raise OperationNotFoundError(handoff_id)
                result = dict(current)
            else:
                current = connection.execute(
                    "SELECT * FROM case_handoffs WHERE handoff_id = ?",
                    (handoff_id,),
                ).fetchone()
                if current is None:
                    raise OperationNotFoundError(handoff_id)
                current_state = current["state"]
                valid = (
                    target_state == "ACCEPTED" and current_state == "CREATED"
                ) or (
                    target_state == "RESOLVED" and current_state == "ACCEPTED"
                )
                if current_state == target_state:
                    valid = True
                if not valid:
                    self._log(
                        handoff_id,
                        operation_id,
                        current_state,
                        started,
                        outcome="rejected",
                        error_code="INVALID_TRANSITION",
                    )
                    raise InvalidHandoffTransitionError(
                        f"Cannot transition handoff from {current_state} to {target_state}."
                    )
                connection.execute(
                    """
                    INSERT INTO operations (
                        operation_id, operation_type, ticket_id, requested_by,
                        state, request_hash, payload_json, result_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'COMPLETED', ?, ?, ?, ?, ?)
                    """,
                    (
                        operation_id,
                        operation_type,
                        handoff["ticket_id"],
                        user_context.user_id,
                        request_hash,
                        json.dumps(
                            {"handoff_id": handoff_id, "target_state": target_state},
                            separators=(",", ":"),
                        ),
                        json.dumps({"handoff_id": handoff_id}),
                        now,
                        now,
                    ),
                )
                if current_state != target_state:
                    if target_state == "ACCEPTED":
                        connection.execute(
                            """
                            UPDATE case_handoffs
                            SET state = 'ACCEPTED', assigned_to = ?,
                                accepted_at = ?, updated_at = ?
                            WHERE handoff_id = ? AND state = 'CREATED'
                            """,
                            (user_context.user_id, now, now, handoff_id),
                        )
                    else:
                        connection.execute(
                            """
                            UPDATE case_handoffs
                            SET state = 'RESOLVED', resolved_at = ?, updated_at = ?
                            WHERE handoff_id = ? AND state = 'ACCEPTED'
                            """,
                            (now, now, handoff_id),
                        )
                result = dict(
                    connection.execute(
                        "SELECT * FROM case_handoffs WHERE handoff_id = ?",
                        (handoff_id,),
                    ).fetchone()
                )
        self._log(
            handoff_id,
            operation_id,
            result["state"],
            started,
            outcome="existing" if existing else "transitioned",
        )
        return self.to_response(result)

    @staticmethod
    def _validate_operation(
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
    def _permitted_case_context(ticket: dict[str, Any]) -> dict[str, str]:
        return {
            key: str(ticket[key])
            for key in (
                "ticket_id",
                "subject",
                "status",
                "priority",
                "owner",
                "updated_at",
            )
        }

    @staticmethod
    def to_response(handoff: dict[str, Any]) -> HandoffResponse:
        return HandoffResponse(
            handoff_id=handoff["handoff_id"],
            operation_id=handoff.get("operation_id"),
            ticket_id=handoff["ticket_id"],
            trigger_type=handoff["trigger_type"],
            state=handoff["state"],
            queue=handoff["queue"],
            reason=handoff["reason"],
            case_context=json.loads(handoff["context_json"]),
            source_references=json.loads(handoff["source_references_json"]),
            troubleshooting_history=json.loads(handoff["troubleshooting_json"]),
            created_by=handoff["created_by"],
            assigned_to=handoff.get("assigned_to"),
            sla_deadline=handoff.get("sla_deadline"),
            backup_routed_at=handoff.get("backup_routed_at"),
            created_at=handoff["created_at"],
            accepted_at=handoff.get("accepted_at"),
            resolved_at=handoff.get("resolved_at"),
            updated_at=handoff["updated_at"],
        )

    @staticmethod
    def _log(
        handoff_id: str,
        operation_id: str,
        state: str,
        started: float,
        *,
        outcome: str,
        error_code: str | None = None,
    ) -> None:
        log_event(
            logger,
            "handoff.transition",
            handoff_id=handoff_id,
            operation_id=operation_id,
            state=state,
            outcome=outcome,
            error_code=error_code,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
