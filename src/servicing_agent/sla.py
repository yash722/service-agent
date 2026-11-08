from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Callable

from .auth import DemoUserContext
from .database import Database
from .handoffs import CaseHandoffService
from .logging_config import log_event
from .schemas import SLAStatusResponse, SLASweepResponse
from .time_utils import parse_utc, to_utc_iso, utc_now
from .tools import MockCRMTool, TicketNotFoundError

logger = logging.getLogger(__name__)

_CLOSED_STATUSES = {"resolved", "closed"}


class SLAService:
    """Deterministic SLA checks; no LLM participates in escalation decisions."""

    def __init__(
        self,
        database: Database,
        crm_tool: MockCRMTool | None = None,
        *,
        clock: Callable[[], datetime] = utc_now,
        backup_after: timedelta = timedelta(minutes=15),
    ):
        self.database = database
        self.crm_tool = crm_tool or MockCRMTool(database)
        self.clock = clock
        self.backup_after = backup_after

    def check_ticket(
        self,
        ticket_id: str,
        user_context: DemoUserContext | None = None,
        *,
        as_of: datetime | None = None,
    ) -> SLAStatusResponse:
        started = time.perf_counter()
        ticket = (
            self.crm_tool.get_ticket(ticket_id, user_context)
            if user_context is not None
            else self.database.fetch_ticket(ticket_id)
        )
        if ticket is None:
            raise TicketNotFoundError(ticket_id)
        now = as_of or self.clock()
        deadline_value = ticket.get("sla_deadline")
        if not deadline_value:
            result = SLAStatusResponse(
                ticket_id=ticket["ticket_id"],
                deadline=None,
                overdue=False,
                action="no_deadline",
            )
        elif ticket["status"].lower() in _CLOSED_STATUSES:
            result = SLAStatusResponse(
                ticket_id=ticket["ticket_id"],
                deadline=deadline_value,
                overdue=now >= parse_utc(deadline_value),
                action="closed",
            )
        elif now < parse_utc(deadline_value):
            result = SLAStatusResponse(
                ticket_id=ticket["ticket_id"],
                deadline=deadline_value,
                overdue=False,
                action="not_due",
            )
        else:
            handoff, created = self._ensure_sla_handoff(ticket, now)
            result = SLAStatusResponse(
                ticket_id=ticket["ticket_id"],
                deadline=deadline_value,
                overdue=True,
                action="created" if created else "already_escalated",
                handoff_id=handoff["handoff_id"],
                handoff_state=handoff["state"],
                queue=handoff["queue"],
            )
        log_event(
            logger,
            "sla.checked",
            ticket_id=ticket["ticket_id"],
            overdue=result.overdue,
            action=result.action,
            handoff_id=result.handoff_id,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return result

    def sweep(self, *, as_of: datetime | None = None) -> SLASweepResponse:
        now = as_of or self.clock()
        checked = 0
        created = 0
        already = 0
        for ticket in self.database.fetch_all_tickets():
            checked += 1
            result = self.check_ticket(ticket["ticket_id"], as_of=now)
            if result.action == "created":
                created += 1
            elif result.action == "already_escalated":
                already += 1
        backup_routed = self.route_unaccepted_to_backup(as_of=now)
        response = SLASweepResponse(
            checked=checked,
            created=created,
            already_escalated=already,
            backup_routed=backup_routed,
            as_of=to_utc_iso(now),
        )
        log_event(logger, "sla.sweep_completed", **response.model_dump())
        return response

    def route_unaccepted_to_backup(
        self, *, as_of: datetime | None = None
    ) -> int:
        now = as_of or self.clock()
        cutoff = to_utc_iso(now - self.backup_after)
        now_value = to_utc_iso(now)
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT handoff_id
                FROM case_handoffs
                WHERE state = 'CREATED'
                  AND backup_routed_at IS NULL
                  AND created_at <= ?
                ORDER BY handoff_id
                """,
                (cutoff,),
            ).fetchall()
            handoff_ids = [row["handoff_id"] for row in rows]
            if handoff_ids:
                placeholders = ",".join("?" for _ in handoff_ids)
                connection.execute(
                    f"""
                    UPDATE case_handoffs
                    SET queue = 'Backup Escalation Queue (simulated)',
                        backup_routed_at = ?, updated_at = ?
                    WHERE handoff_id IN ({placeholders})
                      AND state = 'CREATED'
                      AND backup_routed_at IS NULL
                    """,
                    (now_value, now_value, *handoff_ids),
                )
        for handoff_id in handoff_ids:
            log_event(
                logger,
                "sla.backup_routed",
                handoff_id=handoff_id,
                queue="Backup Escalation Queue (simulated)",
            )
        return len(handoff_ids)

    def _ensure_sla_handoff(
        self, ticket: dict, now: datetime
    ) -> tuple[dict, bool]:
        deadline = ticket["sla_deadline"]
        digest = hashlib.sha256(
            f"{ticket['ticket_id']}|{deadline}".encode("utf-8")
        ).hexdigest()[:20]
        operation_id = f"system-sla-{digest}"
        handoff_id = f"HANDOFF-SLA-{digest}"
        now_value = to_utc_iso(now)
        context = CaseHandoffService._permitted_case_context(ticket)
        reason = (
            f"Deterministic SLA deadline {deadline} passed while the case remained "
            f"{ticket['status']}."
        )
        history = [
            "Automated local SLA rule evaluated the authoritative ticket deadline."
        ]
        payload = {
            "ticket_id": ticket["ticket_id"],
            "sla_deadline": deadline,
            "rule": "open_case_deadline_passed",
        }
        request_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM case_handoffs
                WHERE ticket_id = ? AND trigger_type = 'SLA' AND sla_deadline = ?
                """,
                (ticket["ticket_id"], deadline),
            ).fetchone()
            if existing:
                return dict(existing), False
            connection.execute(
                """
                INSERT INTO operations (
                    operation_id, operation_type, ticket_id, requested_by,
                    state, request_hash, payload_json, expected_ticket_version,
                    result_json, created_at, updated_at
                ) VALUES (?, 'SLA_ESCALATION', ?, 'system:sla', 'COMPLETED',
                          ?, ?, ?, ?, ?, ?)
                """,
                (
                    operation_id,
                    ticket["ticket_id"],
                    request_hash,
                    json.dumps(payload, separators=(",", ":")),
                    ticket["record_version"],
                    json.dumps({"handoff_id": handoff_id}),
                    now_value,
                    now_value,
                ),
            )
            connection.execute(
                """
                INSERT INTO case_handoffs (
                    handoff_id, operation_id, ticket_id, customer_id,
                    trigger_type, state, queue, reason, context_json,
                    source_references_json, troubleshooting_json, created_by,
                    sla_deadline, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'SLA', 'CREATED', ?, ?, ?, '[]', ?,
                          'system:sla', ?, ?, ?)
                """,
                (
                    handoff_id,
                    operation_id,
                    ticket["ticket_id"],
                    ticket["customer_id"],
                    "High Priority SLA Queue (simulated)",
                    reason,
                    json.dumps(context, separators=(",", ":")),
                    json.dumps(history, separators=(",", ":")),
                    deadline,
                    now_value,
                    now_value,
                ),
            )
            created = connection.execute(
                "SELECT * FROM case_handoffs WHERE handoff_id = ?",
                (handoff_id,),
            ).fetchone()
        return dict(created), True
