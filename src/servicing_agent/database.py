from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


CUSTOMERS: tuple[tuple[str, str, str, str], ...] = (
    ("CUST-1001", "Maya Patel", "Acme Solar", "maya.patel@example.test"),
    ("CUST-1002", "Luis Ramirez", "Northstar Bikes", "luis.ramirez@example.test"),
    ("CUST-1003", "Priya Chen", "Bluebird Logistics", "priya.chen@example.test"),
)

TICKETS: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    (
        "CASE-4821",
        "CUST-1001",
        "Telemetry data is intermittently delayed",
        "In Progress",
        "High",
        "Tier 2 Support",
        "2026-11-06T19:30:00Z",
    ),
    (
        "CASE-4822",
        "CUST-1002",
        "Unable to update billing contact",
        "Resolved",
        "Medium",
        "Customer Operations",
        "2026-11-05T14:15:00Z",
    ),
    (
        "CASE-4823",
        "CUST-1003",
        "Mobile dashboard does not refresh",
        "Awaiting Customer",
        "Low",
        "Tier 1 Support",
        "2026-11-04T16:45:00Z",
    ),
    (
        "CASE-4824",
        "CUST-1002",
        "Intermittent telemetry export latency",
        "Monitoring",
        "Medium",
        "Tier 1 Support",
        "2026-11-02T18:20:00Z",
    ),
)

ENTITLEMENTS: tuple[tuple[str, str, str, str, str, str, str | None], ...] = (
    (
        "ENT-1001-TELEMETRY",
        "CUST-1001",
        "PRD-TELEMETRY",
        "Premium",
        "Active",
        "2026-01-01",
        None,
    ),
    (
        "ENT-1002-TELEMETRY",
        "CUST-1002",
        "PRD-TELEMETRY",
        "Standard",
        "Active",
        "2025-08-15",
        None,
    ),
    (
        "ENT-1002-BILLING",
        "CUST-1002",
        "PRD-BILLING",
        "Standard",
        "Active",
        "2025-08-15",
        None,
    ),
    (
        "ENT-1003-DASHBOARD",
        "CUST-1003",
        "PRD-DASHBOARD",
        "Standard",
        "Active",
        "2026-04-01",
        None,
    ),
)

SLA_DEADLINES: tuple[tuple[str, str], ...] = (
    ("2026-11-06T21:30:00Z", "CASE-4821"),
    ("2026-11-08T14:15:00Z", "CASE-4822"),
    ("2026-11-10T16:45:00Z", "CASE-4823"),
    ("2026-11-09T18:20:00Z", "CASE-4824"),
)


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    company TEXT NOT NULL,
                    email TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
                    subject TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS entitlements (
                    entitlement_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
                    product_id TEXT NOT NULL,
                    subscription_tier TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT
                );
                """
            )
            self._ensure_column(
                connection,
                "tickets",
                "record_version",
                "INTEGER NOT NULL DEFAULT 1",
            )
            self._ensure_column(connection, "tickets", "sla_deadline", "TEXT")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS operations (
                    operation_id TEXT PRIMARY KEY,
                    operation_type TEXT NOT NULL,
                    ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id),
                    requested_by TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (
                        state IN ('PENDING', 'COMPLETED', 'FAILED', 'UNKNOWN')
                    ),
                    request_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    expected_ticket_version INTEGER,
                    result_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_operations_ticket
                    ON operations(ticket_id, created_at);

                CREATE TABLE IF NOT EXISTS ticket_notes (
                    note_id TEXT PRIMARY KEY,
                    operation_id TEXT NOT NULL UNIQUE
                        REFERENCES operations(operation_id),
                    ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id),
                    body TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_ticket_notes_ticket
                    ON ticket_notes(ticket_id, created_at);

                CREATE TABLE IF NOT EXISTS case_handoffs (
                    handoff_id TEXT PRIMARY KEY,
                    operation_id TEXT UNIQUE REFERENCES operations(operation_id),
                    ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id),
                    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
                    trigger_type TEXT NOT NULL CHECK (
                        trigger_type IN ('USER', 'SLA')
                    ),
                    state TEXT NOT NULL CHECK (
                        state IN ('CREATED', 'ACCEPTED', 'RESOLVED')
                    ),
                    queue TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    source_references_json TEXT NOT NULL,
                    troubleshooting_json TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    assigned_to TEXT,
                    sla_deadline TEXT,
                    backup_routed_at TEXT,
                    created_at TEXT NOT NULL,
                    accepted_at TEXT,
                    resolved_at TEXT,
                    updated_at TEXT NOT NULL,
                    UNIQUE(ticket_id, trigger_type, sla_deadline)
                );

                CREATE INDEX IF NOT EXISTS idx_handoffs_ticket
                    ON case_handoffs(ticket_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_handoffs_queue_state
                    ON case_handoffs(queue, state, created_at);
                """
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO customers
                    (customer_id, name, company, email)
                VALUES (?, ?, ?, ?)
                """,
                CUSTOMERS,
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO tickets
                    (ticket_id, customer_id, subject, status, priority, owner, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                TICKETS,
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO entitlements
                    (entitlement_id, customer_id, product_id, subscription_tier,
                     status, start_date, end_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ENTITLEMENTS,
            )
            connection.executemany(
                """
                UPDATE tickets
                SET sla_deadline = COALESCE(sla_deadline, ?)
                WHERE ticket_id = ?
                """,
                SLA_DEADLINES,
            )

    def reset_demo_data(self) -> dict[str, int]:
        """Atomically restore the canonical synthetic dataset."""

        self.initialize()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            removed = {
                "ticket_notes": connection.execute(
                    "SELECT COUNT(*) FROM ticket_notes"
                ).fetchone()[0],
                "case_handoffs": connection.execute(
                    "SELECT COUNT(*) FROM case_handoffs"
                ).fetchone()[0],
                "operations": connection.execute(
                    "SELECT COUNT(*) FROM operations"
                ).fetchone()[0],
            }
            connection.execute("DELETE FROM ticket_notes")
            connection.execute("DELETE FROM case_handoffs")
            connection.execute("DELETE FROM operations")
            connection.execute("DELETE FROM entitlements")
            connection.execute("DELETE FROM tickets")
            connection.execute("DELETE FROM customers")
            connection.executemany(
                """
                INSERT INTO customers (customer_id, name, company, email)
                VALUES (?, ?, ?, ?)
                """,
                CUSTOMERS,
            )
            connection.executemany(
                """
                INSERT INTO tickets (
                    ticket_id, customer_id, subject, status, priority, owner,
                    updated_at, record_version, sla_deadline
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, NULL)
                """,
                TICKETS,
            )
            connection.executemany(
                """
                INSERT INTO entitlements (
                    entitlement_id, customer_id, product_id, subscription_tier,
                    status, start_date, end_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ENTITLEMENTS,
            )
            connection.executemany(
                "UPDATE tickets SET sla_deadline = ? WHERE ticket_id = ?",
                SLA_DEADLINES,
            )
        return {
            **removed,
            "customers": len(CUSTOMERS),
            "tickets": len(TICKETS),
            "entitlements": len(ENTITLEMENTS),
        }

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        existing = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def fetch_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    t.ticket_id,
                    t.customer_id,
                    t.subject,
                    t.status,
                    t.priority,
                    t.owner,
                    t.updated_at,
                    t.record_version,
                    t.sla_deadline,
                    c.name AS customer_name,
                    c.company AS customer_company
                FROM tickets AS t
                JOIN customers AS c ON c.customer_id = t.customer_id
                WHERE t.ticket_id = ?
                """,
                (ticket_id.upper(),),
            ).fetchone()
        return dict(row) if row else None

    def fetch_all_tickets(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT ticket_id, customer_id, subject, status, priority, owner,
                       updated_at, record_version, sla_deadline
                FROM tickets
                ORDER BY ticket_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_operation(self, operation_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
        return dict(row) if row else None

    def fetch_note_by_operation(self, operation_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM ticket_notes WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
        return dict(row) if row else None

    def fetch_ticket_notes(self, ticket_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT note_id, operation_id, ticket_id, body, created_by, created_at
                FROM ticket_notes
                WHERE ticket_id = ?
                ORDER BY created_at, note_id
                """,
                (ticket_id.upper(),),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_handoff(self, handoff_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM case_handoffs WHERE handoff_id = ?",
                (handoff_id,),
            ).fetchone()
        return dict(row) if row else None

    def fetch_handoff_by_operation(
        self, operation_id: str
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM case_handoffs WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
        return dict(row) if row else None

    def fetch_handoffs(self, ticket_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM case_handoffs
                WHERE ticket_id = ?
                ORDER BY created_at, handoff_id
                """,
                (ticket_id.upper(),),
            ).fetchall()
        return [dict(row) for row in rows]

    def simulate_external_ticket_update(
        self,
        ticket_id: str,
        *,
        status: str | None = None,
        owner: str | None = None,
        updated_at: str,
    ) -> None:
        """Local-test helper that simulates a CRM change outside the agent."""

        assignments = ["updated_at = ?", "record_version = record_version + 1"]
        values: list[Any] = [updated_at]
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
        if owner is not None:
            assignments.append("owner = ?")
            values.append(owner)
        values.append(ticket_id.upper())
        with self.connect() as connection:
            connection.execute(
                f"UPDATE tickets SET {', '.join(assignments)} WHERE ticket_id = ?",
                values,
            )

    def fetch_active_entitlements(
        self, customer_id: str, as_of: str
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    entitlement_id,
                    customer_id,
                    product_id,
                    subscription_tier,
                    status,
                    start_date,
                    end_date
                FROM entitlements
                WHERE customer_id = ?
                  AND status = 'Active'
                  AND start_date <= ?
                  AND (end_date IS NULL OR end_date >= ?)
                ORDER BY product_id, entitlement_id
                """,
                (customer_id, as_of, as_of),
            ).fetchall()
        return [dict(row) for row in rows]

