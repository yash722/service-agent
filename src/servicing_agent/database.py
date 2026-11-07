from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


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
)


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
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
                    c.name AS customer_name,
                    c.company AS customer_company
                FROM tickets AS t
                JOIN customers AS c ON c.customer_id = t.customer_id
                WHERE t.ticket_id = ?
                """,
                (ticket_id.upper(),),
            ).fetchone()
        return dict(row) if row else None

