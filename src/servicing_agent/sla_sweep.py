from __future__ import annotations

import argparse
import json
from datetime import timedelta

from .database import Database
from .logging_config import configure_logging
from .settings import Settings
from .sla import SLAService
from .time_utils import parse_utc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one deterministic local SLA escalation sweep."
    )
    parser.add_argument(
        "--as-of",
        help="UTC ISO timestamp used for the sweep, for example 2026-11-08T12:00:00Z",
    )
    args = parser.parse_args()
    settings = Settings.from_environment()
    configure_logging(
        settings.log_level,
        json_logs=settings.log_json,
        trace_enabled=settings.trace_enabled,
    )
    database = Database(
        settings.database_path,
        timeout_seconds=settings.sqlite_timeout_seconds,
        busy_timeout_ms=settings.sqlite_busy_timeout_ms,
    )
    database.initialize()
    result = SLAService(
        database,
        backup_after=timedelta(minutes=settings.sla_backup_after_minutes),
    ).sweep(
        as_of=parse_utc(args.as_of) if args.as_of else None
    )
    print(json.dumps(result.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
