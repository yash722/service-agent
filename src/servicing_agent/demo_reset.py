from __future__ import annotations

import argparse
import json
from pathlib import Path

from .database import Database
from .settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reset only the configured local synthetic demo database."
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="Optional SQLite path; defaults to SERVICE_AGENT_DB_PATH.",
    )
    args = parser.parse_args()
    settings = Settings.from_environment()
    path = args.database or settings.database_path
    result = Database(
        path,
        timeout_seconds=settings.sqlite_timeout_seconds,
        busy_timeout_ms=settings.sqlite_busy_timeout_ms,
    ).reset_demo_data()
    print(
        json.dumps(
            {
                "status": "reset",
                "database": str(path),
                "synthetic_data_only": True,
                **result,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
