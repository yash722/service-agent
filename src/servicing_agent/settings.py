from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_path: Path
    log_level: str

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            database_path=Path(
                os.getenv("SERVICE_AGENT_DB_PATH", "data/service_agent.db")
            ),
            log_level=os.getenv("SERVICE_AGENT_LOG_LEVEL", "INFO").upper(),
        )

