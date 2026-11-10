from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any


_trace_enabled = True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }
        if isinstance(record.msg, dict):
            payload.update(record.msg)
        else:
            payload["message"] = record.getMessage()
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    *,
    json_logs: bool = True,
    trace_enabled: bool = True,
) -> None:
    global _trace_enabled
    _trace_enabled = trace_enabled
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter()
        if json_logs
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Connector INFO diagnostics can include account or credential identifiers.
    # Application traces capture only the sanitized history outcome instead.
    logging.getLogger("snowflake.connector").setLevel(logging.WARNING)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    if _trace_enabled:
        logger.info({"event": event, **fields})

