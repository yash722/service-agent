from __future__ import annotations

import re
from typing import Literal

Intent = Literal["ticket_status", "complex_request"]

_CASE_ID = re.compile(r"\bCASE-\d+\b", re.IGNORECASE)
_STATUS_TERMS = re.compile(
    r"\b(status|state|progress|update|where\s+(?:is|was)|what(?:'s|\s+is)\s+happening)\b",
    re.IGNORECASE,
)
_TICKET_TERMS = re.compile(r"\b(case|ticket)\b", re.IGNORECASE)


def extract_ticket_id(message: str) -> str | None:
    match = _CASE_ID.search(message)
    return match.group(0).upper() if match else None


def route_intent(message: str) -> Intent:
    """Route explicit ticket-status questions without invoking an LLM."""

    has_ticket_reference = bool(_CASE_ID.search(message) or _TICKET_TERMS.search(message))
    if has_ticket_reference and _STATUS_TERMS.search(message):
        return "ticket_status"
    return "complex_request"

