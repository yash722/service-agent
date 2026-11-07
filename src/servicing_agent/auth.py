from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoUserContext:
    """Server-resolved identity passed to tools as trusted context."""

    user_id: str
    display_name: str
    role: str
    allowed_customer_ids: frozenset[str]

    def can_access_customer(self, customer_id: str) -> bool:
        return customer_id in self.allowed_customer_ids


# Local-only identities. The request header selects a key, while permissions always
# come from this server-side registry and never from the chat message.
DEMO_USERS: dict[str, DemoUserContext] = {
    "demo-agent-alex": DemoUserContext(
        user_id="demo-agent-alex",
        display_name="Alex Morgan",
        role="servicing_agent",
        allowed_customer_ids=frozenset({"CUST-1001", "CUST-1002"}),
    ),
    "demo-agent-jordan": DemoUserContext(
        user_id="demo-agent-jordan",
        display_name="Jordan Lee",
        role="servicing_agent",
        allowed_customer_ids=frozenset({"CUST-1003"}),
    ),
}


def resolve_demo_user(user_id: str) -> DemoUserContext | None:
    return DEMO_USERS.get(user_id)

