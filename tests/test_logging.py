import logging

from servicing_agent.auth import DEMO_USERS
from servicing_agent.database import Database
from servicing_agent.service import ChatService


def test_chat_emits_structured_execution_events(tmp_path, caplog) -> None:
    database = Database(tmp_path / "logging.db")
    database.initialize()
    service = ChatService(database)

    with caplog.at_level(logging.INFO):
        response = service.chat(
            "What is the status of CASE-4821?",
            DEMO_USERS["demo-agent-alex"],
        )

    event_records = [
        record.msg for record in caplog.records if isinstance(record.msg, dict)
    ]
    assert any(
        event.get("event") == "chat.received"
        and event.get("execution_id") == response.execution_id
        for event in event_records
    )
    assert any(
        event.get("event") == "chat.completed"
        and event.get("execution_id") == response.execution_id
        for event in event_records
    )

