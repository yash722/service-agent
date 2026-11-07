from servicing_agent.intent_router import extract_ticket_id, route_intent


def test_routes_explicit_case_status_request_deterministically() -> None:
    message = "Can I get a status update for case-4821?"

    assert route_intent(message) == "ticket_status"
    assert extract_ticket_id(message) == "CASE-4821"


def test_routes_non_status_request_to_workflow() -> None:
    assert route_intent("Why was CASE-4821 escalated?") == "complex_request"


def test_status_request_without_id_can_ask_for_case_id() -> None:
    assert route_intent("What is the status of my ticket?") == "ticket_status"
    assert extract_ticket_id("What is the status of my ticket?") is None

