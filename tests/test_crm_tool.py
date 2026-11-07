import pytest

from servicing_agent.auth import DEMO_USERS
from servicing_agent.database import Database
from servicing_agent.tools import CRMAuthorizationError, MockCRMTool


def test_mock_crm_enforces_trusted_user_scope(tmp_path) -> None:
    database = Database(tmp_path / "crm.db")
    database.initialize()
    crm = MockCRMTool(database)

    ticket = crm.get_ticket("CASE-4821", DEMO_USERS["demo-agent-alex"])
    assert ticket["status"] == "In Progress"

    with pytest.raises(CRMAuthorizationError):
        crm.get_ticket("CASE-4821", DEMO_USERS["demo-agent-jordan"])

