import json
from datetime import date
from pathlib import Path

from servicing_agent.auth import DEMO_USERS
from servicing_agent.database import Database
from servicing_agent.service import ChatService


def test_golden_servicing_questions(tmp_path) -> None:
    dataset_path = (
        Path(__file__).parents[1] / "evaluations" / "golden_questions.json"
    )
    examples = json.loads(dataset_path.read_text(encoding="utf-8"))
    database = Database(tmp_path / "golden.db")
    database.initialize()
    service = ChatService(database, as_of=date(2026, 11, 7))

    for example in examples:
        if example.get("scenario_type", "chat") != "chat":
            continue
        if not example.get("regression", False):
            continue
        response = service.chat(
            example["question"], DEMO_USERS[example["demo_user"]]
        )
        assert response.intent == example["expected_intent"], example["id"]
        assert (
            response.evidence_status == example["expected_evidence_status"]
        ), example["id"]
        returned_source_ids = {
            citation.source_id for citation in response.citations
        }
        assert set(
            example["expected_source_ids"]
        ).issubset(returned_source_ids), example["id"]
        if example.get("expect_no_citations"):
            assert response.citations == [], example["id"]
        for expected_text in example["expected_contains"]:
            assert expected_text in response.reply, example["id"]
