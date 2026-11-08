from __future__ import annotations

from pathlib import Path

from servicing_agent.evaluation import evaluate_modes
from servicing_agent.settings import Settings


class ScenarioEmbeddingProvider:
    model_id = "scenario-test-embeddings"

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("escalat" in lowered or "tier 2" in lowered),
                    float("response" in lowered or "target" in lowered),
                    float("phone" in lowered or "customer" in lowered),
                    float("telemetry" in lowered or "sync" in lowered),
                    0.1,
                ]
            )
        return vectors


def dataset_path() -> Path:
    return Path(__file__).parents[1] / "evaluations" / "golden_questions.json"


def test_end_to_end_evaluation_covers_actions_sla_and_recovery(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("RETRIEVAL_CACHE_DIR", str(tmp_path / "cache"))

    report = evaluate_modes(
        dataset_path=dataset_path(),
        modes=["lexical", "semantic", "hybrid"],
        runs=1,
        settings=Settings.from_environment(),
        embedding_provider=ScenarioEmbeddingProvider(),
    )

    assert report["schema_version"] == 2
    assert report["scenario_count"] == 18
    assert report["generation_evaluation"]["declared_effective"] == "deterministic"
    assert report["business_outcomes"]["measured"] == []
    assert "CSAT" in report["business_outcomes"]["not_measured"]
    for metrics in report["modes"].values():
        assert metrics["scenario_run_count"] == 18
        assert metrics["action_safety_rate"] == 1.0
        assert metrics["escalation_correctness"] == 1.0
        assert metrics["recovery_success_rate"] == 1.0
        assert metrics["authorization_safety_rate"] == 1.0
        assert metrics["tool_observation_count"] > 0
        assert metrics["end_to_end_observation_count"] == 18
        scenario_types = {item["scenario_type"] for item in metrics["examples"]}
        assert scenario_types == {"chat", "controlled_note", "handoff", "sla"}


def test_configured_llm_evaluation_without_key_is_labeled_fallback(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("RETRIEVAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SERVICE_AGENT_LLM_PROVIDER", "openai_compatible")
    monkeypatch.delenv("SERVICE_AGENT_LLM_API_KEY", raising=False)

    report = evaluate_modes(
        dataset_path=dataset_path(),
        modes=["lexical"],
        runs=1,
        settings=Settings.from_environment(),
        use_configured_llm=True,
    )

    generation = report["generation_evaluation"]
    assert generation["requested"] == "configured_llm"
    assert generation["declared_effective"] == "deterministic_fallback_no_credentials"
    assert generation["credentials_present"] is False
    assert report["modes"]["lexical"]["generation_modes"].get("llm", 0) == 0
