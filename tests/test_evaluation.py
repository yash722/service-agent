from pathlib import Path

from servicing_agent.evaluation import evaluate_modes
from servicing_agent.settings import Settings


class EvaluationEmbeddingProvider:
    model_id = "evaluation-test-embeddings"

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("escalat" in lowered or "tier 2" in lowered),
                    float("response" in lowered or "target" in lowered),
                    float("client" in lowered or "customer" in lowered),
                    0.1,
                ]
            )
        return vectors


def test_evaluation_compares_modes_without_leaking_unauthorized_records(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("RETRIEVAL_CACHE_DIR", str(tmp_path / "cache"))
    report = evaluate_modes(
        dataset_path=(
            Path(__file__).parents[1]
            / "evaluations"
            / "golden_questions.json"
        ),
        modes=["lexical", "semantic", "hybrid"],
        runs=1,
        settings=Settings.from_environment(),
        embedding_provider=EvaluationEmbeddingProvider(),
    )

    assert set(report["modes"]) == {"lexical", "semantic", "hybrid"}
    for metrics in report["modes"].values():
        assert 0 <= metrics["recall_at_5"] <= 1
        assert 0 <= metrics["precision_at_3"] <= 1
        unauthorized = [
            example
            for example in metrics["examples"]
            if example["category"] == "unauthorized_access"
        ]
        assert unauthorized
        assert all(not example["selected_document_ids"] for example in unauthorized)
        assert all(not example["returned_source_ids"] for example in unauthorized)
