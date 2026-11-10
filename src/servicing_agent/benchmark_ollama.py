from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from .api import create_app
from .settings import ConfigurationError, Settings


QUESTIONS = (
    (
        "still_investigating",
        "Why is CASE-4821 still under investigation?",
    ),
    (
        "escalation_reason",
        "Why was CASE-4821 escalated?",
    ),
)


def _answer_relevant(question_id: str, reply: str) -> bool:
    lowered = reply.lower()
    if question_id == "still_investigating":
        return "investigat" in lowered and any(
            marker in lowered
            for marker in (
                "does not confirm",
                "not confirm",
                "not establish",
                "no confirmed",
                "unknown",
                "unclear",
            )
        )
    return "escalat" in lowered and "case-4821" in lowered


def _citation_validity(body: dict[str, Any]) -> bool:
    citations = body.get("citations", [])
    reply = str(body.get("reply", ""))
    return bool(citations) and all(
        citation["citation_label"] in reply for citation in citations
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    generation = [float(item["generation_ms"]) for item in runs]
    total = [float(item["total_ms"]) for item in runs]
    output_tokens = [
        int(item["output_tokens"])
        for item in runs
        if item["output_tokens"] is not None
    ]
    return {
        "runs": len(runs),
        "generation_ms_mean": round(statistics.mean(generation), 3),
        "generation_ms_p50": round(statistics.median(generation), 3),
        "generation_ms_p95": _percentile(generation, 0.95),
        "total_ms_mean": round(statistics.mean(total), 3),
        "total_ms_p50": round(statistics.median(total), 3),
        "total_ms_p95": _percentile(total, 0.95),
        "output_tokens_mean": (
            round(statistics.mean(output_tokens), 3) if output_tokens else None
        ),
        "finish_reasons": dict(Counter(item["finish_reason"] for item in runs)),
        "citation_valid_rate": round(
            sum(bool(item["citation_valid"]) for item in runs) / len(runs), 3
        ),
        "answer_relevance_rate": round(
            sum(bool(item["answer_relevant"]) for item in runs) / len(runs), 3
        ),
        "fallback_rate": round(
            sum(bool(item["fallback"]) for item in runs) / len(runs), 3
        ),
    }


async def _run_benchmark(settings: Settings, repetitions: int) -> dict[str, Any]:
    if settings.history_backend != "snowflake":
        raise ConfigurationError(
            "The live benchmark requires HISTORY_BACKEND=snowflake."
        )
    if settings.retrieval_mode != "hybrid":
        raise ConfigurationError("The live benchmark requires RETRIEVAL_MODE=hybrid.")
    if settings.llm_provider != "openai_compatible" or not settings.llm_api_key:
        raise ConfigurationError(
            "The live benchmark requires the configured OpenAI-compatible LLM."
        )

    with tempfile.TemporaryDirectory(prefix="servicing-agent-benchmark-") as temp:
        database_path = Path(temp) / "benchmark.db"
        app = create_app(
            database_path,
            configure_logs=False,
            settings_override=settings,
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            timeout = httpx.Timeout(settings.ui_api_read_timeout_seconds)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://benchmark",
                timeout=timeout,
            ) as client:
                headers = {"X-Demo-User": "demo-agent-alex"}

                # Warm the embedding model, Snowflake connection path, and Ollama
                # before collecting the requested repetitions.
                warmup = await client.post(
                    "/chat",
                    headers=headers,
                    json={"message": QUESTIONS[0][1]},
                )
                warmup.raise_for_status()

                raw_runs: list[dict[str, Any]] = []
                for question_id, question in QUESTIONS:
                    for repetition in range(1, repetitions + 1):
                        response = await client.post(
                            "/chat",
                            headers=headers,
                            json={"message": question},
                        )
                        response.raise_for_status()
                        body = response.json()
                        generation = body["generation_debug"]
                        raw_runs.append(
                            {
                                "question_id": question_id,
                                "repetition": repetition,
                                "generation_mode": body["generation_mode"],
                                "generation_ms": generation["latency_ms"],
                                "total_ms": body["total_latency_ms"],
                                "finish_reason": generation["finish_reason"],
                                "output_tokens": generation["output_tokens"],
                                "citation_valid": _citation_validity(body),
                                "answer_relevant": _answer_relevant(
                                    question_id, body["reply"]
                                ),
                                "fallback": body["generation_mode"] != "llm",
                                "fallback_reason": generation["fallback_reason"],
                                "citation_source_ids": [
                                    citation["source_id"]
                                    for citation in body["citations"]
                                ],
                                "snowflake_record_count": body["history_debug"][
                                    "record_count"
                                ],
                            }
                        )

    by_question = {
        question_id: _summarize(
            [item for item in raw_runs if item["question_id"] == question_id]
        )
        for question_id, _question in QUESTIONS
    }
    return {
        "model": settings.llm_model,
        "max_tokens": settings.llm_max_tokens,
        "timeout_seconds": settings.llm_timeout_seconds,
        "history_backend": settings.history_backend,
        "retrieval_mode": settings.retrieval_mode,
        "warmup_excluded": True,
        "questions": [
            {"id": question_id, "question": question}
            for question_id, question in QUESTIONS
        ],
        "summary": by_question,
        "runs": raw_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark configured Ollama generation through real Snowflake and "
            "hybrid retrieval."
        )
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--model",
        help=(
            "Optional comparison-model override. The configured model remains "
            "the default."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path for non-sensitive benchmark metrics.",
    )
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    settings = Settings.from_environment()
    if args.model:
        settings = replace(settings, llm_model=args.model)
    result = asyncio.run(_run_benchmark(settings, args.runs))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
