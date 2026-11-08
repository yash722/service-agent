from servicing_agent.generation import (
    DeterministicGroundedGenerator,
    OpenAICompatibleGroundedGenerator,
)
from servicing_agent.service import build_answer_generator
from servicing_agent.settings import Settings


def test_configured_provider_without_key_uses_deterministic_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SERVICE_AGENT_LLM_PROVIDER", "openai_compatible")
    monkeypatch.delenv("SERVICE_AGENT_LLM_API_KEY", raising=False)

    generator = build_answer_generator(Settings.from_environment())

    assert isinstance(generator, DeterministicGroundedGenerator)


def test_provider_is_enabled_only_when_key_is_in_environment(monkeypatch) -> None:
    monkeypatch.setenv("SERVICE_AGENT_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("SERVICE_AGENT_LLM_API_KEY", "synthetic-test-secret")

    generator = build_answer_generator(Settings.from_environment())

    assert isinstance(generator, OpenAICompatibleGroundedGenerator)
