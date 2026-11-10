from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoScenario:
    key: str
    label: str
    demo_user: str
    prompt: str
    expected: str


DEMO_SCENARIOS: tuple[DemoScenario, ...] = (
    DemoScenario(
        key="ticket_status",
        label="1 · Deterministic ticket status",
        demo_user="demo-agent-alex",
        prompt="What is the status of CASE-4821?",
        expected="Fast path returns authoritative CRM status and deterministic SLA state.",
    ),
    DemoScenario(
        key="multi_source",
        label="2 · Multi-source investigation",
        demo_user="demo-agent-alex",
        prompt=(
            "Using the case notes and phone conversation, summarize why CASE-4821 "
            "was escalated and what delay the customer observed."
        ),
        expected="Hybrid retrieval cites the protected case note and call transcript.",
    ),
    DemoScenario(
        key="premium_policy",
        label="3 · Premium policy applicability",
        demo_user="demo-agent-alex",
        prompt="What initial response target applies to CASE-4821?",
        expected="Verified Premium entitlement selects approved policy v2.0.",
    ),
    DemoScenario(
        key="standard_policy",
        label="4 · Standard policy applicability",
        demo_user="demo-agent-alex",
        prompt="What initial response target applies to CASE-4824?",
        expected="Verified Standard entitlement selects approved policy v1.0.",
    ),
    DemoScenario(
        key="unknown_diagnostics",
        label="5 · Deliberately unknown diagnostics",
        demo_user="demo-agent-alex",
        prompt="Were the requested connector diagnostic logs received for CASE-4821?",
        expected=(
            "Protected evidence confirms the request but says receipt is not "
            "recorded; it does not infer that the logs were or were not sent."
        ),
    ),
    DemoScenario(
        key="resolved_incident",
        label="6 · Confirmed resolved incident",
        demo_user="demo-agent-alex",
        prompt="What was confirmed and completed to resolve CASE-4825?",
        expected=(
            "The answer cites the CASE-4825 resolution record and keeps its "
            "completed actions separate from unresolved CASE-4821."
        ),
    ),
    DemoScenario(
        key="unauthorized",
        label="7 · Unauthorized customer access",
        demo_user="demo-agent-jordan",
        prompt="Why was CASE-4821 escalated?",
        expected="No protected evidence, rankings, prompt context, or citations are returned.",
    ),
)


def scenario_by_label(label: str) -> DemoScenario:
    return next(scenario for scenario in DEMO_SCENARIOS if scenario.label == label)
