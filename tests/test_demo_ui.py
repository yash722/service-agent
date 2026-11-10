from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from servicing_agent.demo_scenarios import DEMO_SCENARIOS


def test_demo_scenarios_cover_required_read_paths() -> None:
    keys = {scenario.key for scenario in DEMO_SCENARIOS}
    assert keys == {
        "ticket_status",
        "multi_source",
        "premium_policy",
        "standard_policy",
        "unknown_diagnostics",
        "resolved_incident",
        "unauthorized",
    }
    unauthorized = next(item for item in DEMO_SCENARIOS if item.key == "unauthorized")
    assert unauthorized.demo_user == "demo-agent-jordan"


def test_streamlit_major_paths_render_without_exceptions() -> None:
    app_path = (
        Path(__file__).parents[1]
        / "src"
        / "servicing_agent"
        / "streamlit_app.py"
    )
    app = AppTest.from_file(app_path).run(timeout=15)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Customer servicing",
        "Controlled actions",
        "Evaluation dashboard",
    ]
    button_labels = {button.label for button in app.button}
    assert {
        "Run selected scenario",
        "Prepare note preview",
        "Create simulated handoff",
        "Run SLA check for CASE-4821",
    }.issubset(button_labels)
    scenario_select = next(
        item for item in app.selectbox if item.label == "Reproducible demo scenario"
    )
    assert list(scenario_select.options) == [
        scenario.label for scenario in DEMO_SCENARIOS
    ]
