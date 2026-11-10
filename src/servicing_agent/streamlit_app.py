from __future__ import annotations

import httpx
import streamlit as st

from servicing_agent.demo_scenarios import DEMO_SCENARIOS, scenario_by_label
from servicing_agent.evaluation_dashboard import render_evaluation_dashboard
from servicing_agent.settings import Settings


SETTINGS = Settings.from_environment()
API_URL = SETTINGS.api_url
EVALUATION_RESULTS_PATH = SETTINGS.evaluation_results_path
API_TIMEOUT = httpx.Timeout(
    connect=SETTINGS.api_request_timeout_seconds,
    read=SETTINGS.ui_api_read_timeout_seconds,
    write=SETTINGS.api_request_timeout_seconds,
    pool=SETTINGS.api_request_timeout_seconds,
)
DEMO_IDENTITIES = {
    "Alex Morgan · CUST-1001, CUST-1002": "demo-agent-alex",
    "Jordan Lee · CUST-1003 only": "demo-agent-jordan",
}

st.set_page_config(
    page_title=SETTINGS.ui_page_title,
    page_icon=SETTINGS.ui_page_icon,
    layout=SETTINGS.ui_layout,
)
st.title("Salesforce FDE Servicing Agent PoC")
st.caption(
    "Milestone 4 · evaluation-driven local demo · synthetic customers · "
    "simulated CRM integration"
)

identity_label = st.sidebar.selectbox("Demo identity", list(DEMO_IDENTITIES))
demo_user = DEMO_IDENTITIES[identity_label]
st.sidebar.info(
    "The API resolves permissions from this server-side demo identity. Chat text "
    "cannot elevate access."
)
st.sidebar.warning(
    "No Salesforce connection: CRM writes, queues, SLA actions, and customer "
    "records are synthetic local simulations."
)
with st.sidebar.expander("Reset before a live demo"):
    st.code(
        "python -m servicing_agent.demo_reset",
        language="powershell",
    )

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Choose a reproducible scenario above the chat or ask about an "
                "authorized synthetic service case."
            ),
        }
    ]


def call_api(
    method: str,
    path: str,
    payload: dict | None = None,
    *,
    user_id: str | None = None,
) -> dict | None:
    try:
        response = httpx.request(
            method,
            f"{API_URL}{path}",
            headers={"X-Demo-User": user_id or demo_user},
            json=payload,
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        response_payload = response.json()
        if not isinstance(response_payload, dict):
            raise ValueError("The API response was not a JSON object.")
        return response_payload
    except httpx.HTTPStatusError:
        try:
            error_payload = response.json().get("detail")
        except ValueError:
            error_payload = response.text
        st.error(f"The simulated operation was rejected: {error_payload}")
        if isinstance(error_payload, dict) and error_payload.get("operation_id"):
            return error_payload
        return None
    except httpx.ReadTimeout:
        st.error(
            "The API is still processing or exceeded the configured Streamlit "
            f"read timeout of {SETTINGS.ui_api_read_timeout_seconds:g} seconds. "
            "The backend may still finish; inspect its stage-latency logs before "
            "retrying."
        )
        return None
    except (httpx.ConnectError, httpx.ConnectTimeout):
        st.error(
            f"Streamlit could not connect to the local API at {API_URL}. Start "
            "FastAPI and confirm SERVICE_AGENT_API_URL."
        )
        return None
    except httpx.TimeoutException as exc:
        st.error(
            "The local API request timed out outside response reading "
            f"({type(exc).__name__}). Check the API process and retry."
        )
        return None
    except httpx.RequestError as exc:
        st.error(
            "The local API connection failed while sending or receiving data "
            f"({type(exc).__name__})."
        )
        return None
    except (ValueError, TypeError, KeyError) as exc:
        st.error(
            "The local API returned an invalid JSON response. Inspect FastAPI "
            f"logs and response schema. ({type(exc).__name__})"
        )
        return None


def assistant_message(result: dict) -> dict:
    return {
        "role": "assistant",
        "content": result["reply"],
        "citations": result.get("citations", []),
        "warnings": result.get("warnings", []),
        "details": {
            "execution_id": result["execution_id"],
            "intent": result["intent"],
            "generation_mode": result.get("generation_mode", "not_applicable"),
            "evidence_status": result.get("evidence_status", "not_applicable"),
            "trace": result.get("trace", result["steps"]),
            "retrieval": result.get("retrieval_debug"),
            "history": result.get("history_debug"),
            "generation": result.get("generation_debug"),
            "total_latency_ms": result.get("total_latency_ms"),
            "sla": result.get("sla"),
        },
    }


def submit_chat(prompt: str, user_id: str) -> bool:
    st.session_state.messages.append({"role": "user", "content": prompt})
    result = call_api(
        "POST", "/chat", {"message": prompt}, user_id=user_id
    )
    if result is None:
        return False
    st.session_state.messages.append(assistant_message(result))
    return True


def render_message(message: dict) -> None:
    st.markdown(message["content"])
    citations = message.get("citations", [])
    details = message.get("details")
    if details:
        mode_labels = {
            "llm": "Configured LLM",
            "deterministic": "Deterministic evidence fallback",
            "not_applicable": "Deterministic fast path",
        }
        st.caption(
            f"Generation mode: {mode_labels.get(details['generation_mode'], details['generation_mode'])} "
            f"· Evidence status: {details['evidence_status']}"
        )
        if citations:
            st.caption(
                "Evidence sources: "
                + ", ".join(citation["source_id"] for citation in citations)
            )
        timing_parts = []
        if details.get("history"):
            timing_parts.append(
                f"Snowflake/history {details['history']['latency_ms']:.0f} ms"
            )
        if details.get("retrieval"):
            timing_parts.append(
                f"retrieval {details['retrieval']['latency_ms']:.0f} ms"
            )
        if details.get("generation"):
            timing_parts.append(
                f"generation {details['generation']['latency_ms']:.0f} ms"
            )
        if details.get("total_latency_ms") is not None:
            timing_parts.append(
                f"total {details['total_latency_ms']:.0f} ms"
            )
        if timing_parts:
            st.caption("Stage latency: " + " · ".join(timing_parts))
        generation_details = details.get("generation") or {}
        if generation_details.get("fallback_reason"):
            st.info(
                "A grounded deterministic fallback was used. The technical "
                "validation reason remains available in the diagnostics below."
            )
    if citations:
        with st.expander(f"Evidence citations · {len(citations)}", expanded=True):
            for citation in citations:
                st.markdown(
                    f"**{citation['citation_label']} — {citation['title']}**"
                )
                st.caption(
                    f"{citation['document_type']} · effective "
                    f"{citation['effective_date']}"
                )
                st.write(citation["snippet"])
    warnings = message.get("warnings", [])
    fallback_reason = (
        ((details or {}).get("generation") or {}).get("fallback_reason")
    )
    visible_warnings = [
        warning for warning in warnings if warning != fallback_reason
    ]
    if visible_warnings:
        st.warning(" ".join(visible_warnings))
    if details:
        with st.expander("Execution trace and retrieval diagnostics"):
            st.json(details)


def render_controlled_note() -> None:
    st.markdown("#### Ticket-note confirmation")
    st.caption(
        "Preview and confirmation are separate durable operations. No note is "
        "saved until customer confirmation is attested."
    )
    note_case_id = st.text_input(
        "Case ID", value="CASE-4821", key="note_case_id"
    )
    note_body = st.text_area(
        "Ticket note",
        value="Customer approved the troubleshooting summary for the case.",
        key="note_body",
    )
    if st.button("Prepare note preview", type="primary"):
        prepared = call_api(
            "POST",
            "/actions/notes/prepare",
            {"ticket_id": note_case_id, "note": note_body},
        )
        if prepared:
            st.session_state.pending_note_operation = prepared
            st.rerun()

    pending = st.session_state.get("pending_note_operation")
    if pending:
        st.status(
            f"Operation {pending['operation_id']} · {pending['state']}",
            state=(
                "complete"
                if pending["state"] == "COMPLETED"
                else "error"
                if pending["state"] == "FAILED"
                else "running"
            ),
        )
        st.json(pending.get("preview"))
        if pending["state"] == "PENDING":
            customer_confirmed = st.checkbox(
                "The customer explicitly confirmed this note may be saved",
                key="customer_confirmed_note",
            )
            simulate_timeout = st.checkbox(
                "Simulate timeout after successful save",
                key="simulate_note_timeout",
            )
            confirm_column, cancel_column = st.columns(2)
            if confirm_column.button(
                "Confirm simulated save", disabled=not customer_confirmed
            ):
                outcome = call_api(
                    "POST",
                    f"/actions/notes/{pending['operation_id']}/confirm",
                    {
                        "customer_confirmed": customer_confirmed,
                        "simulate_timeout_after_success": simulate_timeout,
                    },
                )
                if outcome:
                    st.session_state.pending_note_operation = outcome
                    st.session_state.last_action_outcome = outcome
                    st.rerun()
            if cancel_column.button("Cancel pending note"):
                outcome = call_api(
                    "POST",
                    f"/actions/notes/{pending['operation_id']}/cancel",
                    {},
                )
                if outcome:
                    st.session_state.pending_note_operation = outcome
                    st.session_state.last_action_outcome = outcome
                    st.rerun()
        if pending["state"] == "UNKNOWN" and st.button(
            "Reconcile unknown outcome"
        ):
            outcome = call_api(
                "POST",
                f"/actions/operations/{pending['operation_id']}/reconcile",
                {},
            )
            if outcome:
                st.session_state.pending_note_operation = outcome
                st.session_state.last_action_outcome = outcome
                st.rerun()
        outcome = st.session_state.get("last_action_outcome")
        if outcome:
            if outcome["state"] == "COMPLETED":
                st.success(outcome["message"])
            elif outcome["state"] == "UNKNOWN":
                st.warning(outcome["message"])
            elif outcome["state"] == "FAILED":
                st.error(outcome["message"])


def render_handoff_and_sla() -> None:
    st.markdown("#### Human escalation")
    handoff_case_id = st.text_input(
        "Handoff case ID", value="CASE-4821", key="handoff_case_id"
    )
    handoff_reason = st.text_area(
        "Handoff reason",
        value="Repeated telemetry delays require human specialist review.",
        key="handoff_reason",
    )
    troubleshooting = st.text_area(
        "Troubleshooting history · one item per line",
        value="Restarted the connector\nObserved repeated delayed sync windows",
        key="handoff_history",
    )
    if st.button("Create simulated handoff"):
        handoff = call_api(
            "POST",
            "/handoffs",
            {
                "ticket_id": handoff_case_id,
                "reason": handoff_reason,
                "troubleshooting_history": [
                    line.strip()
                    for line in troubleshooting.splitlines()
                    if line.strip()
                ],
            },
        )
        if handoff:
            st.session_state.active_handoff = handoff
            st.rerun()

    active = st.session_state.get("active_handoff")
    if active:
        st.json(active)
        if active["state"] == "CREATED" and st.button(
            "Accept simulated handoff"
        ):
            handoff = call_api(
                "POST", f"/handoffs/{active['handoff_id']}/accept", {}
            )
            if handoff:
                st.session_state.active_handoff = handoff
                st.rerun()
        if active["state"] == "ACCEPTED" and st.button(
            "Resolve simulated handoff"
        ):
            handoff = call_api(
                "POST", f"/handoffs/{active['handoff_id']}/resolve", {}
            )
            if handoff:
                st.session_state.active_handoff = handoff
                st.rerun()

    st.markdown("#### Mandatory SLA demonstration")
    st.write(
        "CASE-4821 has a synthetic overdue deadline. This rule is deterministic "
        "and independent of LLM reasoning."
    )
    if st.button("Run SLA check for CASE-4821"):
        result = call_api("POST", "/sla/CASE-4821/check", {})
        if result:
            st.session_state.last_sla_result = result
    if result := st.session_state.get("last_sla_result"):
        st.json(result)
        st.caption(
            "Use the documented two-pass CLI sweep to demonstrate backup routing."
        )


chat_tab, actions_tab, evaluation_tab = st.tabs(
    ["Customer servicing", "Controlled actions", "Evaluation dashboard"]
)

with chat_tab:
    st.subheader("Authorized customer servicing")
    scenario_label = st.selectbox(
        "Reproducible demo scenario",
        [scenario.label for scenario in DEMO_SCENARIOS],
    )
    scenario = scenario_by_label(scenario_label)
    st.caption(
        f"Expected: {scenario.expected} · identity: {scenario.demo_user}"
    )
    if st.button("Run selected scenario", type="primary"):
        if submit_chat(scenario.prompt, scenario.demo_user):
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_message(message)

    if prompt := st.chat_input("Ask about a synthetic service case"):
        if submit_chat(prompt, demo_user):
            st.rerun()

with actions_tab:
    st.info(
        "Everything in this tab is a local SQLite simulation; no production CRM "
        "or queue is contacted."
    )
    note_column, escalation_column = st.columns(2)
    with note_column:
        render_controlled_note()
    with escalation_column:
        render_handoff_and_sla()

with evaluation_tab:
    render_evaluation_dashboard(
        st,
        EVALUATION_RESULTS_PATH,
        evaluation_modes=SETTINGS.evaluation_modes,
        evaluation_runs=SETTINGS.evaluation_runs,
    )
