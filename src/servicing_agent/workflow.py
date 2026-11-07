from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .auth import DemoUserContext
from .logging_config import log_event
from .tools import (
    CRMAuthorizationError,
    MockCRMTool,
    MockKnowledgeTool,
    MockPolicyTool,
    TicketNotFoundError,
)

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict, total=False):
    message: str
    execution_id: str
    user_context: DemoUserContext
    case_id: str | None
    topic: str
    ticket: dict[str, Any] | None
    error: str | None
    knowledge_guidance: str
    policy_guidance: str
    reply: str
    steps: Annotated[list[str], operator.add]


def build_complex_workflow(
    crm_tool: MockCRMTool,
    knowledge_tool: MockKnowledgeTool,
    policy_tool: MockPolicyTool,
):
    def triage(state: WorkflowState) -> dict[str, Any]:
        topic = "case_follow_up" if state.get("case_id") else "general_servicing"
        log_event(
            logger,
            "workflow.triage",
            execution_id=state["execution_id"],
            topic=topic,
        )
        return {"topic": topic, "steps": ["workflow:triage"]}

    def load_case_context(state: WorkflowState) -> dict[str, Any]:
        case_id = state.get("case_id")
        if case_id is None:
            return {
                "ticket": None,
                "error": None,
                "steps": ["mock_crm:no_case_reference"],
            }
        try:
            ticket = crm_tool.get_ticket(case_id, state["user_context"])
        except CRMAuthorizationError:
            result = {"ticket": None, "error": "unauthorized"}
        except TicketNotFoundError:
            result = {"ticket": None, "error": "not_found"}
        else:
            result = {"ticket": ticket, "error": None}

        log_event(
            logger,
            "workflow.crm_lookup",
            execution_id=state["execution_id"],
            ticket_id=case_id,
            outcome=result["error"] or "authorized",
        )
        return {**result, "steps": ["mock_crm:get_ticket"]}

    def gather_mock_guidance(state: WorkflowState) -> dict[str, Any]:
        if state.get("error"):
            return {
                "knowledge_guidance": "",
                "policy_guidance": "",
                "steps": ["workflow:guidance_skipped"],
            }
        return {
            "knowledge_guidance": knowledge_tool.search(state["message"]),
            "policy_guidance": policy_tool.guidance(state["message"]),
            "steps": ["mock_knowledge:search", "mock_policy:guidance"],
        }

    def compose_response(state: WorkflowState) -> dict[str, Any]:
        case_id = state.get("case_id")
        if state.get("error") == "unauthorized":
            reply = (
                f"You are not authorized to view {case_id} with the selected "
                "demo-user context."
            )
        elif state.get("error") == "not_found":
            reply = f"I could not find {case_id} in the local demo CRM."
        else:
            ticket = state.get("ticket")
            case_context = ""
            if ticket:
                case_context = (
                    f" I found {ticket['ticket_id']} ({ticket['status']}, "
                    f"{ticket['priority']} priority)."
                )
            reply = (
                f"The Milestone 1 workflow categorized this as "
                f"{state['topic'].replace('_', ' ')}.{case_context} "
                f"Mock guidance: {state['knowledge_guidance']} "
                f"{state['policy_guidance']}"
            )
        return {"reply": reply, "steps": ["workflow:compose_response"]}

    graph = StateGraph(WorkflowState)
    graph.add_node("triage", triage)
    graph.add_node("load_case_context", load_case_context)
    graph.add_node("gather_mock_guidance", gather_mock_guidance)
    graph.add_node("compose_response", compose_response)
    graph.add_edge(START, "triage")
    graph.add_edge("triage", "load_case_context")
    graph.add_edge("load_case_context", "gather_mock_guidance")
    graph.add_edge("gather_mock_guidance", "compose_response")
    graph.add_edge("compose_response", END)
    return graph.compile()

