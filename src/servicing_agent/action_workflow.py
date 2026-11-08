from __future__ import annotations

import operator
from typing import Annotated, Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .auth import DemoUserContext
from .schemas import OperationResponse


class NoteActionState(TypedDict, total=False):
    command: Literal["prepare", "confirm", "cancel", "reconcile", "status"]
    operation_id: str | None
    ticket_id: str | None
    note: str | None
    user_context: DemoUserContext
    customer_confirmed: bool
    simulate_timeout_after_success: bool
    result: OperationResponse
    steps: Annotated[list[str], operator.add]


def build_note_action_workflow(
    executor: Callable[[NoteActionState], OperationResponse],
):
    """Build the compact action graph; durable state lives in SQLite operations."""

    def execute(state: NoteActionState) -> dict[str, Any]:
        return {
            "result": executor(state),
            "steps": [f"controlled_action:{state['command']}"],
        }

    graph = StateGraph(NoteActionState)
    graph.add_node("execute", execute)
    graph.add_edge(START, "execute")
    graph.add_edge("execute", END)
    return graph.compile()
