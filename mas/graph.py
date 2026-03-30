from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph, add_messages

from .agents import (
    billing_agent_node,
    claims_agent_node,
    final_answer_agent,
    general_help_agent_node,
    human_escalation_node,
    policy_agent_node,
    supervisor_agent,
)
from .resources import trace_agent


class GraphState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    user_input: str
    conversation_history: Optional[str]

    n_iteration: Optional[int]

    user_intent: Optional[str]
    customer_id: Optional[str]
    policy_number: Optional[str]
    claim_id: Optional[str]

    next_agent: Optional[str]
    task: Optional[str]
    justification: Optional[str]
    end_conversation: Optional[bool]

    extracted_entities: Dict[str, Any]
    database_lookup_result: Dict[str, Any]

    requires_human_escalation: bool
    escalation_reason: Optional[str]

    billing_amount: Optional[float]
    payment_method: Optional[str]
    billing_frequency: Optional[str]
    invoice_date: Optional[str]

    timestamp: Optional[str]
    final_answer: Optional[str]


def decide_next_agent(state):
    if state.get("needs_clarification"):
        return "supervisor_agent"

    if state.get("end_conversation"):
        return "end"

    if state.get("requires_human_escalation"):
        return "human_escalation_agent"

    return state.get("next_agent", "general_help_agent")


def build_app():
    wrap = trace_agent if trace_agent else (lambda f: f)

    workflow = StateGraph(GraphState)

    workflow.add_node("supervisor_agent", wrap(supervisor_agent))
    workflow.add_node("policy_agent", wrap(policy_agent_node))
    workflow.add_node("billing_agent", wrap(billing_agent_node))
    workflow.add_node("claims_agent", wrap(claims_agent_node))
    workflow.add_node("general_help_agent", wrap(general_help_agent_node))
    workflow.add_node("human_escalation_agent", wrap(human_escalation_node))
    workflow.add_node("final_answer_agent", wrap(final_answer_agent))

    workflow.set_entry_point("supervisor_agent")

    workflow.add_conditional_edges(
        "supervisor_agent",
        decide_next_agent,
        {
            "supervisor_agent": "supervisor_agent",
            "policy_agent": "policy_agent",
            "billing_agent": "billing_agent",
            "claims_agent": "claims_agent",
            "human_escalation_agent": "human_escalation_agent",
            "general_help_agent": "general_help_agent",
            "end": "final_answer_agent",
        },
    )

    for node in ["policy_agent", "billing_agent", "claims_agent", "general_help_agent"]:
        workflow.add_edge(node, "supervisor_agent")

    workflow.add_edge("final_answer_agent", END)
    workflow.add_edge("human_escalation_agent", END)

    return workflow.compile()
