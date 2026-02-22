"""LangGraph workflow assembly and test runner."""
from src.agents import *
from langgraph.graph import StateGraph, END


from typing import TypedDict, List, Annotated, Dict, Any, Optional
from langgraph.graph import add_messages
from datetime import datetime

class GraphState(TypedDict):
    # Core conversation tracking
    messages: Annotated[List[Any], add_messages]
    user_input: str
    conversation_history: Optional[str]

    n_iteration: Optional[int]

    # Extracted context & metadata
    user_intent: Optional[str]            # e.g., "query_policy", "billing_issue"
    customer_id: Optional[str]
    policy_number: Optional[str]
    claim_id: Optional[str]
    
    # Supervisor / routing layer
    next_agent: Optional[str]             # e.g., "policy_agent", "claims_agent", etc.
    task: Optional[str]                   # Current task determined by supervisor
    justification: Optional[str]          # Supervisor reasoning/explanation
    end_conversation: Optional[bool]      # Flag for graceful conversation termination
    
    # Entity extraction and DB lookups
    extracted_entities: Dict[str, Any]    # Parsed from user input (dates, names, etc.)
    database_lookup_result: Dict[str, Any]
    
    # Escalation state
    requires_human_escalation: bool
    escalation_reason: Optional[str]
    
    # Billing-specific fields
    billing_amount: Optional[float]
    payment_method: Optional[str]
    billing_frequency: Optional[str]      # "monthly", "quarterly", "annual"
    invoice_date: Optional[str]
    
    # System-level metadata
    timestamp: Optional[str]     # Track time of latest user message or state update
    final_answer: Optional[str]


def decide_next_agent(state):
    # Handle clarification case first
    if state.get("needs_clarification"):
        return "supervisor_agent"  # Return to supervisor to process the clarification
    
    if state.get("end_conversation"):
        return "end"
    
    if state.get("requires_human_escalation"):
        return "human_escalation_agent"
    
    return state.get("next_agent", "general_help_agent")


# Update the workflow to include the final_answer_agent
workflow = StateGraph(GraphState)

workflow.add_node("supervisor_agent", supervisor_agent)
workflow.add_node("policy_agent", policy_agent_node)
workflow.add_node("billing_agent", billing_agent_node)
workflow.add_node("claims_agent", claims_agent_node)
workflow.add_node("general_help_agent", general_help_agent_node)
workflow.add_node("human_escalation_agent", human_escalation_node)
workflow.add_node("final_answer_agent", final_answer_agent)  # Add this

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
        "end": "final_answer_agent"
    }
)

# Return to Supervisor after each specialist
for node in ["policy_agent", "billing_agent", "claims_agent", "general_help_agent"]:
    workflow.add_edge(node, "supervisor_agent")

# Final answer agent → END
workflow.add_edge("final_answer_agent", END)

# Human escalation → END
workflow.add_edge("human_escalation_agent", END)

app = workflow.compile()


# === Display the Graph ===
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))



def run_test_query(query):
    """Test the system with a billing query"""
    initial_state = {
        "n_iteraton":0,
        "messages": [],
        "user_input": query,
        "user_intent": "",
        "claim_id": "",
        "next_agent": "supervisor_agent",
        "extracted_entities": {},
        "database_lookup_result": {},
        "requires_human_escalation": False,
        "escalation_reason": "",
        "billing_amount": None,
        "payment_method": None,
        "billing_frequency": None,
        "invoice_date": None,
        "conversation_history": f"User: {query}", 
        "task": "Help user with their query",
        "final_answer": ""
    }
    
    print(f"\n{'='*50}")
    print(f"QUERY: {query}")
    print(f"\n{'='*50}")
    
    # Run the graph
    final_state = app.invoke(initial_state)
    
    # Print the response
    print("\n---FINAL RESPONSE---")
    final_answer = final_state.get("final_answer", "No final answer generated.")
    print(final_answer)
    
    
    return final_state


def run_default_demo_query():
    return run_test_query("In general, what does life insurance cover?")
