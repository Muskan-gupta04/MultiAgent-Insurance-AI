import sys
import os

# Add parent directory to path so mas can be resolved
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from mas import resources
from mas import tools
# MOCK ask_user to prevent hanging
tools.ask_user = lambda q, m: {"context": "Here is the details: POL-123456", "source": "User Mock"}

from mas.agents import (
    supervisor_agent,
    claims_agent_node,
    policy_agent_node,
    billing_agent_node,
    general_help_agent_node,
    human_escalation_node,
    final_answer_agent
)

# Initialize resources
resources.init_resources()

state = {
    "user_input": "What is my policy checking and claims?",
    "conversation_history": "User: What is my policy checking and claims? My policy number is POL-123456.",
    "messages": [("user", "What is my policy checking and claims? My policy number is POL-123456.")],
    "task": "Check policy and claims details.",
    "policy_number": "POL-123456",
    "customer_id": "CUST-9876",
    "claim_id": "CLM-5555"
}

print("Testing General Help Agent...")
try:
    res = general_help_agent_node(state)
    print("General Help Agent OK:", res.keys() if isinstance(res, dict) else res)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("General Help Error:", e)

print("\nTesting Policy Agent...")
try:
    res = policy_agent_node(state)
    print("Policy Agent OK:", res.keys() if isinstance(res, dict) else res)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Policy Agent Error:", e)

print("\nTesting Billing Agent...")
try:
    res = billing_agent_node(state)
    print("Billing Agent OK:", res.keys() if isinstance(res, dict) else res)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Billing Agent Error:", e)

print("\nTesting Claims Agent...")
try:
    res = claims_agent_node(state)
    print("Claims Agent OK:", res.keys() if isinstance(res, dict) else res)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Claims Agent Error:", e)

print("\nTesting Human Escalation Agent...")
try:
    res = human_escalation_node(state)
    print("Human Escalation Agent OK:", res.keys() if isinstance(res, dict) else res)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Human Escalation Agent Error:", e)

print("\nTesting Final Answer Agent...")
try:
    # We need a specialist response in messages
    state["messages"].append(("assistant", "Here are your policy details..."))
    res = final_answer_agent(state)
    print("Final Answer Agent OK:", res.keys() if isinstance(res, dict) else res)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Final Answer Agent Error:", e)

print("\nTesting Supervisor Agent...")
try:
    res = supervisor_agent(state)
    print("Supervisor Agent OK:", res.keys() if isinstance(res, dict) else res)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Supervisor Agent Error:", e)
