import sys
import os

# Add parent directory to path so mas can be resolved
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from mas import resources
from mas.run import init_all
from mas.agents import (
    supervisor_agent,
    claims_agent_node,
    policy_agent_node,
    billing_agent_node,
    general_help_agent_node,
    human_escalation_node,
    final_answer_agent
)

# 1. Initialize logic and seed DB if necessary (using existing POL000001)
print("Initializing resources...")
init_all(setup_db=True, ingest_faq=True)

# 2. Define a base state for testing
# We use POL000001 which we know exists from our tool tests
test_state = {
    "user_input": "I need help with my policy details and billing",
    "conversation_history": "User: I need help with my policy details and billing. My policy is POL000001.",
    "messages": [("user", "I need help with my policy details and billing. My policy is POL000001.")],
    "task": "Retrieve policy and billing details for POL000001",
    "policy_number": "POL000001",
    "customer_id": "CUST00023",
    "n_iteration": 0
}

def test_agent(name, func, state):
    print(f"\n--- Testing {name} ---")
    try:
        res = func(state.copy())
        print(f"Result keys: {list(res.keys())}")
        if "messages" in res:
            print(f"Response: {res['messages'][-1][1][:100]}...")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# List of agents to test
agents_to_test = [
    ("General Help Agent", general_help_agent_node),
    ("Policy Agent", policy_agent_node),
    ("Billing Agent", billing_agent_node),
    ("Claims Agent", claims_agent_node),
    ("Human Escalation Agent", human_escalation_node),
    ("Final Answer Agent", final_answer_agent),
    ("Supervisor Agent", supervisor_agent)
]

# We need to add a specialist message for Final Answer Agent to work
test_state["messages"].append(("assistant", "I have retrieved your policy details now."))

success_count = 0
for name, func in agents_to_test:
    if test_agent(name, func, test_state):
        success_count += 1

print(f"\n{'='*30}")
print(f"TEST SUMMARY: {success_count}/{len(agents_to_test)} agents working.")
print(f"{'='*30}")
