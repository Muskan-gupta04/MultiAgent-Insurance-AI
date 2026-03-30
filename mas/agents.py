import json

from .llm import run_llm
from .prompts import (
    BILLING_AGENT_PROMPT,
    CLAIMS_AGENT_PROMPT,
    FINAL_ANSWER_PROMPT,
    GENERAL_HELP_PROMPT,
    HUMAN_ESCALATION_PROMPT,
    POLICY_AGENT_PROMPT,
    SUPERVISOR_PROMPT,
)
from . import resources
from . import resources
from .tools import (
    ask_user,
    get_auto_policy_details,
    get_billing_info,
    get_claim_status,
    get_payment_history,
    get_policy_details,
)


def supervisor_agent(state):
    print("---SUPERVISOR AGENT---")
    n_iter = state.get("n_iteration", 0) + 1
    state["n_iteration"] = n_iter
    print(f"Supervisor iteration: {n_iter}")

    if n_iter >= 3:
        print("Maximum supervisor iterations reached - escalating to human agent")
        updated_history = (
            state.get("conversation_history", "")
            + "\nAssistant: It seems this issue requires human review. Escalating to a human support specialist."
        )
        return {
            "escalate_to_human": True,
            "conversation_history": updated_history,
            "next_agent": "human_escalation_agent",
            "n_iteration": n_iter,
        }

    if state.get("needs_clarification", False):
        user_clarification = state.get("user_clarification", "")
        print(f"Processing user clarification: {user_clarification}")

        clarification_question = state.get("clarification_question", "")
        updated_conversation = (
            state.get("conversation_history", "")
            + f"\nAssistant: {clarification_question}\nUser: {user_clarification}"
        )

        updated_state = state.copy()
        updated_state["needs_clarification"] = False
        updated_state["conversation_history"] = updated_conversation

        if "clarification_question" in updated_state:
            del updated_state["clarification_question"]
        if "user_clarification" in updated_state:
            del updated_state["user_clarification"]

        return updated_state

    user_query = state["user_input"]
    conversation_history = state.get("conversation_history", "")

    print(f"User Query: {user_query}")
    print(f"Conversation History: {conversation_history}")

    full_context = f"Full Conversation:\n{conversation_history}"

    prompt = SUPERVISOR_PROMPT.format(
        conversation_history=full_context,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": (
                    "Ask the user for clarification or additional information when their query is "
                    "unclear or missing important details. ONLY use this if essential information "
                    "like policy number or customer ID is missing."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The specific question to ask the user for clarification",
                        },
                        "missing_info": {
                            "type": "string",
                            "description": "What specific information is missing or needs clarification",
                        },
                    },
                    "required": ["question", "missing_info"],
                },
            },
        }
    ]

    print("Calling LLM for supervisor decision...")
    response = resources.client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": prompt}],
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message

    if getattr(message, "tool_calls", None):
        print("Supervisor requesting user clarification")
        for tool_call in message.tool_calls:
            if tool_call.function.name == "ask_user":
                args = json.loads(tool_call.function.arguments)
                question = args.get("question", "Can you please provide more details?")
                missing_info = args.get("missing_info", "additional information")

                print(f"Asking user: {question}")

                user_response_data = ask_user(question, missing_info)
                user_response = user_response_data["context"]

                print(f"User response: {user_response}")

                updated_history = conversation_history + f"\nAssistant: {question}"
                updated_history = updated_history + f"\nUser: {user_response}"

                return {
                    "needs_clarification": True,
                    "clarification_question": question,
                    "user_clarification": user_response,
                    "conversation_history": updated_history,
                }

    message_content = message.content

    try:
        parsed = json.loads(message_content)
        print("Supervisor output parsed successfully")
    except json.JSONDecodeError:
        print("Supervisor output invalid JSON, using fallback")
        parsed = {}

    next_agent = parsed.get("next_agent", "general_help_agent")
    task = parsed.get("task", "Assist the user with their query.")
    justification = parsed.get("justification", "")

    print(f"---SUPERVISOR DECISION: {next_agent}---")
    print(f"Task: {task}")
    print(f"Reason: {justification}")

    updated_conversation = conversation_history + f"\nAssistant: Routing to {next_agent} for: {task}"

    print(f"Routing to: {next_agent}")
    return {
        "next_agent": next_agent,
        "task": task,
        "justification": justification,
        "conversation_history": updated_conversation,
        "n_iteration": n_iter,
    }


def claims_agent_node(state):
    resources.logger.info("Claims agent started")
    resources.logger.debug(f"Claims agent state: { {k: v for k, v in state.items() if k != 'messages'} }")

    prompt = CLAIMS_AGENT_PROMPT.format(
        task=state.get("task"),
        policy_number=state.get("policy_number", "Not provided"),
        claim_id=state.get("claim_id", "Not provided"),
        conversation_history=state.get("conversation_history", ""),
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_claim_status",
                "description": "Retrieve claim details",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "policy_number": {"type": "string"},
                    },
                },
            },
        }
    ]

    result = run_llm(resources.client, prompt, tools, {"get_claim_status": get_claim_status})
    resources.logger.info("Claims agent completed")
    return {"messages": [("assistant", result)]}


def final_answer_agent(state):
    print("---FINAL ANSWER AGENT---")
    resources.logger.info("Final answer agent started")

    user_query = state["user_input"]
    conversation_history = state.get("conversation_history", "")

    recent_responses = []
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "content") and "clarification" not in msg.content.lower():
            recent_responses.append(msg.content)
            if len(recent_responses) >= 2:
                break

    specialist_response = recent_responses[0] if recent_responses else "No response available"

    prompt = FINAL_ANSWER_PROMPT.format(
        specialist_response=specialist_response,
        user_query=user_query,
    )

    print("Generating final summary...")
    response = resources.client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": prompt}],
    )

    final_answer = response.choices[0].message.content
    print(f"Final answer: {final_answer}")

    clean_messages = [("assistant", final_answer)]

    state["final_answer"] = final_answer
    state["end_conversation"] = True
    state["conversation_history"] = conversation_history + f"\nAssistant: {final_answer}"
    state["messages"] = clean_messages

    return state


def policy_agent_node(state):
    print("---POLICY AGENT---")
    resources.logger.info("Policy agent started")
    resources.logger.debug(f"Policy agent state: { {k: v for k, v in state.items() if k != 'messages'} }")

    prompt = POLICY_AGENT_PROMPT.format(
        task=state.get("task"),
        policy_number=state.get("policy_number", "Not provided"),
        customer_id=state.get("customer_id", "Not provided"),
        conversation_history=state.get("conversation_history", ""),
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_policy_details",
                "description": "Fetch policy info by policy number",
                "parameters": {
                    "type": "object",
                    "properties": {"policy_number": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_auto_policy_details",
                "description": "Get auto policy details",
                "parameters": {
                    "type": "object",
                    "properties": {"policy_number": {"type": "string"}},
                },
            },
        },
    ]

    print("Processing policy request...")
    result = run_llm(
        resources.client,
        prompt,
        tools,
        {
            "get_policy_details": get_policy_details,
            "get_auto_policy_details": get_auto_policy_details,
        },
    )

    print("Policy agent completed")
    return {"messages": [("assistant", result)]}


def billing_agent_node(state):
    print("---BILLING AGENT---")
    print("TASK: ", state.get("task"))
    print("USER QUERY: ", state.get("user_input"))
    print("CONVERSATION HISTORY: ", state.get("conversation_history", ""))

    prompt = BILLING_AGENT_PROMPT.format(
        task=state.get("task"),
        conversation_history=state.get("conversation_history", ""),
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_billing_info",
                "description": "Retrieve billing information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "policy_number": {"type": "string"},
                        "customer_id": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_payment_history",
                "description": "Fetch recent payment history",
                "parameters": {
                    "type": "object",
                    "properties": {"policy_number": {"type": "string"}},
                },
            },
        },
    ]

    print("Processing billing request...")
    result = run_llm(
        resources.client,
        prompt,
        tools,
        {
            "get_billing_info": get_billing_info,
            "get_payment_history": get_payment_history,
        },
    )

    print("Billing agent completed")

    updated_state = {"messages": [("assistant", result)]}

    if state.get("policy_number"):
        updated_state["policy_number"] = state["policy_number"]
    if state.get("customer_id"):
        updated_state["customer_id"] = state["customer_id"]

    current_history = state.get("conversation_history", "")
    updated_state["conversation_history"] = current_history + f"\nBilling Agent: {result}"

    return updated_state


def general_help_agent_node(state):
    print("---GENERAL HELP AGENT---")

    user_query = state.get("user_input", "")
    conversation_history = state.get("conversation_history", "")
    task = state.get("task", "General insurance support")

    print("Retrieving FAQs...")
    resources.logger.info("Retrieving FAQs from vector database")
    results = resources.collection.query(
        query_texts=[user_query],
        n_results=3,
        include=["metadatas", "documents", "distances"],
    )

    faq_context = ""
    if results and results.get("metadatas") and results["metadatas"][0]:
        print(f"Found {len(results['metadatas'][0])} relevant FAQs")
        for i, meta in enumerate(results["metadatas"][0]):
            q = meta.get("question", "")
            a = meta.get("answer", "")
            score = results["distances"][0][i]
            faq_context += f"FAQ {i+1} (score: {score:.3f})\nQ: {q}\nA: {a}\n\n"
    else:
        print("No relevant FAQs found")
        faq_context = "No relevant FAQs were found."

    prompt = GENERAL_HELP_PROMPT.format(
        task=task,
        conversation_history=conversation_history,
        faq_context=faq_context,
    )

    print("Calling LLM for general response...")
    final_answer = run_llm(resources.client, prompt)

    print("General help agent completed")
    updated_state = {
        "messages": [("assistant", final_answer)],
        "retrieved_faqs": results.get("metadatas", []),
    }

    updated_state["conversation_history"] = conversation_history + f"\nGeneral Help Agent: {final_answer}"

    return updated_state


def human_escalation_node(state):
    print("---HUMAN ESCALATION AGENT---")
    resources.logger.warning(f"Escalation triggered - State: { {k: v for k, v in state.items() if k != 'messages'} }")

    prompt = HUMAN_ESCALATION_PROMPT.format(
        task=state.get("task"),
        conversation_history=state.get("conversation_history", ""),
    )

    print("Generating escalation response...")
    response = resources.client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": prompt}],
    )

    print("Conversation escalated to human")
    return {
        "final_answer": response.choices[0].message.content,
        "requires_human_escalation": True,
        "escalation_reason": "Customer requested human assistance.",
        "messages": [("assistant", response.choices[0].message.content)],
    }
