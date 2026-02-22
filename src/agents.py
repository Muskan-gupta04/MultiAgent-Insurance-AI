"""LLM client, tools, prompts, and agent node functions."""
from src.setup import *
from src.data_pipeline import *
from src.prompts import *
from openai import OpenAI
import json
from typing import List, Dict, Any, Optional

client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

def run_llm(
    prompt: str,
    tools: Optional[List[Dict]] = None,
    tool_functions: Optional[Dict[str, Any]] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Run an LLM request that optionally supports tools.
    
    Args:
        prompt (str): The system or user prompt to send.
        tools (list[dict], optional): Tool schema list for model function calling.
        tool_functions (dict[str, callable], optional): Mapping of tool names to Python functions.
        model (str): Model name to use (default: DEFAULT_MODEL).

    Returns:
        str: Final LLM response text.
    """

    # Step 1: Initial LLM call
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        tools=tools if tools else None,
        tool_choice="auto" if tools else None
    )

    message = response.choices[0].message
    print("Initial LLM Response:", message)

    # Step 2: If no tools or no tool calls, return simple model response
    if not getattr(message, "tool_calls", None):
        return message.content

    # Step 3: Handle tool calls dynamically
    if not tool_functions:
        return message.content + "\n\n⚠️ No tool functions provided to execute tool calls."

    tool_messages = []
    for tool_call in message.tool_calls:
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments or "{}")
        tool_fn = tool_functions.get(func_name)

        try:
            result = tool_fn(**args) if tool_fn else {"error": f"Tool '{func_name}' not implemented."}
        except Exception as e:
            result = {"error": str(e)}

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

    # Step 4: Second pass — send tool outputs back to the model
    followup_messages = [
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                } for tc in message.tool_calls
            ],
        },
        *tool_messages,
    ]

    final = client.chat.completions.create(model=model, messages=followup_messages)
    return final.choices[0].message.content

# =====================================================
#  Define the TOOL functions (these run locally)
# =====================================================

import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('insurance_agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def ask_user(question: str, missing_info: str = ""):
    """Ask the user for input and return the response."""
    logger.info(f"🗣️ Asking user for input: {question}")
    if missing_info:
        print(f"---USER INPUT REQUIRED---\nMissing information: {missing_info}")
    else:
        print(f"---USER INPUT REQUIRED---")
    
    answer = input(f"{question}: ")
    return {"context": answer, "source": "User Input"}

def get_policy_details(policy_number: str) -> Dict[str, Any]:
    """Fetch a customer's policy details by policy number"""
    logger.info(f"🔍 Fetching policy details for: {policy_number}")
    conn = sqlite3.connect('insurance_support.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.*, c.first_name, c.last_name 
        FROM policies p 
        JOIN customers c ON p.customer_id = c.customer_id 
        WHERE p.policy_number = ?
    """, (policy_number,))
    result = cursor.fetchone()
    conn.close()
    if result:
        logger.info(f"✅ Policy found: {policy_number}")
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, result))
    logger.warning(f"❌ Policy not found: {policy_number}")
    return {"error": "Policy not found"}

def get_claim_status(claim_id: str = None, policy_number: str = None) -> Dict[str, Any]:
    """Get claim status and details"""
    logger.info(f"🔍 Fetching claim status - Claim ID: {claim_id}, Policy: {policy_number}")
    conn = sqlite3.connect('insurance_support.db')
    cursor = conn.cursor()
    if claim_id:
        cursor.execute("""
            SELECT c.*, p.policy_type 
            FROM claims c
            JOIN policies p ON c.policy_number = p.policy_number
            WHERE c.claim_id = ?
        """, (claim_id,))
    elif policy_number:
        cursor.execute("""
            SELECT c.*, p.policy_type 
            FROM claims c
            JOIN policies p ON c.policy_number = p.policy_number
            WHERE c.policy_number = ?
            ORDER BY c.claim_date DESC LIMIT 3
        """, (policy_number,))
    result = cursor.fetchall()
    conn.close()
    if result:
        logger.info(f"✅ Found {len(result)} claim(s)")
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in result]
    logger.warning("❌ No claims found")
    return {"error": "Claim not found"}

def get_billing_info(policy_number: str = None, customer_id: str = None) -> Dict[str, Any]:
    """Get billing information including current balance and due dates"""
    logger.info(f"🔍 Fetching billing info - Policy: {policy_number}, Customer: {customer_id}")
    conn = sqlite3.connect('insurance_support.db')
    cursor = conn.cursor()
    if policy_number:
        cursor.execute("""
            SELECT b.*, p.premium_amount, p.billing_frequency
            FROM billing b
            JOIN policies p ON b.policy_number = p.policy_number
            WHERE b.policy_number = ? AND b.status = 'pending'
            ORDER BY b.due_date DESC LIMIT 1
        """, (policy_number,))
    elif customer_id:
        cursor.execute("""
            SELECT b.*, p.premium_amount, p.billing_frequency
            FROM billing b
            JOIN policies p ON b.policy_number = p.policy_number
            WHERE p.customer_id = ? AND b.status = 'pending'
            ORDER BY b.due_date DESC LIMIT 1
        """, (customer_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        logger.info("✅ Billing info found")
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, result))
    logger.warning("❌ Billing info not found")
    return {"error": "Billing information not found"}

def get_payment_history(policy_number: str) -> List[Dict[str, Any]]:
    """Get payment history for a policy"""
    logger.info(f"🔍 Fetching payment history for policy: {policy_number}")
    conn = sqlite3.connect('insurance_support.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT p.payment_date, p.amount, p.status, p.payment_method
        FROM payments p
        JOIN billing b ON p.bill_id = b.bill_id
        WHERE b.policy_number = ?
        ORDER BY p.payment_date DESC LIMIT 10
    """, (policy_number,))
    
    results = cursor.fetchall()
    conn.close()
    
    if results:
        logger.info(f"✅ Found {len(results)} payment records")
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in results]
    logger.warning("❌ No payment history found")
    return []

def get_auto_policy_details(policy_number: str) -> Dict[str, Any]:
    """Get auto-specific policy details including vehicle info and deductibles"""
    logger.info(f"🔍 Fetching auto policy details for: {policy_number}")
    conn = sqlite3.connect('insurance_support.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT apd.*, p.policy_type, p.premium_amount
        FROM auto_policy_details apd
        JOIN policies p ON apd.policy_number = p.policy_number
        WHERE apd.policy_number = ?
    """, (policy_number,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        logger.info("✅ Auto policy details found")
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, result))
    logger.warning("❌ Auto policy details not found")
    return {"error": "Auto policy details not found"}







@trace_agent
def supervisor_agent(state):
    print("---SUPERVISOR AGENT---")
    # Increment iteration counter
    n_iter = state.get("n_iteration", 0) + 1
    state["n_iteration"] = n_iter
    print(f"🔢 Supervisor iteration: {n_iter}")

    # Force end if iteration limit reached
    # Escalate to human support if iteration limit reached
    if n_iter >= 3:
        print("⚠️ Maximum supervisor iterations reached — escalating to human agent")
        updated_history = (
            state.get("conversation_history", "")
            + "\nAssistant: It seems this issue requires human review. Escalating to a human support specialist."
        )
        return {
            "escalate_to_human": True,
            "conversation_history": updated_history,
            "next_agent": "human_escalation_agent",
            "n_iteration": n_iter
        }
    
    # Check if we're coming from a clarification
    if state.get("needs_clarification", False):
        user_clarification = state.get("user_clarification", "")
        print(f"🔄 Processing user clarification: {user_clarification}")
        
        # Update conversation history with the clarification exchange
        clarification_question = state.get("clarification_question", "")
        updated_conversation = state.get("conversation_history", "") + f"\nAssistant: {clarification_question}\nUser: {user_clarification}"
        
        # Update state to clear clarification flags and update history
        updated_state = state.copy()
        updated_state["needs_clarification"] = False
        updated_state["conversation_history"] = updated_conversation
        
        # Clear clarification fields
        if "clarification_question" in updated_state:
            del updated_state["clarification_question"]
        if "user_clarification" in updated_state:
            del updated_state["user_clarification"]
            
        return updated_state

    user_query = state["user_input"]
    conversation_history = state.get("conversation_history", "")
    
    
    print(f"User Query: {user_query}")
    print(f"Conversation History: {conversation_history}")
    
    
    # Include the ENTIRE conversation history in the prompt
    full_context = f"Full Conversation:\n{conversation_history}"

    
    prompt = SUPERVISOR_PROMPT.format(
        conversation_history=full_context,  # Use full context instead of just history
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": "Ask the user for clarification or additional information when their query is unclear or missing important details. ONLY use this if essential information like policy number or customer ID is missing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The specific question to ask the user for clarification"
                        },
                        "missing_info": {
                            "type": "string", 
                            "description": "What specific information is missing or needs clarification"
                        }
                    },
                    "required": ["question", "missing_info"]
                }
            }
        }
    ]

    print("🤖 Calling LLM for supervisor decision...")
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # Check if supervisor wants to ask user for clarification
    if getattr(message, "tool_calls", None):
        print("🛠️ Supervisor requesting user clarification")
        for tool_call in message.tool_calls:
            if tool_call.function.name == "ask_user":
                args = json.loads(tool_call.function.arguments)
                question = args.get("question", "Can you please provide more details?")
                missing_info = args.get("missing_info", "additional information")
                
                print(f"❓ Asking user: {question}")
             
                
                user_response_data = ask_user(question, missing_info)
                user_response = user_response_data["context"]
                
                print(f"✅ User response: {user_response}")
                
                # Update conversation history with the question
                updated_history = conversation_history + f"\nAssistant: {question}"
                updated_history = updated_history + f"\nUser: {user_response}"
                
                return {
                    "needs_clarification": True,
                    "clarification_question": question,
                    "user_clarification": user_response,
                    "conversation_history": updated_history
                }

    # If no tool calls, proceed with normal supervisor decision
    message_content = message.content
    
    try:
        parsed = json.loads(message_content)
        print("✅ Supervisor output parsed successfully")
    except json.JSONDecodeError:
        print("❌ Supervisor output invalid JSON, using fallback")
        parsed = {}

    next_agent = parsed.get("next_agent", "general_help_agent")
    task = parsed.get("task", "Assist the user with their query.")
    justification = parsed.get("justification", "")

    print(f"---SUPERVISOR DECISION: {next_agent}---")
    print(f"Task: {task}")
    print(f"Reason: {justification}")

    # Update conversation history with the current exchange
    updated_conversation = conversation_history + f"\nAssistant: Routing to {next_agent} for: {task}"


    print(f"➡️ Routing to: {next_agent}")
    return {
        "next_agent": next_agent,
        "task": task,
        "justification": justification,
        "conversation_history": updated_conversation,
        "n_iteration": n_iter
    }

@trace_agent
def claims_agent_node(state):
    logger.info("🏥 Claims agent started")
    logger.debug(f"Claims agent state: { {k: v for k, v in state.items() if k != 'messages'} }")
    
    prompt = CLAIMS_AGENT_PROMPT.format(
        task=state.get("task"),
        policy_number=state.get("policy_number", "Not provided"),
        claim_id=state.get("claim_id", "Not provided"),
        conversation_history=state.get("conversation_history", "")
    )

    tools = [
        {"type": "function", "function": {
            "name": "get_claim_status",
            "description": "Retrieve claim details",
            "parameters": {"type": "object", "properties": {"claim_id": {"type": "string"}, "policy_number": {"type": "string"}}}
        }}
    ]

    result = run_llm(prompt, tools, {"get_claim_status": get_claim_status})
    
    logger.info("✅ Claims agent completed")
    return {"messages": [("assistant", result)]}

@trace_agent
def final_answer_agent(state):
    """Generate a clean final summary before ending the conversation"""
    print("---FINAL ANSWER AGENT---")
    logger.info("🎯 Final answer agent started")
    
    user_query = state["user_input"]
    conversation_history = state.get("conversation_history", "")
    
    # Extract the most recent specialist response
    recent_responses = []
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, 'content') and "clarification" not in msg.content.lower():
            recent_responses.append(msg.content)
            if len(recent_responses) >= 2:  # Get last 2 non-clarification responses
                break
    
    specialist_response = recent_responses[0] if recent_responses else "No response available"
    
    prompt = FINAL_ANSWER_PROMPT.format(

        specialist_response=specialist_response,  
        user_query=user_query,
    )
    
    print("🤖 Generating final summary...")
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    final_answer = response.choices[0].message.content
    
    print(f"✅ Final answer: {final_answer}")
    
    # Replace all previous messages with just the final answer
    clean_messages = [("assistant", final_answer)]

    state["final_answer"] = final_answer
    state["end_conversation"] = True
    state["conversation_history"] = conversation_history + f"\nAssistant: {final_answer}"
    state["messages"] = clean_messages
    
    return state


    
@trace_agent
def policy_agent_node(state):
    print("---POLICY AGENT---")
    logger.info("📄 Policy agent started")
    logger.debug(f"Policy agent state: { {k: v for k, v in state.items() if k != 'messages'} }")
    
    prompt = POLICY_AGENT_PROMPT.format(
        task=state.get("task"),
        policy_number=state.get("policy_number", "Not provided"),
        customer_id=state.get("customer_id", "Not provided"),
        conversation_history=state.get("conversation_history", "")
    )

    tools = [
        {"type": "function", "function": {
            "name": "get_policy_details",
            "description": "Fetch policy info by policy number",
            "parameters": {"type": "object", "properties": {"policy_number": {"type": "string"}}}
        }},
        {"type": "function", "function": {
            "name": "get_auto_policy_details",
            "description": "Get auto policy details",
            "parameters": {"type": "object", "properties": {"policy_number": {"type": "string"}}}
        }}
    ]

    print("🔄 Processing policy request...")
    result = run_llm(prompt, tools, {
        "get_policy_details": get_policy_details,
        "get_auto_policy_details": get_auto_policy_details
    })
    
    print("✅ Policy agent completed")
    return {"messages": [("assistant", result)]}

@trace_agent
def billing_agent_node(state):
    print("---BILLING AGENT---")
    print("TASK: ", state.get("task"))
    print("USER QUERY: ", state.get("user_input"))
    print("CONVERSATION HISTORY: ", state.get("conversation_history", ""))
    
    
    prompt = BILLING_AGENT_PROMPT.format(
        task=state.get("task"),
        conversation_history=state.get("conversation_history", "")
    )

    tools = [
        {"type": "function", "function": {
            "name": "get_billing_info",
            "description": "Retrieve billing information",
            "parameters": {"type": "object", "properties": {"policy_number": {"type": "string"}, "customer_id": {"type": "string"}}}
        }},
        {"type": "function", "function": {
            "name": "get_payment_history",
            "description": "Fetch recent payment history",
            "parameters": {"type": "object", "properties": {"policy_number": {"type": "string"}}}
        }}
    ]

    print("🔄 Processing billing request...")
    result = run_llm(prompt, tools, {
        "get_billing_info": get_billing_info,
        "get_payment_history": get_payment_history
    })
    
    print("✅ Billing agent completed")
    
    # Extract and preserve policy number if mentioned in the conversation
    updated_state = {"messages": [("assistant", result)]}
    
    # If we have a policy number in state, preserve it
    if state.get("policy_number"):
        updated_state["policy_number"] = state["policy_number"]
    if state.get("customer_id"):
        updated_state["customer_id"] = state["customer_id"]
        
    # Update conversation history
    current_history = state.get("conversation_history", "")
    updated_state["conversation_history"] = current_history + f"\nBilling Agent: {result}"
    
    return updated_state


@trace_agent
def general_help_agent_node(state):
    print("---GENERAL HELP AGENT---")

    user_query = state.get("user_input", "")
    conversation_history = state.get("conversation_history", "")
    task = state.get("task", "General insurance support")

    # Step 1: Retrieve relevant FAQs from the vector DB
    print("🔍 Retrieving FAQs...")
    logger.info("🔍 Retrieving FAQs from vector database")
    results = collection.query(
        query_texts=[user_query],
        n_results=3,
        include=["metadatas", "documents", "distances"]
    )

    # Step 2: Format retrieved FAQs
    faq_context = ""
    if results and results.get("metadatas") and results["metadatas"][0]:
        print(f"📚 Found {len(results['metadatas'][0])} relevant FAQs")
        for i, meta in enumerate(results["metadatas"][0]):
            q = meta.get("question", "")
            a = meta.get("answer", "")
            score = results["distances"][0][i]
            faq_context += f"FAQ {i+1} (score: {score:.3f})\nQ: {q}\nA: {a}\n\n"
    else:
        print("❌ No relevant FAQs found")
        faq_context = "No relevant FAQs were found."

    # Step 3: Format the final prompt
    prompt = GENERAL_HELP_PROMPT.format(
        task=task,
        conversation_history=conversation_history,
        faq_context=faq_context
    )

    print("🤖 Calling LLM for general response...")
    final_answer = run_llm(prompt)

    
    
    print("✅ General help agent completed")
    updated_state = {
                        "messages": [("assistant", final_answer)],
                        "retrieved_faqs": results.get("metadatas", []),
                    }


    updated_state["conversation_history"] = conversation_history + f"\nGeneral Help Agent: {final_answer}"

    return updated_state

@trace_agent
def human_escalation_node(state):
    print("---HUMAN ESCALATION AGENT---")
    logger.warning(f"Escalation triggered - State: { {k: v for k, v in state.items() if k != 'messages'} }")
    
    prompt = HUMAN_ESCALATION_PROMPT.format(
        task=state.get("task"),
        #user_query=state.get("user_input"),
        conversation_history=state.get("conversation_history", "")
    )

    print("🤖 Generating escalation response...")
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    print("🚨 Conversation escalated to human")
    return {
        "final_answer": response.choices[0].message.content,
        "requires_human_escalation": True,
        "escalation_reason": "Customer requested human assistance.",
        "messages": [("assistant", response.choices[0].message.content)]
    }



