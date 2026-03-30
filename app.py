import datetime
import streamlit as st

# Must be run from the root directory so `mas` is importable
from mas.run import build_app_with_init

# Set page config
st.set_page_config(page_title="Insurance AI Assistant", page_icon="🛡️", layout="wide")

st.title("🛡️ Insurance AI Assistant")
st.markdown("Ask me questions about your policy, claims, billing, or general queries.")

# Initialize backend app in session state using caching to avoid re-initializing
@st.cache_resource
def load_app():
    return build_app_with_init()

try:
    with st.spinner("Initializing AI Insurance system..."):
        app = load_app()
except Exception as e:
    st.error(f"Failed to initialize backend: {e}")
    st.stop()

# Initialize session state for messages and history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for settings (optional, but good for test values)
with st.sidebar:
    st.header("Session Context")
    st.markdown("Enter optional context values to simulate a logged-in user.")
    customer_id = st.text_input("Customer ID (Optional)", value="")
    policy_number = st.text_input("Policy Number (Optional)", value="")
    claim_id = st.text_input("Claim ID (Optional)", value="")
    
    st.divider()
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build conversation string
    history_str = ""
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    # Initial state mimicking GraphState
    initial_state = {
        "n_iteration": 0,
        "messages": [],
        "user_input": prompt,
        "user_intent": "",
        "customer_id": customer_id if customer_id else "",
        "policy_number": policy_number if policy_number else "",
        "claim_id": claim_id if claim_id else "",
        "next_agent": "supervisor_agent",
        "extracted_entities": {},
        "database_lookup_result": {},
        "requires_human_escalation": False,
        "escalation_reason": "",
        "billing_amount": None,
        "payment_method": None,
        "billing_frequency": None,
        "invoice_date": None,
        "conversation_history": history_str,
        "task": "Help user with their query",
        "final_answer": "",
        "timestamp": datetime.datetime.now().isoformat(),
    }

    with st.spinner("Processing query..."):
        final_state = {}
        try:
            # Invoke LangGraph
            final_state = app.invoke(initial_state)
            final_answer = final_state.get("final_answer", "")
            
            if not final_answer:
                # Fallback if no final answer is provided but another state might have response
                final_answer = "I'm sorry, I couldn't process your request thoroughly. Please verify the backend output."
                
        except Exception as e:
            final_answer = f"**Error encountered:** {str(e)}"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(final_answer)
        
        # Optionally display escalating or next agent
        requires_escalation = final_state.get("requires_human_escalation", False)
        if requires_escalation:
            st.warning(f"⚠️ Escalation Required: {final_state.get('escalation_reason', 'Unknown reason')}")
        
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
