from .data_faq import ingest_faqs, load_faq_dataframe
from .db import setup_insurance_database
from .graph import build_app
from . import resources
from .resources import init_resources
from .sample_data import generate_sample_data
from .tools import set_db_path


def init_all(setup_db: bool = False, ingest_faq: bool = False) -> dict:
    config = init_resources()
    set_db_path(config["db_path"])

    if setup_db:
        sample_data = generate_sample_data()
        setup_insurance_database(sample_data, config["db_path"])

    if ingest_faq:
        df = load_faq_dataframe()
        ingest_faqs(resources.collection, df)

    return config


def run_test_query(query: str, app=None):
    if app is None:
        app = build_app()

    initial_state = {
        "n_iteration": 0,
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
        "final_answer": "",
    }

    print(f"\n{'='*50}")
    print(f"QUERY: {query}")
    print(f"\n{'='*50}")

    final_state = app.invoke(initial_state)

    print("\n---FINAL RESPONSE---")
    final_answer = final_state.get("final_answer", "No final answer generated.")
    print(final_answer)

    return final_state


def build_app_with_init():
    config = init_resources()
    set_db_path(config["db_path"])
    return build_app()
