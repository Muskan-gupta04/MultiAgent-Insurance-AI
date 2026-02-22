"""Shared setup and tracing utilities extracted from notebook."""
import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import time
from dotenv import load_dotenv
import openai
import os
from datasets import load_dataset
import chromadb
from datetime import datetime, timedelta
import random
import sqlite3
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.trace import get_current_span
from phoenix.otel import register


# ✅ Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")


from functools import wraps
import time
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.trace import get_current_span
from phoenix.otel import register

tracer_provider = register(
    project_name="insurance_multi-agent-system",
    endpoint=phoenix_endpoint,
    auto_instrument=True
)
tracer = tracer_provider.get_tracer(__name__)



# --- Decorator ---
def trace_agent(func):
    """
    Decorator to wrap multi-agent functions in a Phoenix span with metadata.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        state = args[0] if args else {}
        agent_name = func.__name__

        with tracer.start_as_current_span(agent_name) as span:
            # --- Add standard metadata ---
            span.set_attribute("agent.name", agent_name)
            span.set_attribute("agent.type", agent_name.replace("_node", ""))
            span.set_attribute("user.id", state.get("customer_id", "unknown"))
            span.set_attribute("policy.number", state.get("policy_number", "unknown"))
            span.set_attribute("claim.id", state.get("claim_id", "unknown"))
            span.set_attribute("task", state.get("task", "none"))
            span.set_attribute("timestamp", state.get("timestamp", "n/a"))

            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                # --- Record execution metadata ---
                duration = time.time() - start_time
                span.set_attribute("execution.duration_sec", duration)
                if isinstance(result, dict):
                    span.set_attribute("result.keys", list(result.keys()))
                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    return wrapper