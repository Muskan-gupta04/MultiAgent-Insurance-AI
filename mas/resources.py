import logging

import chromadb
from openai import OpenAI

from .config import load_config
from .observability import init_tracer, make_trace_agent


CONFIG = {}
client = None
chroma_client = None
collection = None
trace_agent = None
logger = logging.getLogger("mas")


def init_resources() -> dict:
    global CONFIG, client, chroma_client, collection, trace_agent, logger

    CONFIG = load_config()

    client = OpenAI(api_key=CONFIG["openai_api_key"])

    chroma_client = chromadb.PersistentClient(path=CONFIG["chroma_path"])
    collection = chroma_client.get_or_create_collection(
        name=CONFIG["chroma_collection"]
    )

    tracer = init_tracer(CONFIG["phoenix_endpoint"])
    trace_agent = make_trace_agent(tracer)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("insurance_agent.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    return CONFIG
