import logging

import chromadb
from openai import OpenAI

from .config import load_config


CONFIG = {}
client = None
chroma_client = None
collection = None
logger = logging.getLogger("mas")


def init_resources() -> dict:
    global CONFIG, client, chroma_client, collection, logger

    CONFIG = load_config()

    client = OpenAI(
        api_key=CONFIG["groq_api_key"],
        base_url="https://api.groq.com/openai/v1"
    )

    chroma_client = chromadb.PersistentClient(path=CONFIG["chroma_path"])
    collection = chroma_client.get_or_create_collection(
        name=CONFIG["chroma_collection"]
    )

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
