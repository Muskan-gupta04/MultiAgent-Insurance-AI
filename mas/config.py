import os
from dotenv import load_dotenv


def load_config() -> dict:
    load_dotenv()
    return {
        "openai_api_key": os.getenv("OPEN_AI_KEY"),
        "phoenix_endpoint": os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
        "chroma_path": os.getenv("CHROMA_PATH", "./chroma_db"),
        "chroma_collection": os.getenv("CHROMA_COLLECTION", "insurance_FAQ_collection"),
        "db_path": os.getenv("INSURANCE_DB_PATH", "insurance_support.db"),
    }
