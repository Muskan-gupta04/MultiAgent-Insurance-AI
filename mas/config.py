import os
from dotenv import load_dotenv


def load_config() -> dict:
    load_dotenv()
    return {
        "groq_api_key": os.getenv("GROQ_API_KEY"),
        "chroma_path": os.getenv("CHROMA_PATH", "./chroma_db"),
        "chroma_collection": os.getenv("CHROMA_COLLECTION", "insurance_FAQ_collection"),
        "db_path": os.getenv("INSURANCE_DB_PATH", "insurance_support.db"),
    }
