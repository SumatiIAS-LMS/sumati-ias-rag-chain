import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

LLM_MODEL = "gpt-5-mini"

VECTOR_DB_DIR = "vectordb"

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://localhost:27017/"
)

MONGODB_DATABASE = "sumatiias"

MONGODB_COLLECTION = "lecture_chunks"

MONGODB_WEB_EDITORIAL_COLLECTION = "webeditorials"

MONGODB_WEB_HOT_TOPIC_COLLECTION = "webhottopics"

MONGODB_WEB_CURRENT_AFFAIRS_COLLECTION = "currentaffairs"

MONGODB_DATA_CHUNKS_COLLECTION = "sumati_ias_data_chunks"

# Optional public base for files stored as object keys, for example
# https://assets.example.com . When unset, keys are signed for download.
PUBLIC_ASSET_BASE_URL = os.getenv("PUBLIC_ASSET_BASE_URL", "").strip()

CF_ACCESS_KEY_ID = os.getenv("CF_ACCESS_KEY_ID")
CF_SECRET_ACCESS_KEY = os.getenv("CF_SECRET_ACCESS_KEY")
CF_S3_API_ENDPOINT = os.getenv("CF_S3_API_ENDPOINT", "").strip()
CF_S3_BUCKET_NAME = os.getenv("CF_S3_BUCKET_NAME", "").strip()

VECTOR_INDEX_NAME = "lecture_vector_index"

SUMATI_IAS_DATA_CHUNKS_VECTOR_INDEX_NAME = "sumati_ias_data_chunks_vector_index"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

EMBEDDING_DIMENSIONS = 384

CONTEXT_EXPIREY = 60 * 60 * 24 * 7 # in seconds

WEB_CONTEXT_EXPIRY = 60 * 60 * 24 # in seconds

WEB_DAILY_MESSAGE_LIMIT = 20