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

VECTOR_INDEX_NAME = "lecture_vector_index"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

EMBEDDING_DIMENSIONS = 384

CONTEXT_EXPIREY = 60 * 60 * 24 * 7 # in seconds