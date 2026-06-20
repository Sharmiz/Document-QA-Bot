from dotenv import load_dotenv
import os
import logging

load_dotenv()

# General paths and collection names
CHROMA_DIR = os.getenv("CHROMA_DIR", "db/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "document_knowledge_base")

# Google API key for Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ingestion chunking defaults
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Similarity / distance filtering
# If Chroma returns distances, only accept results with distance <= MAX_DISTANCE.
# Default tuned for cosine distance-like outputs; adjust if your backend uses other metrics.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.5"))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger("document_qa")

# Prompt engineering defaults
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the provided context.\n"
    "Rules:\n"
    "- Use ONLY the provided context blocks. Do not use external knowledge.\n"
    "- If the information is not present, reply exactly: 'I cannot find the answer in the provided documents.'\n"
    "- Keep answers concise and include inline citation markers like [1], [2] that refer to the provided sources.\n"
)

