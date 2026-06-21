import os
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def _load_env_file(path: Path) -> dict[str, str]:
    """Load simple KEY=value pairs without warning on display-name labels."""
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip("\"'")
        values[name] = value
        if name.replace("_", "").isalnum() and " " not in name:
            os.environ.setdefault(name, value)
    return values


ENV_VALUES = _load_env_file(ENV_PATH)

# General paths and collection names
CHROMA_DIR = os.getenv("CHROMA_DIR", "db/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "document_knowledge_base")

# Google API key for Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")


def _looks_like_openai_key(value: str) -> bool:
    return value.startswith(("sk-", "sk-proj-", "sk-proj_", "sk-pro-"))


def require_google_api_key() -> str:
    """Return a Gemini API key or raise a helpful error."""
    api_key = GOOGLE_API_KEY
    fallback_key = ENV_VALUES.get("Gemini API Key")
    if fallback_key and (not api_key or _looks_like_openai_key(api_key)):
        api_key = fallback_key

    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set in .env")
    if any(ch.isspace() for ch in api_key):
        raise RuntimeError("GOOGLE_API_KEY contains whitespace. Paste the key as one continuous value.")
    if _looks_like_openai_key(api_key):
        raise RuntimeError(
            "GOOGLE_API_KEY contains an OpenAI-style key. Replace it with your Google AI Studio Gemini API key."
        )
    if len(api_key) < 20:
        raise RuntimeError(
            "GOOGLE_API_KEY is too short to be a valid Gemini API key."
        )
    return api_key

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

