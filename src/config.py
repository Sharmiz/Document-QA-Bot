from dotenv import load_dotenv
import os

load_dotenv()

# Where Chroma will persist its DB files (default under project `db/`)
CHROMA_DIR = os.getenv("CHROMA_DIR", "db/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "document_qa")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_embedding(text: str) -> list:
    """Return an embedding vector for `text`.

    TODO: implement this using Google Gemini / Generative AI embeddings.
    The project intentionally leaves this as a stub so you can plug
    in the correct model name and API usage for your account.
    """
    raise NotImplementedError("Implement get_embedding(text) using Google Gemini embeddings and set GOOGLE_API_KEY in .env")

def generate_answer(prompt: str, context: str) -> str:
    """Generate an answer using Google Gemini.

    TODO: implement the call to the Gemini text generation API and
    return the model's answer string.
    """
    raise NotImplementedError("Implement generate_answer(prompt, context) using Google Gemini text generation API")
