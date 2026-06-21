from typing import List, Dict, Any

import chromadb

from google import genai
from google.genai import types

try:
    from src import config as cfg
except ImportError:
    import config as cfg

CHROMA_DIR = cfg.CHROMA_DIR
CHROMA_COLLECTION = cfg.CHROMA_COLLECTION
GOOGLE_API_KEY = cfg.GOOGLE_API_KEY
logger = cfg.logger


def _get_chroma_client(persist_directory: str = CHROMA_DIR):
    return chromadb.PersistentClient(path=persist_directory)


_genai_client = None


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=cfg.require_google_api_key())
    return _genai_client


def _generate_query_embedding(text: str) -> List[float]:
    client = _get_genai_client()
    resp = client.models.embed_content(model=cfg.EMBEDDING_MODEL, contents=text)
    try:
        return resp.embeddings[0].values  # type: ignore[attr-defined]
    except Exception:
        try:
            return resp["embeddings"][0]["values"]
        except Exception:
            logger.error("Unexpected embedding response: %s", resp)
            raise


def _call_gemini_flash(system_prompt: str, user_prompt: str, model: str = "gemini-2.5-flash") -> str:
    client = _get_genai_client()
    try:
        resp = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
    except Exception:
        logger.exception("Gemini generation failed")
        raise

    # Parse response text from common shapes
    try:
        return resp.text  # type: ignore[attr-defined]
    except Exception:
        try:
            return resp["text"]
        except Exception:
            try:
                return resp["output"][0]["content"][0]["text"]
            except Exception:
                logger.error("Unexpected generation response: %s", resp)
                raise


def query(question: str, collection_name: str = CHROMA_COLLECTION, k: int = 3) -> Dict[str, Any]:
    """Answer `question` using a strict RAG flow with ChromaDB + Gemini Flash.

    Steps:
    1. Embed the question and search ChromaDB for top-k similar chunks.
    2. Extract documents and metadata and format context blocks.
    3. Build a strict system prompt forbidding hallucination and requiring use of context only.
    4. Call the configured Gemini generation model to generate the answer.

    Returns a dict with keys: `answer` (string), `citations` (list), `raw_context` (string).
    """
    # 1) prepare client and embed question
    client = _get_chroma_client(CHROMA_DIR)
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        logger.exception("Failed to open collection '%s'", collection_name)
        collection = client.get_or_create_collection(collection_name)

    q_emb = _generate_query_embedding(question)
    results = collection.query(query_embeddings=[q_emb], n_results=max(k, 10), include=["documents", "metadatas", "distances"])

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Build list of candidate hits with optional distance filtering
    candidates = []
    for idx, (d, m) in enumerate(zip(docs, metas)):
        dist = None
        try:
            dist = distances[idx]
        except Exception:
            dist = None
        candidates.append({"document": d, "metadata": m, "distance": dist})

    # Filter by similarity/distance threshold when possible
    filtered = []
    for c in candidates:
        dval = c.get("distance")
        if dval is None:
            filtered.append(c)
            continue
        try:
            if dval <= cfg.MAX_DISTANCE:
                filtered.append(c)
        except Exception:
            filtered.append(c)

    if not filtered:
        # fallback to top-k if filtering removed all
        filtered = candidates[:k]

    # Deduplicate by document text
    seen = set()
    hits = []
    citations = []
    context_blocks = []
    for c in filtered:
        text = (c.get("document") or "")
        key = text.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)

        m = c.get("metadata", {})
        dist = c.get("distance")
        source = m.get("source") or m.get("file") or m.get("filename") or "unknown"
        page = m.get("page") or m.get("chunk") or m.get("page_num") or "1"
        chunk_range = m.get("chunk_range") or m.get("chunk") or "0-0"

        hits.append({"document": text, "metadata": m, "distance": dist})
        citations.append({"source": source, "page": page, "chunk_range": chunk_range, "distance": dist})

        block_header = f"[Source:{source} Page:{page}]"
        context_blocks.append(f"{block_header}\n{text}")

    raw_context = "\n\n".join(context_blocks)

    # Build numbered context blocks (limit to k blocks)
    limited_blocks = context_blocks[:k]
    numbered_blocks = []
    numbered_citations = []
    for idx, (blk, cit) in enumerate(zip(limited_blocks, citations), start=1):
        numbered_blocks.append(f"[{idx}] {blk}")
        numbered_citations.append({"index": idx, **cit})

    raw_context = "\n\n".join(numbered_blocks)

    # 4) strict system prompt (from config) with an extra instruction to use numeric citations
    system_prompt = cfg.SYSTEM_PROMPT + (
        "\nWhen you refer to a source, use numeric citation markers like [1], [2].\n"
        "At the end of your answer, do not invent any additional sources.\n"
    )

    user_prompt = f"Question: {question}\n\nContext:\n{raw_context}\n\nAnswer using only the context above and include citation markers where appropriate."

    # 5) Call Gemini generation model
    try:
        answer_text = _call_gemini_flash(system_prompt=system_prompt, user_prompt=user_prompt, model=cfg.GENERATION_MODEL)
    except Exception:
        logger.exception("Generation failed; returning a safe fallback message")
        answer_text = "I cannot find the answer in the provided documents."

    return {"answer": answer_text, "citations": numbered_citations, "raw_context": raw_context}
