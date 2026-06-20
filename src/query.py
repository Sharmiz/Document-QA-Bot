from typing import List

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_DIR, CHROMA_COLLECTION, get_embedding, generate_answer


def query(question: str, collection_name: str = CHROMA_COLLECTION, k: int = 3) -> dict:
    """Run a vector search for `question` and return top-k docs and a generated answer.

    This function uses `get_embedding` to embed the question, queries ChromaDB,
    then calls `generate_answer` (a stub) to create an answer using retrieved context.
    """
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
    collection = client.get_collection(collection_name)

    q_emb = get_embedding(question)
    results = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"])

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    for d, m in zip(docs, metas):
        hits.append({"document": d, "metadata": m})

    # Build a simple context string for the generator
    context = "\n\n".join([h["document"] for h in hits])

    # generate_answer is a stub in config.py — implement with Gemini
    answer = generate_answer(question, context)

    return {"answer": answer, "hits": hits}
