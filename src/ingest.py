import os
import uuid
from pathlib import Path
from typing import List

from pypdf import PdfReader
from docx import Document

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_DIR, CHROMA_COLLECTION, get_embedding


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n\n".join(texts)


def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n\n".join(paragraphs)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Simple sliding-window chunking by characters."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def ingest_file(path: str, collection_name: str = CHROMA_COLLECTION):
    """Extract text, chunk it, embed, and store in ChromaDB."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in [".pdf"]:
        text = extract_text_from_pdf(path)
    elif p.suffix.lower() in [".docx"]:
        text = extract_text_from_docx(path)
    else:
        # treat as plain text
        text = p.read_text(encoding="utf-8")

    chunks = [c for c in chunk_text(text) if c.strip()]

    # Initialise Chroma client (persistent directory)
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
    collection = client.get_or_create_collection(collection_name)

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for i, chunk in enumerate(chunks):
        cid = f"{p.name}-{i}-{uuid.uuid4()}"
        ids.append(cid)
        documents.append(chunk)
        metadatas.append({"source": str(p), "chunk": i})
        # get_embedding is a stub in `src.config` — implement it using Gemini embeddings
        emb = get_embedding(chunk)
        embeddings.append(emb)

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    # persist is automatic for the chosen chroma backend when using a persistent directory
    return len(ids)
