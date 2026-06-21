import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader
from docx import Document

import chromadb

from google import genai
from tqdm import tqdm

try:
    from src import config as cfg
except ImportError:
    import config as cfg

CHROMA_DIR = cfg.CHROMA_DIR
CHROMA_COLLECTION = cfg.CHROMA_COLLECTION
GOOGLE_API_KEY = cfg.GOOGLE_API_KEY

logger = cfg.logger


def _get_chroma_client(persist_directory: str = CHROMA_DIR):
    """Return a persistent Chroma client using the current Chroma API."""
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_directory)


def extract_pdf_pages(path: str) -> List[Dict[str, Any]]:
    """Extract text from each page of a PDF and return a list of page documents.

    Each item has the form:
      {"text": "page text...", "metadata": {"source": "file.pdf", "page": 1}}

    Errors reading a file or a page are logged and that page/file is skipped.
    """
    p = Path(path)
    pages: List[Dict[str, Any]] = []
    try:
        reader = PdfReader(path)
    except Exception as e:
        logger.exception(f"Failed to open PDF {path}: {e}")
        return pages

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.exception(f"Failed to extract text from page {i+1} of {path}: {e}")
            text = ""

        pages.append({"text": text, "metadata": {"source": p.name, "page": i + 1}})

    logger.info(f"Extracted {len(pages)} page(s) from {p.name}")
    return pages


def extract_docx(path: str) -> List[Dict[str, Any]]:
    """Extract text from a DOCX file and return a single document entry.

    DOCX files don't have page boundaries in the same way PDFs do. This function
    concatenates paragraph text and returns a single item with `page` = 1.
    """
    p = Path(path)
    try:
        doc = Document(path)
    except Exception as e:
        logger.exception(f"Failed to open DOCX {path}: {e}")
        return []

    try:
        paragraphs = [para.text for para in doc.paragraphs if para.text and para.text.strip()]
        text = "\n\n".join(paragraphs)
    except Exception as e:
        logger.exception(f"Failed to extract text from DOCX {path}: {e}")
        text = ""

    logger.info(f"Extracted DOCX {p.name} (chars={len(text)})")
    return [{"text": text, "metadata": {"source": p.name, "page": 1}}]


def load_documents(data_dir: str = "data") -> List[Dict[str, Any]]:
    """Scan `data_dir` for supported files and return a flat list of documents.

    Supported files: PDF (.pdf), Word (.docx), and plain text (.txt).

    Returns a list of dicts like:
      [ {"text":"...","metadata":{"source":"file.pdf","page":1}}, ... ]
    """
    base = Path(data_dir)
    if not base.exists():
        logger.warning("Data directory does not exist: %s", data_dir)
        return []

    documents: List[Dict[str, Any]] = []

    for file in sorted(base.rglob("*")):
        if not file.is_file():
            continue

        suffix = file.suffix.lower()
        try:
            if suffix == ".pdf":
                documents.extend(extract_pdf_pages(str(file)))
            elif suffix == ".docx":
                documents.extend(extract_docx(str(file)))
            elif suffix == ".txt":
                try:
                    text = file.read_text(encoding="utf-8")
                except Exception:
                    text = file.read_text(encoding="latin-1")
                documents.append({"text": text, "metadata": {"source": file.name, "page": 1}})
            else:
                logger.debug("Skipping unsupported file type: %s", file)
        except Exception as e:
            logger.exception("Error processing file %s: %s", file, e)

    # Remove empty documents (safety check)
    non_empty = [d for d in documents if d.get("text") and d.get("text").strip()]
    if len(non_empty) < len(documents):
        logger.info("Dropped %d empty document items", len(documents) - len(non_empty))

    logger.info("Loaded %d document items from %s", len(non_empty), data_dir)
    return non_empty


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Simple sliding-window chunking by characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be 0 or greater")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start = end - overlap
    return chunks


def chunk_extracted_pages(pages: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """Chunk a list of extracted pages while preserving metadata.

    Input `pages` should be a list of items like:
      {"text": "...", "metadata": {"source": "file.pdf", "page": 3}}

    Output is a flat list of chunks with the shape:
      {"text": "...", "metadata": {"source": "file.pdf", "page": 3, "chunk_range": "0-1000"}}

    Parameters:
    - chunk_size: target size in characters for each chunk.
    - chunk_overlap: how many characters to overlap between consecutive chunks.

    Overlap explanation: with `chunk_size=1000` and `chunk_overlap=200`, the
    second chunk will start at character 800 (1000 - 200) of the original text,
    so the final 200 characters of the first chunk appear at the beginning of
    the second chunk. Overlap helps retain context across chunk boundaries,
    which improves retrieval quality for queries that span chunk edges.
    """
    if not isinstance(pages, list):
        logger.error("chunk_extracted_pages expects a list of pages")
        return []

    result: List[Dict[str, Any]] = []

    for item in pages:
        try:
            text = item.get("text", "")
            meta = item.get("metadata", {})
            source = meta.get("source", "unknown")
            page_no = meta.get("page", 1)

            if not text or not text.strip():
                logger.debug("Skipping empty text for %s page %s", source, page_no)
                continue

            text_len = len(text)
            start = 0
            chunk_index = 0
            while start < text_len:
                end = min(start + chunk_size, text_len)
                chunk_text = text[start:end]
                chunk_meta = {
                    "source": source,
                    "page": page_no,
                    "chunk_range": f"{start}-{end}"
                }
                result.append({"text": chunk_text, "metadata": chunk_meta})

                if end >= text_len:
                    break
                start = end - chunk_overlap
                if start < 0:
                    start = 0
                chunk_index += 1

            logger.info("Chunked %s page %s into %d chunk(s)", source, page_no, chunk_index + 1)
        except Exception as e:
            logger.exception("Failed to chunk page item: %s", e)
            continue

    logger.info("Produced %d total chunks from %d pages", len(result), len(pages))
    return result


def ingest_file(path: str, collection_name: str = CHROMA_COLLECTION):
    """Extract text, chunk it, embed, and store in ChromaDB.

    This keeps backward compatibility with the original simple ingest flow.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".pdf":
        # join all pages into one long text for chunking
        pages = extract_pdf_pages(path)
        text = "\n\n".join([pg.get("text", "") for pg in pages])
    elif p.suffix.lower() == ".docx":
        docs = extract_docx(path)
        text = docs[0].get("text", "") if docs else ""
    else:
        text = p.read_text(encoding="utf-8")

    chunks = [c for c in chunk_text(text, chunk_size=cfg.CHUNK_SIZE, overlap=cfg.CHUNK_OVERLAP) if c.strip()]
    if not chunks:
        logger.warning("No text chunks produced from %s", path)
        return 0

    # Initialise Chroma client (persistent directory)
    client = _get_chroma_client(CHROMA_DIR)
    collection = client.get_or_create_collection(collection_name)

    ids = []
    documents = []
    metadatas = []

    embeddings = _generate_embeddings(chunks)

    for i, chunk in enumerate(chunks):
        cid = f"{p.name}-{i}-{uuid.uuid4()}"
        ids.append(cid)
        documents.append(chunk)
        metadatas.append({"source": str(p), "chunk": i})

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(ids)


_genai_client = None


def _get_genai_client():
    """Return a configured Google Gen AI client."""
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=cfg.require_google_api_key())
    return _genai_client


def _generate_embedding(text: str) -> List[float]:
    """Generate an embedding vector for `text` using the configured Gemini embedding model.

    Returns a list of floats. Defensive parsing handles a few SDK response shapes.
    """
    return _generate_embeddings([text])[0]


def _generate_embeddings(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Generate embeddings for multiple texts using batched Gemini requests."""
    if not texts:
        return []

    client = _get_genai_client()
    embeddings: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        try:
            resp = client.models.embed_content(model=cfg.EMBEDDING_MODEL, contents=batch)
        except Exception as e:
            logger.exception("Embedding request failed for batch starting at %d: %s", start, e)
            raise

        batch_embeddings = _parse_embedding_response(resp)
        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} embeddings, got {len(batch_embeddings)} from Gemini"
            )
        embeddings.extend(batch_embeddings)

    return embeddings


def _parse_embedding_response(resp: Any) -> List[List[float]]:
    """Parse embeddings from the current SDK response and a few older shapes."""
    try:
        return [embedding.values for embedding in resp.embeddings]  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        return [item["values"] for item in resp["embeddings"]]
    except Exception:
        pass

    try:
        return [resp["embedding"]]
    except Exception:
        logger.error("Unable to parse embedding response: %s", resp)
        raise RuntimeError("Unexpected embedding response shape")


def save_to_vector_db(ids: List[str], documents: List[str], metadatas: List[dict], embeddings: List[List[float]],
                      collection_name: str = "document_knowledge_base", persist_directory: str = "db") -> None:
    """Save vectors + metadata to a persistent ChromaDB collection.

    Stores vectors under `persist_directory` (defaults to `db/`).
    """
    client = _get_chroma_client(persist_directory)

    try:
        collection = client.get_or_create_collection(collection_name)
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        logger.info("Stored %d vectors in collection '%s'", len(ids), collection_name)
    except Exception as e:
        logger.exception("Failed to save vectors to ChromaDB: %s", e)
        raise


def run_ingest(data_dir: str = "data", collection_name: str = "document_knowledge_base",
               persist_directory: str = "db", chunk_size: int | None = None, chunk_overlap: int | None = None) -> int:
    """Main ingestion flow.

    Steps:
    1. Load documents from `data_dir` using `load_documents()` (page-level items).
    2. Chunk documents using `chunk_extracted_pages()` preserving metadata.
    3. Generate embeddings for each chunk using the configured Gemini embedding model (with `tqdm`).
    4. Store ids, documents, metadatas, and embeddings in a persistent ChromaDB collection.

    Returns the number of vectors stored.
    """
    logger.info("Starting ingestion: data_dir=%s, collection=%s", data_dir, collection_name)

    # 1) Load page-level documents
    pages = load_documents(data_dir)
    if not pages:
        logger.warning("No documents found in %s", data_dir)
        return 0

    # 2) Chunk pages (use defaults from config when not provided)
    if chunk_size is None:
        chunk_size = cfg.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = cfg.CHUNK_OVERLAP

    chunks = chunk_extracted_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        logger.warning("No chunks produced from documents")
        return 0

    # 3) Generate embeddings with progress bar
    ids = []
    docs = []
    metas = []
    embs = []

    # Remove duplicate chunks by exact text match (keeps first occurrence)
    seen_texts = set()
    unique_chunks = []
    for c in chunks:
        t = c.get("text", "")
        key = t.strip()
        if not key:
            continue
        if key in seen_texts:
            continue
        seen_texts.add(key)
        unique_chunks.append(c)

    logger.info("Reduced %d chunks to %d unique chunks after deduplication", len(chunks), len(unique_chunks))

    for i, chunk in enumerate(tqdm(unique_chunks, desc="Embedding chunks")):
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        unique_id = f"{metadata.get('source','unknown')}-{metadata.get('page',1)}-{metadata.get('chunk_range','0')}-{i}"
        try:
            embedding = _generate_embedding(text)
        except Exception:
            logger.exception("Skipping chunk due to embedding error: %s", unique_id)
            continue

        ids.append(unique_id)
        docs.append(text)
        metas.append(metadata)
        embs.append(embedding)

    # 4) Save to ChromaDB
    save_to_vector_db(ids=ids, documents=docs, metadatas=metas, embeddings=embs,
                      collection_name=collection_name, persist_directory=persist_directory)

    logger.info("Ingestion complete: stored %d vectors", len(ids))
    return len(ids)
