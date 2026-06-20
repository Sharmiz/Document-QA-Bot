# Document-QA-Bot

Document-QA-Bot is a beginner-friendly RAG (Retrieval-Augmented Generation) starter project using Python, Google Gemini (Generative AI), and ChromaDB for embeddings and vector search. It provides:

- Document ingestion from PDF/DOCX/TXT
- Chunking with overlap and deduplication
- Embedding generation (Gemini)
- Persistent vector store using ChromaDB
- A Streamlit demo app for asking questions and viewing sources

Project structure

document-qa-bot/
├── .env
├── .gitignore
├── requirements.txt
├── README.md
├── data/
├── db/
└── src/
	 ├── __init__.py
	 ├── config.py
	 ├── ingest.py
	 ├── query.py
	 └── main.py

Getting started (Windows)

1. Install Python 3.11+.
2. Create a virtual environment and activate it:

	```bash
	python -m venv .venv
	.venv\Scripts\activate    # cmd.exe
	# or
	.venv\Scripts\Activate.ps1  # PowerShell
	```

3. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

4. Create a `.env` file (copy the provided `.env` and add your API key):

	- `GOOGLE_API_KEY` — your Google Generative AI / Gemini API key
	- `CHROMA_DIR` — directory where ChromaDB will persist its database (default: `db/chroma`)
	- `CHROMA_COLLECTION` — Chroma collection name (default: `document_knowledge_base`)
	- `CHUNK_SIZE` and `CHUNK_OVERLAP` — control chunking (defaults: 1000, 200)
	- `MAX_DISTANCE` — numeric threshold used to filter search results by distance (defaults: 0.5)

5. Run the demo app:

	```bash
	streamlit run src/main.py
	```

Notes for beginners

- The code is organized so the UI (`src/main.py`) only calls business logic in `src/ingest.py` and `src/query.py`. This makes it easier to test and modify behavior.
- `src/config.py` centralizes configuration, logging, and prompt defaults. Edit `.env` to change behavior without modifying code.
- The project includes safety and quality improvements:
	- Empty documents are skipped during loading.
	- Duplicate chunks are removed before embedding to reduce storage and redundant results.
	- Search results can be filtered by distance (`MAX_DISTANCE`) to avoid low-quality matches.
	- The system prompt enforces strict use of provided context and requests numeric citations.

Next steps you might want:

- Tune `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `MAX_DISTANCE` in `.env` for your documents.
- Add unit tests that mock Gemini and Chroma to verify logic without network calls.
- Enable CI (GitHub Actions) to run linters and tests.

If you want, I can implement further improvements such as automated tests, CI, or example ingestion scripts.

