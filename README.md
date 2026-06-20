# Document-QA-Bot

Document-QA-Bot is a starter project that demonstrates how to build a simple Document Question Answering (RAG) system using Python, Google Gemini (Generative AI), and ChromaDB for embeddings and vector search. This repository includes a minimal ingestion pipeline, a query interface, and a Streamlit demo app.

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

5. Run the demo app:

	```bash
	streamlit run src/main.py
	```

Notes for beginners

- This scaffold contains working file readers for PDF and DOCX, chunking utilities, and a basic ChromaDB integration. Embeddings and text-generation calls using Google Gemini are provided as clear TODOs/stubs — you must insert your API usage details (the Google Generative AI SDK) and set your `GOOGLE_API_KEY` in `.env`.
- The `src/` code is intentionally simple and well-commented so you can extend it to your needs.

If you want, I can:

- Implement the Gemini embedding + generation calls once you provide which Gemini model (and confirm access).
- Add example unit tests or GitHub Actions for CI.

