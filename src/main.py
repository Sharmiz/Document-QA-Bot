import streamlit as st
from pathlib import Path

from src import ingest, query


st.set_page_config(page_title="Document Q&A Bot", layout="wide")

st.title("Document Q&A Bot")


def sidebar_ingest():
    """Sidebar upload + ingest controls.

    Keeps ingestion UI separate from the main QA interface. Uploaded files
    are saved to `data/` and then ingested using `ingest.ingest_file`.
    """
    with st.sidebar:
        st.header("Ingest documents")
        uploaded = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"], accept_multiple_files=True)
        if st.button("Ingest uploaded files") and uploaded:
            n_total = 0
            for f in uploaded:
                dest = Path("data") / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as out:
                    out.write(f.getbuffer())
                try:
                    n = ingest.ingest_file(str(dest))
                    n_total += n
                except Exception as e:
                    st.error(f"Failed to ingest {f.name}: {e}")
            st.success(f"Ingested {n_total} chunks from {len(uploaded)} file(s)")


def main_ui():
    """Main QA UI. Presents an input box and Submit button, then displays
    the returned answer and sources. All business logic is delegated to
    `src.query.query` to keep UI and logic separate.
    """
    st.header("Ask a question")
    question = st.text_input("Ask your question")
    submit = st.button("Submit")

    if submit and question.strip():
        # Call business logic in a try/except to handle errors gracefully.
        try:
            with st.spinner("Searching documents and generating an answer..."):
                result = query.query(question)

            answer = result.get("answer")
            citations = result.get("citations", [])
            raw_context = result.get("raw_context", "")

            st.subheader("Answer")
            if answer:
                st.write(answer)
            else:
                st.write("No answer returned.")

            st.subheader("Sources")
            if citations:
                for c in citations:
                    src = c.get("source", "unknown")
                    page = c.get("page", "1")
                    st.write(f"{src} Page {page}")
            else:
                st.write("No sources found.")

            # Optionally show raw retrieved context for debugging / transparency
            with st.expander("Raw retrieved context"):
                st.text(raw_context)

        except Exception as e:
            st.error(f"An error occurred while answering the question: {e}")


if __name__ == "__main__":
    sidebar_ingest()
    main_ui()
