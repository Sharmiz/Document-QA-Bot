import streamlit as st
from pathlib import Path
import logging
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import ingest, query

# Prefer the ingest module's configured logger when available.
logger = getattr(ingest, "logger", logging.getLogger(__name__))


st.set_page_config(page_title="Document Q&A Bot", layout="wide")

st.title("Document Q&A Bot")


def sidebar_ingest():
    """Sidebar upload + ingest controls.

    Keeps ingestion UI separate from the main QA interface. Uploaded files
    are saved to `data/` and then ingested using `ingest.ingest_file`.
    """
    with st.sidebar:
        st.header("Ingest documents")
        uploaded = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)

        last_result = st.session_state.get("last_ingest_result")
        if last_result:
            message_type, message = last_result
            if message_type == "success":
                st.success(message)
            elif message_type == "warning":
                st.warning(message)
            else:
                st.error(message)

        if st.button("Ingest uploaded files"):
            if not uploaded:
                st.session_state["last_ingest_result"] = ("warning", "Please upload at least one file before ingesting.")
                st.rerun()

            data_dir = Path("data")
            data_dir.mkdir(parents=True, exist_ok=True)

            n_total = 0
            successful_files = 0
            processed_names = set()
            failure_messages = []
            zero_chunk_files = []

            with st.spinner("Ingesting uploaded documents..."):
                for f in uploaded:
                    # Skip duplicate filenames in the same upload session
                    if f.name in processed_names:
                        st.warning(f"Skipping duplicate upload: {f.name}")
                        continue

                    dest = data_dir / f.name
                    try:
                        with open(dest, "wb") as out:
                            # Use getbuffer() to write the uploaded file contents reliably
                            out.write(f.getbuffer())
                    except Exception as e:
                        failure_messages.append(f"{f.name}: {e}")
                        logger.exception("Failed to save uploaded file %s: %s", f.name, e)
                        st.error(f"Failed to save uploaded file {f.name}: {e}")
                        continue

                    # Log saved filename and size
                    try:
                        size = dest.stat().st_size
                    except Exception:
                        size = 0
                    logger.info("Saved uploaded file %s (size=%d bytes)", dest.name, size)

                    if size == 0:
                        failure_messages.append(f"{f.name}: uploaded file is empty")
                        st.error(f"Uploaded file {f.name} is empty and was skipped.")
                        try:
                            dest.unlink(missing_ok=True)
                        except Exception:
                            pass
                        continue

                    processed_names.add(f.name)

                    try:
                        n = ingest.ingest_file(str(dest))
                        n_total += n
                        if n > 0:
                            successful_files += 1
                        else:
                            zero_chunk_files.append(f.name)
                    except Exception as e:
                        failure_messages.append(f"{f.name}: {e}")
                        logger.exception("Failed to ingest %s: %s", dest, e)
                        st.error(f"Failed to ingest {f.name}: {e}")

            if n_total > 0:
                message = f"Ingested {n_total} chunks from {successful_files} file(s)."
                if failure_messages or zero_chunk_files:
                    skipped_count = len(failure_messages) + len(zero_chunk_files)
                    message += f" Skipped {skipped_count} file(s)."
                st.session_state["last_ingest_result"] = (
                    "success",
                    message,
                )
            elif failure_messages:
                st.session_state["last_ingest_result"] = (
                    "error",
                    f"No files were ingested. First error: {failure_messages[0]}",
                )
            elif processed_names:
                st.session_state["last_ingest_result"] = (
                    "warning",
                    "The upload finished, but no text chunks were ingested. The PDF may be scanned/image-only or empty.",
                )
            else:
                st.session_state["last_ingest_result"] = ("error", "No files were ingested. Please check the upload and try again.")
            st.rerun()


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
