import streamlit as st
from pathlib import Path

from src import ingest, query


st.set_page_config(page_title="Document QA Bot", layout="wide")

st.title("Document QA Bot — RAG demo")

with st.sidebar:
    st.header("Ingest documents")
    uploaded = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"], accept_multiple_files=True)
    if st.button("Ingest uploaded files") and uploaded:
        n_total = 0
        for f in uploaded:
            # Save to data/ and ingest
            dest = Path("data") / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as out:
                out.write(f.getbuffer())
            n = ingest.ingest_file(str(dest))
            n_total += n
        st.success(f"Ingested {n_total} chunks from {len(uploaded)} file(s)")

st.header("Ask a question")
q = st.text_input("Question")
if st.button("Run") and q.strip():
    with st.spinner("Searching and generating answer..."):
        result = query.query(q)
    st.subheader("Answer")
    st.write(result.get("answer"))

    st.subheader("Top sources")
    for i, hit in enumerate(result.get("hits", []), start=1):
        st.markdown(f"**{i}. Source:** {hit['metadata'].get('source')}")
        st.write(hit["document"]) 
