"""
app.py
======
Streamlit UI: lets a user upload a document, then ask questions about it.
Run with: streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from rag_pipeline import load_documents, split_into_chunks, build_vector_store, build_qa_chain

st.set_page_config(page_title="Chat With Your Document", page_icon="📄")
st.title("📄 Chat With Your Document (RAG Demo)")
st.caption("Upload a PDF or text file, then ask questions grounded in its content.")

# --- Sidebar: API key input ---
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Groq API Key (free)", type="password")
    st.markdown("[Get a free API key](https://console.groq.com/keys)")

# --- Session state: keep the QA chain alive across reruns ---
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- File upload + ingestion ---
uploaded_file = st.file_uploader("Upload a PDF or .txt file", type=["pdf", "txt"])

if uploaded_file and groq_api_key:
    if st.button("Process Document"):
        with st.spinner("Reading, chunking, and embedding your document..."):
            # Save uploaded file to a temp path so our loaders can read it
            suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            documents = load_documents(tmp_path)
            chunks = split_into_chunks(documents)
            vector_store = build_vector_store(chunks)
            st.session_state.qa_chain = build_qa_chain(vector_store, groq_api_key)
            os.unlink(tmp_path)  # clean up temp file

        st.success(f"Done! Split into {len(chunks)} chunks and indexed. Ask a question below.")

elif uploaded_file and not groq_api_key:
    st.warning("Enter your free Groq API key in the sidebar first.")

# --- Chat interface ---
if st.session_state.qa_chain:
    st.divider()
    question = st.text_input("Ask a question about your document:")

    if question:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain.invoke({"query": question})
            answer = result["result"]
            sources = result["source_documents"]

        st.session_state.chat_history.append((question, answer, sources))

    # Show conversation history, most recent first
    for q, a, sources in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")
        with st.expander("View source chunks used"):
            for i, doc in enumerate(sources):
                st.markdown(f"**Chunk {i+1}:**")
                st.text(doc.page_content[:400])
        st.divider()
