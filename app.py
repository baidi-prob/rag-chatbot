"""
app.py
======
Streamlit UI: lets a user upload a document, then ask questions about it.
Run with: streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from rag_pipeline import (
    load_documents,
    split_into_chunks,
    build_vector_store,
    load_vector_store,
    build_qa_chain,
)

st.set_page_config(page_title="Chat With Your Document", page_icon="📄")
st.title("📄 Chat With Your Document (RAG Demo)")
st.caption("Upload a PDF or text file, then ask questions grounded in its content.")

# --- Sidebar: settings + API key ---
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Groq API Key (free)", type="password")
    st.markdown("[Get a free API key](https://console.groq.com/keys)")

    st.divider()
    st.subheader("Retrieval")
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=3000, value=800, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=100, step=50)
    retrieval_k = st.number_input("Chunks to retrieve", min_value=1, max_value=10, value=3, step=1)

    st.divider()
    st.subheader("Index")
    index_path = st.text_input("Index path", value="faiss_index")
    reuse_index = st.toggle("Reuse existing index", value=True,
                            help="Skip re-embedding if a saved index exists for this file.")

# --- Session state ---
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_filename" not in st.session_state:
    st.session_state.indexed_filename = None

# --- File upload + ingestion ---
uploaded_file = st.file_uploader("Upload a PDF or .txt file", type=["pdf", "txt"])

if uploaded_file and groq_api_key:
    if st.button("Process Document"):
        with st.spinner("Reading, chunking, and embedding your document..."):
            try:
                # Save uploaded file to a temp path so our loaders can read it
                suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Reuse a previously built index only when the same filename is uploaded
                # AND the user has toggled reuse on AND the index actually exists on disk.
                can_reuse = (
                    reuse_index
                    and st.session_state.indexed_filename == uploaded_file.name
                    and os.path.isdir(index_path)
                )

                if can_reuse:
                    vector_store = load_vector_store(persist_path=index_path)
                    with st.empty():
                        st.caption("Reusing previously built index (no re-embedding).")
                else:
                    documents = load_documents(tmp_path)
                    chunks = split_into_chunks(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    vector_store = build_vector_store(chunks, persist_path=index_path)
                    st.session_state.indexed_filename = uploaded_file.name

                st.session_state.qa_chain = build_qa_chain(vector_store, groq_api_key, k=retrieval_k)
                os.unlink(tmp_path)  # clean up temp file
            except Exception as e:
                st.error(f"Failed to process document: {e}")
                st.session_state.qa_chain = None
        if st.session_state.qa_chain:
            st.success("Done! Ask a question below.")

elif uploaded_file and not groq_api_key:
    st.warning("Enter your free Groq API key in the sidebar first.")

# --- Chat interface ---
if st.session_state.qa_chain:
    st.divider()
    question = st.text_input("Ask a question about your document:")

    if question:
        # Show the question immediately, then stream the answer
        st.session_state.chat_history.append({"question": question, "answer": "", "sources": []})

        full_answer = ""
        sources = []

        try:
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"input": question})
                sources = result.get("source_documents", [])
                full_answer = result.get("answer", "")
        except Exception as e:
            full_answer = f"Sorry, that question failed: {e}"

        st.session_state.chat_history[-1]["answer"] = full_answer
        st.session_state.chat_history[-1]["sources"] = sources

    # Show conversation history, most recent first
    for turn in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {turn['question']}")
        if turn["answer"]:
            st.markdown(f"**Assistant:** {turn['answer']}")
        if turn.get("sources"):
            with st.expander("View source chunks used"):
                for i, doc in enumerate(turn["sources"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.text(doc.page_content[:400])
        st.divider()