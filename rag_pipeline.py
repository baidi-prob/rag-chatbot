"""
rag_pipeline.py
================
This file contains the core RAG (Retrieval-Augmented Generation) logic.
Read the comments carefully -- you should be able to explain every step
of this file in an interview.

The 3 stages of RAG:
1. INGEST  -> load documents, split into chunks, embed each chunk, store in a vector database
2. RETRIEVE -> given a user question, find the most similar chunks in the vector database
3. GENERATE -> feed the retrieved chunks + question to an LLM, so it answers using YOUR data
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


def load_documents(file_path: str):
    """
    STAGE 1a: Load a document from disk.
    - PDFs use PyPDFLoader, plain text files use TextLoader.
    - Returns a list of LangChain "Document" objects (text + metadata).
    """
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def split_into_chunks(documents, chunk_size: int = 800, chunk_overlap: int = 100):
    """
    STAGE 1b: Split documents into smaller chunks.

    WHY CHUNK AT ALL?
    - LLMs have a limited context window, so we can't dump an entire book in.
    - Smaller, focused chunks retrieve more precisely: if you ask a narrow
      question, you want the 2-3 sentences that answer it, not a whole chapter.

    WHY chunk_overlap?
    - If a sentence explaining an answer gets cut exactly at a chunk boundary,
      overlap ensures that context isn't lost between adjacent chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]  # tries paragraph breaks first, then sentences, then words
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks, persist_path: str = "faiss_index"):
    """
    STAGE 1c: Convert chunks into embeddings (vectors) and store them in FAISS.

    WHAT IS AN EMBEDDING?
    - A vector (list of numbers) that represents the MEANING of a piece of text.
    - Texts with similar meaning end up with vectors that are close together
      in this high-dimensional space (measured via cosine similarity).

    WHY HuggingFaceEmbeddings (all-MiniLM-L6-v2)?
    - It's free and runs locally -- no API key or cost needed for this step.
    - It's small (~80MB) and fast, while still being accurate enough for
      a portfolio-scale project.

    WHAT IS FAISS?
    - Facebook AI Similarity Search -- a library for fast nearest-neighbor
      search over vectors. Given a query vector, it quickly finds the
      chunks whose vectors are closest (i.e. most semantically similar).
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(persist_path)
    return vector_store


def load_vector_store(persist_path: str = "faiss_index"):
    """Load a previously built FAISS index back from disk."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)


def build_qa_chain(vector_store, groq_api_key: str, k: int = 3):
    """
    STAGE 2 + 3 combined: build the retrieval + generation chain.

    STAGE 2 (RETRIEVE): vector_store.as_retriever(k=3) means "when a question
    comes in, fetch the 3 most similar chunks from FAISS".

    STAGE 3 (GENERATE): Those 3 chunks get inserted into a prompt template
    along with the user's question, and sent to the LLM (Llama 3.1 8B via
    Groq's free API here). The LLM is instructed to answer ONLY using the
    provided context -- this is what makes it "grounded" rather than
    hallucinating.

    WHY GROQ?
    - Groq offers a genuinely free API tier (no credit card needed) and is
      extremely fast, since it runs models on custom hardware built for
      inference speed rather than general-purpose GPUs.
    """
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,  # 0 = deterministic, factual answers (not creative)
        api_key=groq_api_key
    )

    # This prompt is the key to reducing hallucination: it explicitly tells
    # the model to say "I don't know" rather than making something up.
    prompt_template = """Use the following pieces of context to answer the question at the end.
If the answer isn't in the context, just say you don't know -- do NOT make up an answer.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" = simply stuff all retrieved chunks into one prompt
        retriever=vector_store.as_retriever(search_kwargs={"k": k}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True  # lets us show which chunks the answer came from
    )
    return qa_chain
