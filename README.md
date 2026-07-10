# Chat With Your Document — RAG Chatbot

A Retrieval-Augmented Generation (RAG) system that lets you upload a PDF or text
file and ask questions grounded in its actual content — instead of relying on
an LLM's general knowledge, which can hallucinate facts.

## How it works

1. **Ingest** — the document is split into overlapping chunks (~800 characters
   each), and each chunk is converted into a vector embedding using a free,
   local model (`all-MiniLM-L6-v2` from Sentence Transformers).
2. **Store** — embeddings are stored in a FAISS vector index for fast
   similarity search.
3. **Retrieve** — when you ask a question, it's also embedded, and FAISS
   returns the top 3 most semantically similar chunks from the document.
4. **Generate** — those chunks + your question are inserted into a prompt and
   sent to an LLM (Llama 3.1 8B via Groq's free API), which is explicitly
   instructed to answer only from the provided context — this is what keeps
   it grounded rather than making things up.

## Tech stack

- **LangChain** — orchestrates the pipeline (loaders, splitter, chains)
- **FAISS** — vector similarity search
- **Sentence Transformers** — free, local embeddings (no API cost for this step)
- **Groq (Llama 3.1 8B)** — free-tier LLM for final answer generation
- **Streamlit** — web UI

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Get a **free** Groq API key at [console.groq.com/keys](https://console.groq.com/keys)
(no credit card required). Then open the local URL Streamlit prints, paste in
your Groq API key in the sidebar, upload `sample_data/sample_notes.txt`
(included) or your own PDF, and start asking questions.

## Example questions to try with the sample data

- "What's the difference between IDW and Kriging?"
- "Why would you use iterative imputation instead of KNN?"
- "What is transformer fine-tuning?"

## What I learned building this

- How chunking strategy (size + overlap) affects retrieval quality
- Why grounding the LLM's prompt with retrieved context reduces hallucination
- How vector similarity search works differently from keyword search
- Trade-offs between local embeddings (free, fast) vs. API-based embeddings
  (often more accurate, but cost money per call)
