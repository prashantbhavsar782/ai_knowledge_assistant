# AI Knowledge Assistant 🧠

An end-to-end RAG (Retrieval-Augmented Generation) application that lets you ask questions over your documents using FAISS vector search** and LLM-powered question answering with a Flask web interface.

---

## Features
- Data ingestion & chunking
- FAISS vector search for efficient retrieval
- LLM-powered question answering
- Flask web interface for easy interaction

---

## Setup

# Activate virtual environment
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
setx OPENAI_API_KEY "your_api_key_here"

# Ingest your data to create FAISS index
python src\data_ingestion.py

# Run the web app
python src\app.py
