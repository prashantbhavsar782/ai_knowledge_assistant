from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

DATA_PATH = "data/sample.txt"
FAISS_INDEX_PATH = "faiss_index"

def ingest_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    db.save_local(FAISS_INDEX_PATH)
    print(f"✅ Data Ingested & FAISS index saved at '{FAISS_INDEX_PATH}'")

if __name__ == "__main__":
    ingest_data()
