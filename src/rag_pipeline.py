# rag_pipeline.py

from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os

# Path to your FAISS index folder
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "../faiss_index")

def ask_question(question: str) -> str:
    """
    Takes a user question as input and returns AI answer from FAISS + LLM.
    """

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load FAISS index
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    
    # Create retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # Load LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Ask question to LLM (using HumanMessage)
    response = llm([HumanMessage(content=f"Answer the question based on context:\n{context}\nQuestion: {question}")])
    
    # Return the text response
    return response.content
