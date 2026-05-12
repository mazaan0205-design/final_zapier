import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def run_rag_engine(bot_id, pdf_text, user_query, rules, model_choice):
    # 1. Local Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(pdf_text)
    
    # 2. Local Embedding (Free, stays on your Mac)
    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory=f"./knowledge_base/{bot_id}"
    )
    
    # 3. Smart Search (Snippet Only)
    relevant_docs = vector_db.similarity_search(user_query, k=3)
    context = "\n".join([d.page_content for d in relevant_docs])
    
    # 4. Multi-Model Call
    prompt = [SystemMessage(content=f"{rules}\nContext:\n{context}"), HumanMessage(content=user_query)]
    
    if "Groq" in model_choice:
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
    
    return llm.invoke(prompt).content