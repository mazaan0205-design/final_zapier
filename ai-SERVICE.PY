import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def process_rag_logic(chatbot_id, text, question, instructions, model_choice):
    # CHUNKING (Happens on your Mac API)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = splitter.split_text(text)
    
    # EMBEDDING (Happens on your Mac API - No API limit hit here)
    vector_store = Chroma.from_texts(
        texts=splits,
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory=f"./db/{chatbot_id}"
    )
    
    # SEARCH (Get only the small relevant part)
    docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])
    
    # MULTIMODAL SWITCH (Groq or Gemini)
    system_prompt = f"{instructions}\n\nCONTEXT:\n{context}"
    payload = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    
    if "Groq" in model_choice:
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
    
    return llm.invoke(payload).content