import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- PART A: ADDING KNOWLEDGE ---
def ingest_pdf_to_vector(bot_id, pdf_text):
    # 1. Chunk the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_text(pdf_text)
    
    # 2. Local Embedding
    # This points to a folder NAMED after the bot_id (e.g., ./knowledge_base/meezan-bank)
    persist_dir = f"./knowledge_base/{bot_id}"
    
    vector_db = Chroma.from_texts(
        texts=splits,
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory=persist_dir
    )
    return True

# --- PART B: ASKING QUESTIONS ---
def run_ai_logic(bot_id, question, instructions, model_choice):
    persist_dir = f"./knowledge_base/{bot_id}"
    
    # Load ONLY this bot's specific vector database
    if not os.path.exists(persist_dir):
        return "This bot has no knowledge base yet. Please upload a PDF first."

    vector_db = Chroma(
        persist_directory=persist_dir,
        embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    
    # Search for relevant snippets
    docs = vector_db.similarity_search(question, k=4)
    context = "\n".join([d.page_content for d in docs])
    
    system_prompt = f"{instructions}\n\nRELEVANT DOCUMENT CONTEXT:\n{context}"
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    
    if "Groq" in str(model_choice):
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
    
    return llm.invoke(messages).content
