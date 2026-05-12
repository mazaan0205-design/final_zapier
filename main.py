import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from PyPDF2 import PdfReader

# Import your own files
from database import SessionLocal, Chatbot, Message, DocumentMetadata, get_db
from ai_service import run_ai_logic # Assuming this is your updated RAG function

app = FastAPI(title="W3X Zapier-Style Engine")

# --- ENDPOINT 1: CREATE A NEW BOT ---
@app.post("/bots/create")
async def create_bot(
    bot_id: str = Form(...),
    name: str = Form(...),
    instructions: str = Form(...),
    model: str = Form("Groq"),
    db: Session = Depends(get_db)
):
    # Check if bot already exists
    existing_bot = db.query(Chatbot).filter(Chatbot.id == bot_id).first()
    if existing_bot:
        raise HTTPException(status_code=400, detail="Bot ID already exists")
    
    new_bot = Chatbot(
        id=bot_id,
        name=name,
        instructions=instructions,
        preferred_model=model
    )
    db.add(new_bot)
    db.commit()
    return {"status": "Bot created successfully", "bot_id": bot_id}

# --- ENDPOINT 2: ASK & PROCESS PDF (The Heavy Lifter) ---
@app.post("/bots/{bot_id}/ask")
async def ask_bot(
    bot_id: str,
    question: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. Verify Bot exists in SQL
    bot = db.query(Chatbot).filter(Chatbot.id == bot_id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    # 2. Extract PDF Text (Local)
    try:
        reader = PdfReader(file.file)
        pdf_text = "".join([page.extract_text() for page in reader.pages])
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read PDF")

    # 3. Call your AI Service (Local Chunking/Embedding happens here)
    # This sends only the snippet to Groq/Gemini to save your limits
    answer = run_ai_logic(
        bot_id=bot_id,
        text=pdf_text,
        question=question,
        instructions=bot.instructions,
        model_choice=bot.preferred_model
    )

    # 4. Save the Document Metadata
    doc_entry = DocumentMetadata(bot_id=bot_id, filename=file.filename, file_type="pdf")
    db.add(doc_entry)

    # 5. Save the Message History
    new_message = Message(
        bot_id=bot_id,
        user_query=question,
        ai_response=answer,
        model_used=bot.preferred_model
    )
    db.add(new_message)
    
    db.commit()
    return {"answer": answer}

# --- ENDPOINT 3: GET FULL HISTORY ---
@app.get("/bots/{bot_id}/history")
async def get_history(bot_id: str, db: Session = Depends(get_db)):
    # Pull all messages for this specific bot ordered by time
    messages = db.query(Message).filter(Message.bot_id == bot_id).order_by(Message.timestamp.asc()).all()
    
    history_list = []
    for msg in messages:
        history_list.append({
            "user": msg.user_query,
            "bot": msg.ai_response,
            "time": msg.timestamp,
            "model": msg.model_used
        })
    
    return {"bot_id": bot_id, "chat_history": history_list}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)