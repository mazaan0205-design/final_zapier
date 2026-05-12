import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from PyPDF2 import PdfReader

from database import get_db, Chatbot, Message, DocumentMetadata, init_db
from ai_service import ingest_pdf_to_vector, run_ai_logic 

app = FastAPI()

@app.on_event("startup")
def startup():
    init_db()

@app.get("/")
def root():
    return {"status": "Online", "docs": "/docs"}

@app.post("/bots/create")
async def create_bot(bot_id: str = Form(...), name: str = Form(...), instructions: str = Form(...), model: str = Form("Groq"), db: Session = Depends(get_db)):
    db.add(Chatbot(id=bot_id, name=name, instructions=instructions, preferred_model=model))
    db.commit()
    return {"message": "Bot Created"}

@app.post("/bots/{bot_id}/upload")
async def upload_pdf(bot_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    bot = db.query(Chatbot).filter(Chatbot.id == bot_id).first()
    if not bot: raise HTTPException(status_code=404, detail="Bot not found")
    
    text = "".join([p.extract_text() for p in PdfReader(file.file).pages])
    ingest_pdf_to_vector(bot_id, text)
    
    db.add(DocumentMetadata(bot_id=bot_id, filename=file.filename))
    db.commit()
    return {"message": f"Knowledge added to {bot_id}"}

@app.post("/bots/{bot_id}/ask")
async def ask_bot(bot_id: str, question: str = Form(...), db: Session = Depends(get_db)):
    bot = db.query(Chatbot).filter(Chatbot.id == bot_id).first()
    if not bot: raise HTTPException(status_code=404, detail="Bot not found")
    
    answer = run_ai_logic(bot_id, question, bot.instructions, bot.preferred_model)
    db.add(Message(bot_id=bot_id, user_query=question, ai_response=answer))
    db.commit()
    return {"answer": answer}

@app.get("/bots/list")
async def list_bots(db: Session = Depends(get_db)):
    return db.query(Chatbot).all()

# THIS IS THE PART I FORGOT! 
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
