from sqlalchemy import create_engine, Column, String, Text, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

DATABASE_URL = "sqlite:///./zapier_clone_v1.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Chatbot(Base):
    __tablename__ = "chatbots"
    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    instructions = Column(Text)
    preferred_model = Column(String, default="Groq")

class DocumentMetadata(Base):
    __tablename__ = "documents"
    id = Column(DateTime, primary_key=True, default=datetime.datetime.utcnow)
    bot_id = Column(String, ForeignKey("chatbots.id"))
    filename = Column(String)

class Message(Base):
    __tablename__ = "messages"
    id = Column(DateTime, primary_key=True, default=datetime.datetime.utcnow)
    bot_id = Column(String, ForeignKey("chatbots.id"))
    user_query = Column(Text)
    ai_response = Column(Text)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
