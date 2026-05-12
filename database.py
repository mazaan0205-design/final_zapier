import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Text, DateTime, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# 1. SETUP BASE
Base = declarative_base()

# 2. THE BOT REGISTRY: Stores the configuration for every unique bot
class Chatbot(Base):
    __tablename__ = 'chatbots'
    
    id = Column(String, primary_key=True) # Unique ID (e.g., 'marketing-bot-01')
    name = Column(String, nullable=False) # Human-readable name
    instructions = Column(Text, nullable=False) # The "System Prompt" / Rules
    preferred_model = Column(String, default="Groq") # Default model choice
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    messages = relationship("Message", back_populates="bot", cascade="all, delete-orphan")
    documents = relationship("DocumentMetadata", back_populates="bot", cascade="all, delete-orphan")

# 3. DOCUMENT METADATA: Tracks which files were uploaded to which bot
class DocumentMetadata(Base):
    __tablename__ = 'documents'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    bot_id = Column(String, ForeignKey('chatbots.id'))
    filename = Column(String)
    file_type = Column(String) # e.g., 'pdf'
    upload_date = Column(DateTime, default=datetime.utcnow)
    
    bot = relationship("Chatbot", back_populates="documents")

# 4. MESSAGE LOGS: Stores every single chat interaction
class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    bot_id = Column(String, ForeignKey('chatbots.id'))
    user_query = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    model_used = Column(String) # Track if Groq or Gemini was used
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    bot = relationship("Chatbot", back_populates="messages")

# --- DATABASE ENGINE SETUP ---
# This creates a local file named 'zapier_clone_v1.db' on your Mac
DATABASE_URL = "sqlite:///./zapier_clone_v1.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to create all tables (Run this once at start)
def init_db():
    Base.metadata.create_all(bind=engine)

# Dependency for FastAPI to handle opening/closing connections
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Ensure tables are created when this script runs
if __name__ == "__main__":
    init_db()
    print("Database and Tables created successfully!")