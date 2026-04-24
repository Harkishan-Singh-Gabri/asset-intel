import os
from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./asset_intel.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class ScanRecord(Base):
    __tablename__ = "scans"
    scan_id    = Column(String, primary_key=True)
    image_path = Column(String)
    detected   = Column(String)
    category   = Column(String)
    confidence = Column(Float)
    clip_score = Column(Float)
    flagged    = Column(Boolean, default=False)
    metadata_  = Column(JSON)
    scanned_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)
    print("DB initialized")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()