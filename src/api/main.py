import uuid
import io
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from .db import init_db, get_db, ScanRecord
from .cache import get_cached, set_cached
from src.inference.pipeline import process_photo

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("Asset Intel API started")
    yield

app = FastAPI(title="Asset Intel API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Asset Intel API running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/scan")
async def scan_asset(
    file: UploadFile = File(...),
    worker_id: str = "anon",
    db: Session = Depends(get_db)
):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Check cache
    cached = get_cached(img)
    if cached:
        print("Cache hit")
        return cached

    # Save temp image
    temp_path = f"temp_{uuid.uuid4()}.jpg"
    img.save(temp_path)

    try:
        results = process_photo(temp_path)

        for item in results:
            top = item.get("top_match")
            record = ScanRecord(
                scan_id=str(uuid.uuid4()),
                image_path=temp_path,
                detected=item.get("detected_class"),
                category=top["metadata"].get("category") if top else None,
                confidence=item.get("yolo_confidence"),
                clip_score=top["score"] if top else None,
                flagged=top["score"] < 0.75 if top else True,
                metadata_=top["metadata"] if top else {},
                scanned_at=datetime.utcnow()
            )
            db.add(record)
        db.commit()

        response = jsonable_encoder({
            "status": "ok",
            "worker_id": worker_id,
            "items_found": len(results),
            "results": results
        })

        set_cached(img, response)
        return response

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/scans")
def get_scans(limit: int = 20, db: Session = Depends(get_db)):
    scans = db.query(ScanRecord).order_by(
        ScanRecord.scanned_at.desc()
    ).limit(limit).all()
    return {"scans": [
        {
            "scan_id": s.scan_id,
            "detected": s.detected,
            "category": s.category,
            "confidence": s.confidence,
            "clip_score": s.clip_score,
            "flagged": s.flagged,
            "scanned_at": str(s.scanned_at)
        } for s in scans
    ]}