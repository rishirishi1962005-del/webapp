"""
Brain Tumor Detection API
FastAPI backend with YOLOv8 inference and hospital recommendation
"""

import os
import sys
import uuid
import base64
import logging
from pathlib import Path
from typing import Optional

# Fix YOLO config path (Render issue)
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

from model_loader import TumorDetector
from hospital_service import HospitalService

# ── Setup Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Path Constants ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "best.pt"
HOSPITAL_CSV = BASE_DIR / "data" / "data.csv"
GEOJSON_PATH = BASE_DIR / "data" / "TamilNadu.geojson"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ── Initialize FastAPI ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Brain Tumor Detection API",
    description="YOLOv8-powered brain tumor detection with hospital recommendations",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# ── Lazy Loaded Services ───────────────────────────────────────────────────────
tumor_detector: Optional[TumorDetector] = None
hospital_service: Optional[HospitalService] = None


def get_tumor_detector():
    global tumor_detector

    if tumor_detector is None:
        logger.info("🧠 Lazy loading YOLO model...")

        if MODEL_PATH.exists():
            tumor_detector = TumorDetector(str(MODEL_PATH))
            logger.info("✅ Model loaded")
        else:
            logger.warning("⚠️ Model not found, using demo mode")
            tumor_detector = TumorDetector(None)

    return tumor_detector


def get_hospital_service():
    global hospital_service

    if hospital_service is None:
        logger.info("🏥 Loading hospital data...")
        hospital_service = HospitalService(str(HOSPITAL_CSV))
        logger.info("✅ Hospital data loaded")

    return hospital_service


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    frontend_path = BASE_DIR / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    return {"message": "API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]

    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    result_path = RESULT_DIR / f"result_{file_id}.jpg"

    try:
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)

        logger.info(f"📁 Saved: {upload_path}")

        detector = get_tumor_detector()
        result = detector.predict(str(upload_path), str(result_path))

        # Encode image
        if result_path.exists():
            with open(result_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
        else:
            with open(upload_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

        return JSONResponse({
            "success": True,
            "class": result["class"],
            "confidence": round(result["confidence"] * 100, 2),
            "has_tumor": result["class"] != "notumor",
            "image": img_b64
        })

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if upload_path.exists():
            os.remove(upload_path)


@app.post("/hospitals")
async def get_hospitals(city: str = Form(...)):
    if not city or len(city.strip()) < 2:
        raise HTTPException(status_code=400, detail="Invalid city")

    service = get_hospital_service()
    result = service.get_hospitals(city.strip(), top_n=5)

    return JSONResponse(result)


@app.get("/geojson")
async def get_geojson():
    if not GEOJSON_PATH.exists():
        raise HTTPException(status_code=404, detail="GeoJSON not found")
    return FileResponse(str(GEOJSON_PATH))


@app.get("/cities")
async def get_cities():
    service = get_hospital_service()
    if "city" not in service.df.columns:
        return {"cities": []}
    return {"cities": sorted(service.df["city"].dropna().unique().tolist())}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": tumor_detector is not None,
        "hospitals_loaded": hospital_service is not None
    }


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port)
