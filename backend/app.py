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

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ── Initialize FastAPI ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Brain Tumor Detection API",
    description="YOLOv8-powered brain tumor detection with hospital recommendations in Tamil Nadu",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# ── Load Services at Startup ───────────────────────────────────────────────────
tumor_detector: Optional[TumorDetector] = None
hospital_service: Optional[HospitalService] = None

@app.on_event("startup")
async def startup_event():
    """Load model and hospital data once at startup."""
    global tumor_detector, hospital_service

    logger.info("🧠 Loading YOLOv8 model...")
    if MODEL_PATH.exists():
        tumor_detector = TumorDetector(str(MODEL_PATH))
        logger.info(f"✅ Model loaded from {MODEL_PATH}")
    else:
        logger.warning(f"⚠️  Model file not found at {MODEL_PATH}. Using demo mode.")
        tumor_detector = TumorDetector(None)  # Demo/mock mode

    logger.info("🏥 Loading hospital data...")
    hospital_service = HospitalService(str(HOSPITAL_CSV))
    logger.info(f"✅ Hospital data loaded: {len(hospital_service.df)} records")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the frontend HTML."""
    frontend_path = BASE_DIR / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    return {"message": "Brain Tumor Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an MRI scan and run tumor detection.
    Returns: class name, confidence, annotated image (base64).
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload JPEG or PNG.")

    # Save uploaded file
    file_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    result_path = RESULT_DIR / f"result_{file_id}.jpg"

    try:
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)

        logger.info(f"📁 Saved upload: {upload_path}")

        # Run inference
        if tumor_detector is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        result = tumor_detector.predict(str(upload_path), str(result_path))

        # Encode result image to base64
        result_image_b64 = None
        if result_path.exists():
            with open(result_path, "rb") as img_f:
                result_image_b64 = base64.b64encode(img_f.read()).decode("utf-8")
        else:
            # Return original if no bounding box
            with open(upload_path, "rb") as img_f:
                result_image_b64 = base64.b64encode(img_f.read()).decode("utf-8")

        return JSONResponse({
            "success": True,
            "class": result["class"],
            "confidence": round(result["confidence"] * 100, 2),
            "has_tumor": result["class"] != "notumor",
            "image": result_image_b64,
            "message": result.get("message", "")
        })

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Clean up upload file
        if upload_path.exists():
            os.remove(upload_path)


@app.post("/hospitals")
async def get_hospitals(city: str = Form(...)):
    """
    Get top 5 nearest hospitals for a given Tamil Nadu city.
    Falls back to nearest hospitals by Haversine distance if city not found.
    """
    if not city or len(city.strip()) < 2:
        raise HTTPException(status_code=400, detail="Please enter a valid city name.")

    if hospital_service is None:
        raise HTTPException(status_code=503, detail="Hospital service not available.")

    city = city.strip()
    result = hospital_service.get_hospitals(city, top_n=5)

    return JSONResponse({
        "success": True,
        "city": city,
        "matched_city": result["matched_city"],
        "exact_match": result["exact_match"],
        "hospitals": result["hospitals"],
        "user_coords": result.get("user_coords"),
        "message": result["message"]
    })


@app.get("/geojson")
async def get_geojson():
    """Serve the Tamil Nadu GeoJSON for map rendering."""
    if not GEOJSON_PATH.exists():
        raise HTTPException(status_code=404, detail="GeoJSON file not found.")
    return FileResponse(str(GEOJSON_PATH), media_type="application/json")


@app.get("/cities")
async def get_cities():
    """Return list of available cities for autocomplete."""
    if hospital_service is None:
        return {"cities": []}
    if "city" not in hospital_service.df.columns:
        return {"cities": []}
    cities = sorted(hospital_service.df["city"].dropna().unique().tolist())
    return {"cities": cities}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": tumor_detector is not None and tumor_detector.model is not None,
        "hospitals_loaded": hospital_service is not None
    }


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
