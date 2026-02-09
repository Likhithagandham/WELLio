"""
FastAPI Backend for rPPG Vitals Estimation
===========================================

Production-ready REST API with WebSocket support.
Handles async video processing, result caching, and real-time streaming.

Run with: uvicorn rppg_fastapi:app --reload
Test at:  http://localhost:8000/docs
"""

import os
import asyncio
import tempfile
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path
import io

from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiofiles

from rppg_refactored import (
    estimate_vitals_from_video,
    VitalsEstimate,
    FilteredSignal,
    RiskAssessment,
    HAVE_MEDIAPIPE
)

try:
    from s3_utils import generate_presigned_url
    from auth import get_supabase_client
    HAVE_S3 = True
except ImportError:
    HAVE_S3 = False

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, Security

# ============================================================================
# CONFIG & LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache for processed videos (keep last 100 results)
VIDEO_CACHE: Dict[str, Dict] = {}
MAX_CACHE_SIZE = 100
CACHE_TTL = 3600  # 1 hour

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

# ============================================================================
# PYDANTIC MODELS (API Schemas)
# ============================================================================

class VitalsResponse(BaseModel):
    """Vitals estimation response"""
    heart_rate_bpm: Optional[float]
    heart_rate_confidence: str
    sdnn_ms: Optional[float] = None
    pnn50_percent: Optional[float] = None
    stress_level: Optional[float] = None
    bp_systolic: Optional[float] = None
    bp_diastolic: Optional[float] = None
    bp_note: str
    spo2: Optional[float] = None
    spo2_note: str
    rr_interval_count: int
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "heart_rate_bpm": 72.5,
                "heart_rate_confidence": "HIGH",
                "sdnn_ms": 45.2,
                "pnn50_percent": 18.5,
                "stress_level": 3.2,
                "bp_systolic": 125.0,
                "bp_diastolic": 85.0,
                "bp_note": "Experimental estimate ±~15 mmHg. Not validated.",
                "spo2": None,
                "spo2_note": "Not estimated (visible light only)",
                "rr_interval_count": 72
            }
        }
    }


class RiskResponse(BaseModel):
    """Risk assessment response"""
    risk_score: int
    risk_level: str  # LOW, MODERATE, HIGH
    alerts: List[str]
    recommendation: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "risk_score": 2,
                "risk_level": "LOW",
                "alerts": [],
                "recommendation": "Low risk indicators (experimental). Consult a professional if concerned."
            }
        }
    }


class AnalysisResponse(BaseModel):
    """Complete analysis response"""
    request_id: str
    timestamp: str
    video_duration_sec: Optional[float]
    analysis_time_sec: float
    vitals: VitalsResponse
    risk: RiskResponse
    message: str
    disclaimer: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    mediapipe_available: bool

class UploadURLRequest(BaseModel):
    filename: str
    file_type: str  # e.g. "video/mp4"

class UploadURLResponse(BaseModel):
    upload_url: str
    s3_key: str
    bucket: str
    filename: str
    expires_in: int



# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="rPPG Vitals API",
    description="Experimental camera-based vitals estimation (Remote Photoplethysmography)",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# CORS middleware for web/mobile clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# UTILITIES
# ============================================================================

def generate_request_id() -> str:
    """Generate unique request ID"""
    return f"rppg_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"


def vitals_to_response(vitals: VitalsEstimate, filtered: FilteredSignal) -> VitalsResponse:
    """Convert VitalsEstimate to API response model"""
    
    # Determine confidence string based on validation or filtered signal quality
    if vitals.heart_rate_valid:
        if filtered.confidence_percent > 80:
            conf_str = "HIGH"
        elif filtered.confidence_percent > 60:
            conf_str = "MEDIUM"
        else:
            conf_str = "LOW"
    else:
        conf_str = "VERY LOW"

    return VitalsResponse(
        heart_rate_bpm=vitals.heart_rate_bpm,
        heart_rate_confidence=conf_str,
        sdnn_ms=vitals.sdnn,
        pnn50_percent=None, # Not calculated in simple backend
        stress_level=vitals.stress_level,
        bp_systolic=vitals.bp_systolic,
        bp_diastolic=vitals.bp_diastolic,
        bp_note="Experimental heuristic. NOT VALIDATED.",
        spo2=vitals.spo2,
        spo2_note="Experimental estimate from green channel (uncalibrated).",
        rr_interval_count=len(vitals.rr_intervals) if vitals.rr_intervals is not None else 0
    )


def risk_to_response(risk: RiskAssessment) -> RiskResponse:
    """Convert RiskAssessment to API response model"""
    return RiskResponse(
        risk_score=risk.risk_score,
        risk_level=risk.risk_level,
        alerts=risk.alerts,
        recommendation=risk.recommendation
    )


async def cleanup_cache():
    """Periodically clean expired cache entries"""
    now = datetime.utcnow()
    expired = [
        req_id for req_id, data in VIDEO_CACHE.items()
        if datetime.fromisoformat(data['timestamp']) + timedelta(seconds=CACHE_TTL) < now
    ]
    for req_id in expired:
        del VIDEO_CACHE[req_id]
    
    # Keep only most recent if over limit
    if len(VIDEO_CACHE) > MAX_CACHE_SIZE:
        sorted_cache = sorted(VIDEO_CACHE.items(), 
                             key=lambda x: x[1]['timestamp'], 
                             reverse=True)
        VIDEO_CACHE.clear()
        for req_id, data in sorted_cache[:MAX_CACHE_SIZE]:
            VIDEO_CACHE[req_id] = data


# ============================================================================
# ROUTES: HEALTH & INFO
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        mediapipe_available=HAVE_MEDIAPIPE
    )


@app.get("/info")
async def get_info():
    """API information and capabilities"""
    return {
        "name": "rPPG Vitals Estimation API",
        "description": "Experimental camera-based vital signs from facial video",
        "version": "1.0.0",
        "capabilities": {
            "heart_rate": "Supported (±5-10 BPM error typical)",
            "hrv": "Supported (SDNN, pNN50)",
            "stress": "Experimental heuristic (0-10 scale)",
            "blood_pressure": "Experimental (±15 mmHg error, NOT validated)",
            "spo2": "Not supported (visible light only)",
            "risk_scoring": "Experimental heuristic"
        },
        "disclaimer": (
            "This is a research prototype. All outputs are experimental and NOT clinically validated. "
            "Do NOT use for medical decisions. Consult healthcare professionals."
        ),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "cache_enabled": True,
        "cache_ttl_sec": CACHE_TTL,
    }


# ============================================================================
# ROUTES: ANALYSIS (Main Endpoint)
# ============================================================================

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    use_mediapipe: bool = Query(default=HAVE_MEDIAPIPE, description="Use MediaPipe for face detection"),
    background_tasks: BackgroundTasks = None
):
    """
    Analyze video for vital signs.
    
    **Parameters:**
    - `file`: MP4/MOV/AVI/MKV video file (max 500 MB)
    - `use_mediapipe`: Use MediaPipe face mesh (default: True if available)
    
    **Response:**
    - `vitals`: Heart rate, HRV, stress, experimental BP/SpO₂
    - `risk`: Risk score and alerts
    - `disclaimer`: Important safety disclaimer
    
    **Typical latency:** 5-30 seconds depending on video length
    """
    
    request_id = generate_request_id()
    logger.info(f"[{request_id}] New analysis request: {file.filename}")
    
    # Validate file
    if not file.filename or '.' not in file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    ext = file.filename.split('.')[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type .{ext} not supported. Use {ALLOWED_EXTENSIONS}")
    
    # Check file size
    file_data = await file.read()
    if len(file_data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {MAX_FILE_SIZE/(1024*1024):.0f} MB")
    
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, f"video_input.{ext}")
    
    try:
        # Save file
        async with aiofiles.open(tmp_path, 'wb') as f:
            await f.write(file_data)
        
        logger.info(f"[{request_id}] Processing video: {len(file_data) / (1024*1024):.1f} MB")
        
        # Run analysis in thread pool (CPU-bound)
        start_time = datetime.utcnow()
        vitals, filtered_signal, risk = await asyncio.to_thread(
            estimate_vitals_from_video,
            tmp_path,
            use_mediapipe
        )
        analysis_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"[{request_id}] Analysis complete. HR: {vitals.heart_rate_bpm:.1f} BPM, "
                   f"Risk: {risk.risk_level}, Time: {analysis_time:.1f}s")
        
        # Build response
        response = AnalysisResponse(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            video_duration_sec=None,  # could compute from video metadata
            analysis_time_sec=analysis_time,
            vitals=vitals_to_response(vitals, filtered_signal),
            risk=risk_to_response(risk),
            message=f"Analysis complete in {analysis_time:.1f}s",
            disclaimer=(
                "⚠️  EXPERIMENTAL RESEARCH PROTOTYPE. Not clinically validated. "
                "Do NOT use for medical decisions. "
                "BP/SpO₂ estimates are uncalibrated and may have ±15% error. "
                "Consult healthcare professionals for medical concerns."
            )
        )
        
        # Cache result
        VIDEO_CACHE[request_id] = {
            'timestamp': response.timestamp,
            'filename': file.filename,
            'response': response.dict()
        }
        
        # Cleanup old cache entries in background
        if background_tasks:
            background_tasks.add_task(cleanup_cache)
        
        return response
    
    except ValueError as e:
        logger.warning(f"[{request_id}] Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Video processing error: {str(e)}")
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Cleanup temp files
        try:
            os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except:
            pass


# ============================================================================
# ROUTES: RESULTS & CACHE
# ============================================================================

@app.get("/results/{request_id}", response_model=AnalysisResponse)
async def get_result(request_id: str):
    """Retrieve cached analysis result by request ID"""
    if request_id not in VIDEO_CACHE:
        raise HTTPException(status_code=404, detail=f"Result not found or expired: {request_id}")
    
    cached_data = VIDEO_CACHE[request_id]
    return AnalysisResponse(**cached_data['response'])


@app.get("/cache/stats")
async def cache_stats():
    """Cache statistics"""
    return {
        "cached_analyses": len(VIDEO_CACHE),
        "max_cache_size": MAX_CACHE_SIZE,
        "ttl_seconds": CACHE_TTL,
        "oldest_entry": min([d['timestamp'] for d in VIDEO_CACHE.values()]) if VIDEO_CACHE else None,
    }


@app.delete("/cache/{request_id}")
async def delete_cached_result(request_id: str):
    """Delete cached result"""
    if request_id in VIDEO_CACHE:
        del VIDEO_CACHE[request_id]
        return {"message": f"Result {request_id} deleted"}
    raise HTTPException(status_code=404, detail="Result not found")


# ============================================================================
# WEBSOCKET: REAL-TIME STREAMING (Optional, for mobile apps)
# ============================================================================

@app.websocket("/ws/vitals")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time vital sign streaming.
    
    Client sends: JSON with 'action' and 'data'
    Server sends: Analysis results as JSON
    
    Example flow:
    1. Client: {"action": "start_recording", "duration_sec": 15}
    2. Client (streams frames or video file)
    3. Server: {"status": "processing", "progress": 0.5}
    4. Server: {"status": "complete", "vitals": {...}, "risk": {...}}
    """
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action", "")
            
            if action == "ping":
                await websocket.send_json({"status": "pong"})
            
            elif action == "test_analysis":
                # Dummy test response
                await websocket.send_json({
                    "status": "complete",
                    "vitals": {
                        "heart_rate_bpm": 72.0,
                        "heart_rate_confidence": "HIGH",
                        "stress_level": 3.5
                    },
                    "risk": {
                        "risk_level": "LOW",
                        "risk_score": 1
                    }
                })
            
            else:
                await websocket.send_json({
                    "error": f"Unknown action: {action}. Try 'ping' or 'test_analysis'."
                })
    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
        await websocket.close()


# ============================================================================
# ROUTES: S3 UPLOAD
# ============================================================================

security = HTTPBearer()

@app.post("/api/upload/url", response_model=UploadURLResponse)
async def get_upload_url(
    item: UploadURLRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Generate a presigned S3 upload URL for authenticated users.
    
    1. Verifies Supabase JWT.
    2. Generates unique S3 key.
    3. Returns presigned URL.
    """
    if not HAVE_S3:
        raise HTTPException(status_code=501, detail="S3 integration not enabled")

    token = credentials.credentials
    supabase = get_supabase_client()
    
    if not supabase:
         raise HTTPException(status_code=500, detail="Database connection error")

    # Verify JWT with Supabase
    try:
        user = supabase.auth.get_user(token)
        if not user:
             raise HTTPException(status_code=401, detail="Invalid token")
        user_id = user.user.id
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")

    # Generate unique key
    if item.file_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF reports can be uploaded")

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    s3_key = f"reports/{user_id}/{timestamp}_{os.urandom(4).hex()}.pdf"
    bucket_name = os.environ.get("AWS_S3_BUCKET", "wellio-uploads") # Default or env var

    # Generate URL
    expires_in = 3600
    presigned_data = generate_presigned_url(
        bucket_name,
        s3_key,
        expiration=expires_in,
        content_type=item.file_type
    )
    
    if not presigned_data:
         raise HTTPException(status_code=500, detail="Failed to generate upload URL")
         
    return UploadURLResponse(
        upload_url=presigned_data['url'],
        s3_key=s3_key,
        bucket=bucket_name,
        filename=item.filename,
        expires_in=expires_in
    )


# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

# Note: on_event decorators are deprecated in FastAPI 0.93+
# Cache cleanup happens automatically as it's session-scoped


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1  # Keep single worker for simplicity; scale with Gunicorn in production
    )
