"""
Phase 9: FastAPI Microservice Engine.
Exposes endpoints for health check, model info, real-time single inference, batch inference,
and serves the interactive Cyber Threat Detection SOC Dashboard frontend.
"""

import os
import sys
import time
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.abspath("."))

from api.schemas import (
    NetworkTrafficPayload,
    ThreatPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse
)
from src.realtime.inference import RealTimeThreatDetector

# Global Detector Instance
detector_instance: RealTimeThreatDetector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load models once upon service startup."""
    global detector_instance
    print("[FASTAPI API] Pre-loading ML models into memory on server startup...")
    try:
        detector_instance = RealTimeThreatDetector(model_type="ensemble")
        print("[FASTAPI API] Preloader finished. API ready to serve requests.")
    except Exception as e:
        print(f"[FASTAPI API ERROR] Failed to load models: {str(e)}")
        detector_instance = None
    yield
    print("[FASTAPI API] Shutting down threat detection API.")



app = FastAPI(
    title="Real-Time Cyber Threat Detection API",
    description="Machine Learning Powered Cyber Threat Detection Framework microservice & dashboard.",
    version="1.0.0",
    lifespan=lifespan
)

# Allow CORS for external dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def serve_frontend_dashboard():
    """GET / - Serve interactive HTML SOC dashboard."""
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Cyber Threat Detection Framework API is running."}


@app.get("/styles.css", include_in_schema=False)
def serve_css():
    """Serve frontend CSS styling."""
    return FileResponse(os.path.join("frontend", "styles.css"))


@app.get("/app.js", include_in_schema=False)
def serve_js():
    """Serve frontend JavaScript logic."""
    return FileResponse(os.path.join("frontend", "app.js"))


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """GET /health - Check service and model readiness status."""
    if detector_instance is None or not detector_instance.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML Threat Detection models are not loaded or ready."
        )
    return HealthResponse(
        status="HEALTHY",
        version="1.0.0",
        model_loaded=True,
        expected_raw_features=len(detector_instance.raw_feature_names)
    )


@app.get("/model-info", tags=["Information"])
def model_info():
    """GET /model-info - Return current ML model details and feature expectations."""
    if detector_instance is None:
        raise HTTPException(status_code=503, detail="Detector not ready.")
    
    return {
        "framework": "Hybrid Threat Detection Framework (RF + LightGBM)",
        "primary_dataset": "Edge-IIoTset",
        "ensemble_strategy": "Soft Voting Probability Averaging (alpha=0.50)",
        "expected_input_feature_count": len(detector_instance.raw_feature_names),
        "expected_features": detector_instance.raw_feature_names
    }


@app.post("/predict", response_model=ThreatPredictionResponse, tags=["Inference"])
def predict_threat(payload: NetworkTrafficPayload):
    """POST /predict - Evaluate real-time threat probability for a single network traffic flow."""
    if detector_instance is None:
        raise HTTPException(status_code=503, detail="Threat detector not initialized.")

    try:
        result = detector_instance.predict_single(payload.features)
        return ThreatPredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_threats(batch_request: BatchPredictionRequest):
    """POST /predict/batch - Evaluate real-time threat predictions for a batch of traffic flows."""
    if detector_instance is None:
        raise HTTPException(status_code=503, detail="Threat detector not initialized.")

    if not batch_request.events:
        raise HTTPException(status_code=400, detail="Event list cannot be empty.")

    start_batch_time = time.time()
    try:
        results = detector_instance.predict_batch(batch_request.events)
        total_time_sec = time.time() - start_batch_time
        total_latency_ms = total_time_sec * 1000
        throughput = len(batch_request.events) / total_time_sec if total_time_sec > 0 else 0.0

        predictions = [ThreatPredictionResponse(**r) for r in results]
        return BatchPredictionResponse(
            total_events=len(predictions),
            predictions=predictions,
            total_latency_ms=round(total_latency_ms, 4),
            throughput_events_per_sec=round(throughput, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
