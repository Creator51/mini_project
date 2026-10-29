"""
Phase 9: FastAPI Request & Response Schemas.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class NetworkTrafficPayload(BaseModel):
    """Schema for incoming single network flow payload."""
    features: Dict[str, Any] = Field(
        ..., 
        example={
            "custom_instruction": "SELECT * FROM users WHERE id=1' OR '1'='1",
            "tcp.srcport": "8080",
            "http.request.method": "POST"
        }
    )


class ThreatPredictionResponse(BaseModel):
    """Schema for single threat prediction output with detected attack category."""
    prediction: str
    detected_attack_category: str
    is_attack: bool
    confidence: float
    threat_probability: float
    random_forest_probability: float
    lightgbm_probability: float
    selected_model: str
    inference_latency_ms: float


class BatchPredictionRequest(BaseModel):
    """Schema for batch network flow requests."""
    events: List[Dict[str, Any]]


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction output."""
    total_events: int
    predictions: List[ThreatPredictionResponse]
    total_latency_ms: float
    throughput_events_per_sec: float


class HealthResponse(BaseModel):
    """Schema for health status endpoint."""
    status: str
    version: str
    model_loaded: bool
    expected_raw_features: int
