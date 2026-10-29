"""
Unit and Integration Tests for FastAPI microservice and Real-Time Inference Engine.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("."))

from api.main import app


@pytest.fixture
def client():
    """Create test client with lifespan initialization."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Test GET /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True


def test_model_info_endpoint(client):
    """Test GET /model-info endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "expected_features" in data
    assert len(data["expected_features"]) > 0


def test_predict_single_endpoint(client):
    """Test POST /predict with realistic network sample features."""
    sample_payload = {
        "features": {
            "arp.opcode": 0.0,
            "tcp.ack": 100.0,
            "tcp.srcport": "80",
            "http.request.method": "GET"
        }
    }
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "is_attack" in data
    assert "confidence" in data
    assert "inference_latency_ms" in data


def test_predict_batch_endpoint(client):
    """Test POST /predict/batch endpoint with multiple events."""
    sample_batch = {
        "events": [
            {"arp.opcode": 0.0, "tcp.ack": 50.0},
            {"arp.opcode": 1.0, "tcp.ack": 500.0}
        ]
    }
    response = client.post("/predict/batch", json=sample_batch)
    assert response.status_code == 200
    data = response.json()
    assert data["total_events"] == 2
    assert len(data["predictions"]) == 2
    assert "throughput_events_per_sec" in data
