import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test that the health check endpoint returns success and correct schema."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vector_index_loaded" in data
    assert "total_candidates" in data

def test_config_get_and_post():
    """Test getting and updating configurations dynamically at runtime."""
    # 1. Get current config
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "min_experience_years" in data
    original_yoe = data["min_experience_years"]
    
    # 2. Update config
    update_data = {"min_experience_years": 4.5}
    response = client.post("/api/config", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "updated"
    assert data["config"]["min_experience_years"] == 4.5
    
    # 3. Restore original config
    restore_data = {"min_experience_years": original_yoe}
    client.post("/api/config", json=restore_data)

def test_evaluate_endpoint():
    """Test that the evaluate endpoint runs successfully and returns baseline metrics."""
    response = client.get("/api/evaluate")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert "metrics" in data
    metrics = data["metrics"]
    assert "NDCG@10" in metrics
    assert "Composite_Score" in metrics
    assert metrics["NDCG@10"] > 0.8

def test_rank_endpoint():
    """Test that the /api/rank endpoint returns successfully ranked candidates."""
    payload = {
        "jd_text": "We are looking for a Senior AI Engineer with experience in FAISS, vector databases, and Python.",
        "use_llm": False,
        "top_k": 5
    }
    response = client.post("/api/rank", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "job_title" in data
    assert "company" in data
    assert "total_candidates_evaluated" in data
    assert "shortlist" in data
    
    shortlist = data["shortlist"]
    assert len(shortlist) > 0
    assert len(shortlist) <= 5
    
    # Check the schema of a candidate response
    first = shortlist[0]
    assert "candidate_id" in first
    assert "name" in first
    assert "score" in first
    assert "rank" in first
    assert "justification" in first
    assert first["rank"] == 1

