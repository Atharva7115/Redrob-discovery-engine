import pytest
from datetime import date
from src.retrieval.filters import (
    is_honeypot_duration_anomaly,
    is_honeypot_skill_anomaly,
    is_honeypot,
    apply_hard_filters
)

# Mock Candidates for Testing


@pytest.fixture
def duration_honeypot():
    """A honeypot candidate with an impossible work duration."""
    return {
        "candidate_id": "CAND_0000002",
        "profile": {
            "anonymized_name": "Trap One",
            "headline": "Backend Engineer",
            "years_of_experience": 9.9,
            "country": "India",
            "location": "pune"
        },
        "career_history": [
            {
                "company": "Wayne Enterprises",
                "title": "Frontend Engineer",
                "start_date": "2023-09-10",
                "end_date": None,
                "duration_months": 166, # Physically impossible!
                "is_current": True,
                "description": "Frontend developer."
            }
        ],
        "skills": [],
        "redrob_signals": {}
    }

@pytest.fixture
def skill_honeypot():
    """A honeypot candidate with expert skills but 0 experience."""
    return {
        "candidate_id": "CAND_0000003",
        "profile": {
            "anonymized_name": "Trap Two",
            "years_of_experience": 4.0,
            "country": "India",
            "location": "noida"
        },
        "career_history": [],
        "skills": [
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 10, "duration_months": 0},
            {"name": "Milvus", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
            {"name": "Qdrant", "proficiency": "advanced", "endorsements": 15, "duration_months": 0},
            {"name": "FAISS", "proficiency": "expert", "endorsements": 20, "duration_months": 0},
            {"name": "LoRA", "proficiency": "expert", "endorsements": 2, "duration_months": 0}
        ],
        "redrob_signals": {}
    }

@pytest.fixture
def junior_candidate(clean_candidate):
    """A candidate with less than the minimum required experience."""
    cand = clean_candidate.copy()
    cand["profile"] = clean_candidate["profile"].copy()
    cand["profile"]["years_of_experience"] = 1.5
    return cand

@pytest.fixture
def foreign_candidate(clean_candidate):
    """A candidate located outside India who requires visa sponsorship."""
    cand = clean_candidate.copy()
    cand["profile"] = clean_candidate["profile"].copy()
    cand["profile"]["country"] = "Canada"
    cand["profile"]["location"] = "Toronto"
    cand["redrob_signals"] = clean_candidate["redrob_signals"].copy()
    cand["redrob_signals"]["willing_to_relocate"] = False
    return cand

# Test Cases

def test_clean_candidate_passes(clean_candidate):
    assert is_honeypot_duration_anomaly(clean_candidate) is False
    assert is_honeypot_skill_anomaly(clean_candidate) is False
    assert is_honeypot(clean_candidate) is False
    
    passed, reason = apply_hard_filters(clean_candidate)
    assert passed is True
    assert reason == ""

def test_duration_honeypot_fails(duration_honeypot):
    assert is_honeypot_duration_anomaly(duration_honeypot) is True
    assert is_honeypot(duration_honeypot) is True
    
    passed, reason = apply_hard_filters(duration_honeypot)
    assert passed is False
    assert "Honeypot" in reason

def test_skill_honeypot_fails(skill_honeypot):
    assert is_honeypot_skill_anomaly(skill_honeypot) is True
    assert is_honeypot(skill_honeypot) is True
    
    passed, reason = apply_hard_filters(skill_honeypot)
    assert passed is False
    assert "Honeypot" in reason

def test_junior_candidate_fails_yoe_floor(junior_candidate):
    passed, reason = apply_hard_filters(junior_candidate)
    assert passed is False
    assert "Experience floor" in reason

def test_foreign_candidate_fails_visa(foreign_candidate):
    passed, reason = apply_hard_filters(foreign_candidate)
    assert passed is False
    assert "Visa restriction" in reason
