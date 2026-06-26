import pytest

@pytest.fixture
def clean_candidate():
    """A perfectly valid and highly qualified candidate profile."""
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "John Doe",
            "headline": "Senior AI Engineer | RAG & Search",
            "summary": "6 years experience building ML systems.",
            "location": "noida",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "AI Engineer",
            "current_company": "Acme Corp",
            "current_company_size": "51-200",
            "current_industry": "Software"
        },
        "career_history": [
            {
                "company": "Acme Corp",
                "title": "AI Engineer",
                "start_date": "2023-01-01",
                "end_date": None,
                "duration_months": 42, # Valid: Jan 2023 to June 2026 is 42 months
                "is_current": True,
                "industry": "Software",
                "company_size": "51-200",
                "description": "Building search systems."
            }
        ],
        "skills": [
            {
                "name": "Python",
                "proficiency": "expert",
                "endorsements": 50,
                "duration_months": 72
            },
            {
                "name": "NLP",
                "proficiency": "advanced",
                "endorsements": 20,
                "duration_months": 48
            }
        ],
        "redrob_signals": {
            "willing_to_relocate": True,
            "notice_period_days": 30,
            "recruiter_response_rate": 0.90
        }
    }
