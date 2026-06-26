import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(BASE_DIR / "outputs", exist_ok=True)

# File paths
CANDIDATES_FILE = RAW_DATA_DIR / "candidates.jsonl"
SAMPLE_CANDIDATES_FILE = RAW_DATA_DIR / "sample_candidates.json"
JOB_DESCRIPTION_FILE = RAW_DATA_DIR / "job_description.docx"
VALIDATOR_SCRIPT = RAW_DATA_DIR / "validate_submission.py"

# Pre-computed artifact paths
FAISS_INDEX_PATH = PROCESSED_DATA_DIR / "candidates_faiss.index"
CANDIDATE_METADATA_PATH = PROCESSED_DATA_DIR / "candidate_metadata.pkl"
VECTOR_METADATA_PATH = PROCESSED_DATA_DIR / "vector_metadata.pkl"
SKILL_GRAPH_PATH = PROCESSED_DATA_DIR / "skill_graph.json"
LTR_MODEL_PATH = PROCESSED_DATA_DIR / "ltr_model.txt"
CACHE_DIR = PROCESSED_DATA_DIR / "cache"

# Funnel Configuration
STAGE_1_RETRIEVAL_TOP_K = 1000  # Number of candidates retrieved from vector + hard filters
STAGE_1_LTR_TOP_K = 50          # Number of candidates sent to Stage 2 LLM deep re-rank
FINAL_RANKED_TOP_N = 100        # Final shortlist size for the recruiter / submission

# Model Configurations
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, highly efficient local CPU model
# BGE-large or E5 can be used, but all-MiniLM-L6-v2 is extremely fast and fits in 5 min budget.
# Let's write the code to support switching to other models easily.

# Hard Filter (Stage 0) Configurations
MIN_EXPERIENCE_YEARS = 3.0       # Experience floor (JD is 5-9 but open to strong 4+)
ALLOWED_COUNTRIES = ["India"]    # Target Indian talent pool
PREFER_LOCATIONS = ["pune", "noida", "hyderabad", "bangalore", "mumbai", "delhi", "gurgaon", "ncr"]
DISQUALIFIED_TITLES = [
    "marketing manager", "civil engineer", "graphic designer", "sales executive", 
    "operations manager", "accountant", "customer support", "mechanical engineer", 
    "electrical engineer", "chemical engineer", "hr manager", "recruiter", 
    "ui/ux designer", "product designer", "content writer", "social media",
    "marketing specialist", "sales manager", "business development"
]

# Skill Adjacency Configurations
SKILL_ADJACENCY_THRESHOLD = 0.6  # Similarity threshold for adjacent skills

# Multi-vector Default Weights (Dynamically adjusted based on JD analysis)
DEFAULT_SKILLS_WEIGHT = 0.4
DEFAULT_TRAJECTORY_WEIGHT = 0.3
DEFAULT_PROJECTS_WEIGHT = 0.3

# Behavioral Signal Weights (Multipliers)
BEHAVIORAL_WEIGHTS = {
    "open_to_work_multiplier": 1.15,
    "last_active_max_months": 6,
    "last_active_inactive_penalty": 0.8,
    "min_recruiter_response_rate": 0.3,
    "response_rate_penalty_floor": 0.85,
    "notice_period_preferred_days": 30,
    "notice_period_penalty_floor": 0.9,
    "interview_ghosting_penalty_floor": 0.8,
}

# LLM Re-ranker Configuration
LLM_PROVIDER = "gemini"  # "gemini", "openai" or "anthropic"
LLM_MODEL_NAME = "gemini-1.5-flash"
MAX_CONCURRENT_LLM_CALLS = 5
COOLDOWN_BETWEEN_BATCHES = 0.1
