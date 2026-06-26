import json
import numpy as np
from typing import List, Dict, Any
from src.config import CANDIDATES_FILE, SAMPLE_CANDIDATES_FILE
from src.ingestion.parser import CandidateParser
from src.ranking.features import CandidateFeatureExtractor
from src.ranking.ltr_model import LearningToRankModel

def train_ltr_model(sample_limit: int = 5000):
    """Generate training data from candidate pool using weak-supervision,
    and train the LightGBM LambdaMART ranker.
    """
    print("Preparing training data...")
    
    # We will load candidates. First try sample_candidates for safety, then candidates.jsonl
    candidates_to_use = []
    
    # Since candidates.jsonl is large, we can read the first `sample_limit` records
    # representing a diverse set of candidates.
    count = 0
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            candidates_to_use.append(json.loads(line))
            count += 1
            if count >= sample_limit:
                break
                
    print(f"Loaded {len(candidates_to_use)} candidates for training.")
    
    # Standardize JD data (what our parser would return for the Senior AI Engineer role)
    jd_data = {
        "title": "Senior AI Engineer — Founding Team",
        "company": "Redrob AI",
        "locations": ["pune", "noida"],
        "experience_min": 5.0,
        "experience_max": 9.0,
        "required_skills": [
            "embeddings", "retrieval", "ranking", "vector databases", "hybrid search",
            "sentence-transformers", "faiss", "qdrant", "milvus", "pinecone", "weaviate",
            "opensearch", "elasticsearch", "python", "evaluation", "ndcg", "mrr", "map"
        ],
        "preferred_skills": [
            "fine-tuning", "lora", "qlora", "peft", "learning-to-rank", "xgboost", 
            "lightgbm", "distributed systems", "inference optimization"
        ],
        "disqualified_titles": ["marketing manager", "civil engineer", "graphic designer", "sales executive", "operations manager", "accountant", "customer support"],
        "notice_period_limit": 30
    }
    
    X_list = []
    y_list = []
    
    # We will initialize the LTR model to get the feature names in order
    ltr_model = LearningToRankModel()
    feature_names = ltr_model.feature_names
    
    print("Extracting features and generating weak-supervision labels...")
    for raw_cand in candidates_to_use:
        # Parse and normalize candidate
        cand = CandidateParser.parse_candidate(raw_cand)
        
        # Extract features
        # Note: we don't have the semantic similarities pre-computed for all during training,
        # so we compute structural features, and we can simulate a variety of semantic scores
        # or use our feature extractor. To make the model robust, we simulate semantic scores
        # that correlate with their skill matches.
        
        # Compute skill overlap to simulate semantic similarity
        skills = [s["name"].lower().strip() for s in cand["skills"]]
        overlap = len(set(skills) & set(jd_data["required_skills"]))
        sim_skills = min(0.9, 0.2 + overlap * 0.1)
        
        # Simulate trajectory similarity based on years of experience and company size
        sim_traj = min(0.9, 0.3 + float(cand["profile"]["years_of_experience"]) * 0.04)
        
        # Simulate projects similarity
        sim_proj = min(0.9, 0.2 + len(cand["career_history"]) * 0.1)
        
        semantic_scores = {
            "skills": sim_skills,
            "trajectory": sim_traj,
            "projects": sim_proj
        }
        
        features = CandidateFeatureExtractor.extract_features(cand, jd_data, semantic_scores)
        
        # Compute weak-supervision score (0.0 to 1.0)
        h_score = CandidateFeatureExtractor.compute_composite_heuristics_score(features)
        
        # Bucket into 5 relevance grades (0 to 4) for LambdaMART
        if h_score >= 0.85:
            label = 4  # Perfect fit
        elif h_score >= 0.70:
            label = 3  # Good fit
        elif h_score >= 0.50:
            label = 2  # Acceptable fit
        elif h_score >= 0.30:
            label = 1  # Unlikely fit
        else:
            label = 0  # Poor/disqualified
            
        # Append to lists
        x_vector = [features[f] for f in feature_names]
        X_list.append(x_vector)
        y_list.append(label)
        
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    # In LightGBM LambdaRank, we need to specify query groups.
    # Since we have a single query (our JD), all samples belong to one group.
    group = np.array([len(X_list)], dtype=np.int32)
    
    print(f"Training LightGBM model on {len(X_list)} samples...")
    ltr_model.train(X, y, group)
    print("Training completed!")

if __name__ == "__main__":
    train_ltr_model(5000)
