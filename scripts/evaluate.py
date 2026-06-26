import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import CANDIDATES_FILE, CANDIDATE_METADATA_PATH
from src.ingestion.parser import JobDescriptionParser, CandidateParser
from src.ranking.features import CandidateFeatureExtractor
from src.ranking.ltr_model import LearningToRankModel
from src.embeddings.embedder import CandidateEmbedder
from src.embeddings.vector_store import CandidateVectorStore

def dcg_at_k(r: np.ndarray, k: int) -> float:
    """Calculate Discounted Cumulative Gain (DCG) at K."""
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0

def ndcg_at_k(r: np.ndarray, k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG) at K."""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max

def mean_reciprocal_rank(rs: List[np.ndarray]) -> float:
    """Calculate Mean Reciprocal Rank (MRR)."""
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def average_precision(r: np.ndarray) -> float:
    """Calculate Average Precision (AP)."""
    r = np.asfarray(r) != 0
    out = [np.mean(r[:i + 1]) for i in range(r.size) if r[i]]
    if not out:
        return 0.0
    return np.mean(out)

def mean_average_precision(rs: List[np.ndarray]) -> float:
    """Calculate Mean Average Precision (MAP)."""
    return np.mean([average_precision(r) for r in rs])

def evaluate_pipeline(eval_limit: int = 1000):
    """Run evaluation metrics (NDCG, MRR, MAP) of our ranking pipeline
    against the weak-supervision ground truth.
    """
    print("=" * 60)
    print("RUNNING PIPELINE EVALUATION")
    print("=" * 60)
    
    # 1. Load candidates
    print("Loading candidate metadata...")
    if not os.path.exists(CANDIDATE_METADATA_PATH):
        print(f"Error: Candidate metadata pkl not found at {CANDIDATE_METADATA_PATH}.")
        print("Please run scripts/build_index.py first to build the index and train the model.")
        return
        
    with open(CANDIDATE_METADATA_PATH, "rb") as f:
        candidate_metadata = pickle.load(f)
        
    cids = list(candidate_metadata.keys())
    eval_cids = cids[:eval_limit]
    print(f"Evaluating on a sample of {len(eval_cids)} candidates.")
    
    # JD Data
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
    
    # Load LTR Model
    ltr_model = LearningToRankModel()
    model_loaded = ltr_model.load_model()
    print(f"LightGBM Model Loaded: {model_loaded}")
    
    # Generate simulated semantic scores for evaluation
    # Compute "ground truth" (heuristic score) and LTR predictions
    gt_scores = []
    pred_scores = []
    
    for cid in eval_cids:
        cand = candidate_metadata[cid]
        
        # Simulate semantic similarities based on skill overlap
        skills = [s["name"].lower().strip() for s in cand["skills"]]
        overlap = len(set(skills) & set(jd_data["required_skills"]))
        sim_skills = min(0.9, 0.2 + overlap * 0.1)
        sim_traj = min(0.9, 0.3 + float(cand["profile"]["years_of_experience"]) * 0.04)
        sim_proj = min(0.9, 0.2 + len(cand["career_history"]) * 0.1)
        
        sims = {"skills": sim_skills, "trajectory": sim_traj, "projects": sim_proj}
        
        features = CandidateFeatureExtractor.extract_features(cand, jd_data, sims)
        
        # Ground Truth is the exact composite heuristic score (our target)
        gt_score = CandidateFeatureExtractor.compute_composite_heuristics_score(features)
        gt_scores.append((cid, gt_score))
        
        # Prediction is what our LTR model outputs
        pred_score = ltr_model.predict_score(features)
        pred_scores.append((cid, pred_score))
        
    # Sort both to get rankings
    gt_scores.sort(key=lambda x: x[1], reverse=True)
    pred_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Map candidate_id -> ground truth score
    gt_map = dict(gt_scores)
    
    # Calculate binary relevance (relevance >= 0.5 is "relevant", 1, else 0)
    # and graded relevance (0 to 4 based on gt score)
    graded_relevance = []
    binary_relevance = []
    
    for cid, _ in pred_scores:
        gt_val = gt_map[cid]
        
        # Graded relevance (0 to 4)
        if gt_val >= 0.85:
            r_grade = 4
        elif gt_val >= 0.70:
            r_grade = 3
        elif gt_val >= 0.50:
            r_grade = 2
        elif gt_val >= 0.30:
            r_grade = 1
        else:
            r_grade = 0
            
        graded_relevance.append(r_grade)
        binary_relevance.append(1 if r_grade >= 2 else 0)
        
    # Compute metrics
    ndcg_10 = ndcg_at_k(graded_relevance, 10)
    ndcg_50 = ndcg_at_k(graded_relevance, 50)
    p_10 = np.mean(binary_relevance[:10])
    p_50 = np.mean(binary_relevance[:50])
    
    # MAP and MRR
    mrr = mean_reciprocal_rank([binary_relevance])
    map_score = mean_average_precision([binary_relevance])
    
    print("-" * 60)
    print("EVALUATION RESULTS (Against Weak-Supervision Target)")
    print("-" * 60)
    print(f"NDCG@10:  {ndcg_10:.4f}  (Measures top-10 ranking quality)")
    print(f"NDCG@50:  {ndcg_50:.4f}  (Measures top-50 ranking quality)")
    print(f"Precision@10 (Tier 2+): {p_10:.4f} ({int(p_10*10)}/10 relevant)")
    print(f"Precision@50 (Tier 2+): {p_50:.4f} ({int(p_50*50)}/50 relevant)")
    print(f"MAP:      {map_score:.4f}  (Mean Average Precision)")
    print(f"MRR:      {mrr:.4f}  (Mean Reciprocal Rank)")
    
    # Composite score as defined by hackathon rules:
    # 0.50 * NDCG@10 + 0.30 * NDCG@50 + 0.15 * MAP + 0.05 * P@10
    composite = 0.50 * ndcg_10 + 0.30 * ndcg_50 + 0.15 * map_score + 0.05 * p_10
    print(f"Final Composite Score:  {composite:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    import os
    evaluate_pipeline(1000)
