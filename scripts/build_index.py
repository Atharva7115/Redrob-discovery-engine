import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import CANDIDATES_FILE, FAISS_INDEX_PATH, CANDIDATE_METADATA_PATH
from src.ingestion.parser import CandidateParser
from src.embeddings.embedder import CandidateEmbedder
from src.embeddings.vector_store import CandidateVectorStore
from src.ranking.train import train_ltr_model

def build_index(sample_size: int = None, batch_size: int = 128):
    """Offline indexing pipeline:
    1. Parse candidates
    2. Generate multi-vector embeddings (skills, trajectory, projects)
    3. Add to FAISS index and save
    4. Save candidate metadata for fast query-time lookup
    5. Train LightGBM LTR model
    """
    print("=" * 60)
    print("STARTING OFFLINE INDEX BUILD")
    print("=" * 60)
    
    # 1. Load candidates
    print(f"Reading candidates from {CANDIDATES_FILE}...")
    raw_candidates = []
    
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw_candidates.append(json.loads(line))
            if sample_size and len(raw_candidates) >= sample_size:
                break
                
    total_candidates = len(raw_candidates)
    print(f"Loaded {total_candidates} candidates.")
    
    # 2. Initialize Embedder and Vector Store
    print("Initializing embedder...")
    embedder = CandidateEmbedder()
    vector_store = CandidateVectorStore(dimension=embedder.model.get_sentence_embedding_dimension())
    
    # We will accumulate metadata and texts to embed in batches
    metadata_store = {}
    
    cids = []
    skills_texts = []
    trajectory_texts = []
    projects_texts = []
    
    print("Normalizing profiles and preparing texts...")
    for raw_cand in tqdm(raw_candidates, desc="Parsing"):
        cand = CandidateParser.parse_candidate(raw_cand)
        cid = cand["candidate_id"]
        
        # Save simplified metadata for fast runtime loading
        metadata_store[cid] = cand
        
        # Get texts for embedding
        s_txt, t_txt, p_txt = embedder.get_candidate_texts(cand)
        
        cids.append(cid)
        skills_texts.append(s_txt)
        trajectory_texts.append(t_txt)
        projects_texts.append(p_txt)
        
    # 3. Generate embeddings in batches
    print(f"Generating embeddings in batches of {batch_size}...")
    
    skills_embs_list = []
    trajectory_embs_list = []
    projects_embs_list = []
    
    for i in range(0, total_candidates, batch_size):
        batch_cids = cids[i : i + batch_size]
        batch_skills = skills_texts[i : i + batch_size]
        batch_traj = trajectory_texts[i : i + batch_size]
        batch_proj = projects_texts[i : i + batch_size]
        
        # Embed batch using sentence-transformers model directly (more efficient than individual calls)
        s_embs = embedder.model.encode(batch_skills, convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size)
        t_embs = embedder.model.encode(batch_traj, convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size)
        p_embs = embedder.model.encode(batch_proj, convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size)
        
        skills_embs_list.append(s_embs)
        trajectory_embs_list.append(t_embs)
        projects_embs_list.append(p_embs)
        
        if (i + batch_size) % 1024 == 0 or (i + batch_size) >= total_candidates:
            progress = min(100.0, (i + len(batch_cids)) / total_candidates * 100)
            print(f"  Embedded {i + len(batch_cids)} / {total_candidates} candidates ({progress:.1f}%)")
            
    # Concatenate all embeddings
    skills_embs = np.vstack(skills_embs_list)
    trajectory_embs = np.vstack(trajectory_embs_list)
    projects_embs = np.vstack(projects_embs_list)
    
    # 4. Add to Vector Store
    print("Adding embeddings to FAISS index...")
    vector_store.add_candidates(cids, skills_embs, trajectory_embs, projects_embs)
    
    # 5. Save FAISS index and Metadata to disk
    print(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
    vector_store.save()
    
    print(f"Saving candidate metadata to {CANDIDATE_METADATA_PATH}...")
    with open(CANDIDATE_METADATA_PATH, "wb") as f:
        pickle.dump(metadata_store, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Index build completed successfully!")
    print("-" * 60)
    
    # 6. Train LTR model
    print("Training LightGBM Learning-to-Rank model...")
    # Train on a subset (e.g. 5000) for speed, or all if small sample
    train_limit = min(5000, total_candidates)
    train_ltr_model(train_limit)
    print("=" * 60)
    print("INDEX AND MODELS ARE FULLY BUILT AND READY FOR RUNTIME")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Candidate Search Index")
    parser.add_argument("--sample", type=int, default=None, help="Limit number of candidates to index (for fast testing)")
    parser.add_argument("--batch", type=int, default=128, help="Batch size for embedding generation")
    args = parser.parse_args()
    
    build_index(sample_size=args.sample, batch_size=args.batch)
