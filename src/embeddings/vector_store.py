import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from src.config import FAISS_INDEX_PATH, VECTOR_METADATA_PATH

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length (L2 normalization) so that
    inner product searches return exact Cosine Similarity.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

class CandidateVectorStore:
    """FAISS-backed multi-vector store for candidate semantic search."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # Flat Inner Product indexes (when normalized, inner product = cosine similarity)
        self.index_skills = faiss.IndexFlatIP(dimension)
        self.index_trajectory = faiss.IndexFlatIP(dimension)
        self.index_projects = faiss.IndexFlatIP(dimension)
        
        # List of candidate IDs mapping 1-to-1 with FAISS index positions
        self.candidate_ids: List[str] = []
        
    def add_candidates(self, cids: List[str], skills_embs: np.ndarray, trajectory_embs: np.ndarray, projects_embs: np.ndarray):
        """Add candidate embeddings to the vector store.
        Vectors are normalized to unit length before adding.
        """
        assert len(cids) == len(skills_embs) == len(trajectory_embs) == len(projects_embs), "Length mismatch"
        
        # Normalize all vectors
        norm_skills = np.array([normalize_vector(v) for v in skills_embs], dtype=np.float32)
        norm_traj = np.array([normalize_vector(v) for v in trajectory_embs], dtype=np.float32)
        norm_proj = np.array([normalize_vector(v) for v in projects_embs], dtype=np.float32)
        
        # Add to FAISS indexes
        self.index_skills.add(norm_skills)
        self.index_trajectory.add(norm_traj)
        self.index_projects.add(norm_proj)
        
        # Append candidate IDs
        self.candidate_ids.extend(cids)
        
    def search_component(self, index: faiss.Index, query_emb: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Search a single FAISS index using a normalized query vector.
        Returns a list of (candidate_id, cosine_similarity) tuples.
        """
        norm_q = normalize_vector(query_emb).reshape(1, -1).astype(np.float32)
        
        # Perform search (FAISS returns distances/similarities and indices)
        similarities, indices = index.search(norm_q, top_k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1 and idx < len(self.candidate_ids):
                results.append((self.candidate_ids[idx], float(sim)))
        return results

    def search_hybrid_semantic(
        self, 
        q_skills: np.ndarray, 
        q_trajectory: np.ndarray, 
        q_projects: np.ndarray, 
        top_k: int = 1000,
        skills_weight: float = 0.4,
        trajectory_weight: float = 0.3,
        projects_weight: float = 0.3
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """Search across all three embedding vectors and combine the results.
        Returns a list of (candidate_id, weighted_score, component_scores) sorted by score descending.
        """
        # Since we want a robust union of candidate scores, we search each index for the top candidates
        # and then combine the scores. We search for a slightly larger K to ensure high recall in the union.
        search_k = min(len(self.candidate_ids), top_k * 2)
        if search_k == 0:
            return []
            
        res_skills = self.search_component(self.index_skills, q_skills, search_k)
        res_traj = self.search_component(self.index_trajectory, q_trajectory, search_k)
        res_proj = self.search_component(self.index_projects, q_projects, search_k)
        
        # Build a mapping of candidate_id -> {component: score}
        score_map: Dict[str, Dict[str, float]] = {}
        
        for cid, sim in res_skills:
            if cid not in score_map:
                score_map[cid] = {"skills": 0.0, "trajectory": 0.0, "projects": 0.0}
            score_map[cid]["skills"] = sim
            
        for cid, sim in res_traj:
            if cid not in score_map:
                score_map[cid] = {"skills": 0.0, "trajectory": 0.0, "projects": 0.0}
            score_map[cid]["trajectory"] = sim
            
        for cid, sim in res_proj:
            if cid not in score_map:
                score_map[cid] = {"skills": 0.0, "trajectory": 0.0, "projects": 0.0}
            score_map[cid]["projects"] = sim
            
        # Compute weighted score for each candidate in the union
        final_scores = []
        for cid, comps in score_map.items():
            weighted_score = (
                comps["skills"] * skills_weight +
                comps["trajectory"] * trajectory_weight +
                comps["projects"] * projects_weight
            )
            final_scores.append((cid, weighted_score, comps))
            
        # Sort by weighted score descending
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:top_k]
        
    def save(self, index_path: str = str(FAISS_INDEX_PATH), meta_path: str = str(VECTOR_METADATA_PATH)):
        """Save the FAISS indexes and candidate IDs to disk."""
        # Save FAISS indices
        faiss.write_index(self.index_skills, index_path + ".skills")
        faiss.write_index(self.index_trajectory, index_path + ".trajectory")
        faiss.write_index(self.index_projects, index_path + ".projects")
        
        # Save candidate IDs
        with open(meta_path, "wb") as f:
            pickle.dump(self.candidate_ids, f)
            
    def load(self, index_path: str = str(FAISS_INDEX_PATH), meta_path: str = str(VECTOR_METADATA_PATH)) -> bool:
        """Load the FAISS indexes and candidate IDs from disk."""
        if not (os.path.exists(index_path + ".skills") and os.path.exists(meta_path)):
            return False
            
        try:
            # Load FAISS indices
            self.index_skills = faiss.read_index(index_path + ".skills")
            self.index_trajectory = faiss.read_index(index_path + ".trajectory")
            self.index_projects = faiss.read_index(index_path + ".projects")
            
            # Load candidate IDs
            with open(meta_path, "rb") as f:
                self.candidate_ids = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
