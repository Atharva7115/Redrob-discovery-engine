import os
import hashlib
import numpy as np
from typing import Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME, CACHE_DIR
import diskcache as dc

# Set up local cache for embeddings
cache = dc.Cache(str(CACHE_DIR / "embeddings"))

class CandidateEmbedder:
    """Generates multi-vector semantic embeddings for candidates."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        # Disable tokenizers parallelism warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model = SentenceTransformer(model_name)
        
    def _get_cache_key(self, text: str) -> str:
        """Generate a stable MD5 hash for cache keying."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()
        
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string, utilizing cache if available."""
        if not text.strip():
            # Return a zero vector of correct dimensions
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros(dim, dtype=np.float32)
            
        key = self._get_cache_key(text)
        if key in cache:
            return cache[key]
            
        # Generate embedding (returns numpy array)
        embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        embedding = embedding.astype(np.float32)
        
        # Cache it
        cache[key] = embedding
        return embedding
        
    def get_candidate_texts(self, candidate: Dict[str, Any]) -> Tuple[str, str, str]:
        """Extract the three separate narrative texts for multi-vector embedding:
        1. Skills/Tech-Stack narrative
        2. Career Trajectory narrative
        3. Project/Work Description narrative
        """
        profile = candidate.get("profile", {})
        career = candidate.get("career_history", [])
        skills = candidate.get("skills", [])
        
        # 1. Skills narrative
        skills_list = []
        for s in skills:
            skills_list.append(f"{s['name']} ({s['proficiency']}, {s['duration_months']} months, {s['endorsements']} endorsements)")
        skills_text = "Skills and tech-stack: " + ", ".join(skills_list) if skills_list else "No skills listed."
        
        # 2. Career trajectory narrative
        jobs_summary = []
        for i, job in enumerate(career):
            curr_tag = "Current role: " if job["is_current"] else f"Past role {i+1}: "
            jobs_summary.append(
                f"{curr_tag}{job['title']} at {job['company']} in {job['industry']} industry for {job['duration_months']} months."
            )
        traj_text = (
            f"Professional trajectory: Stated years of experience: {profile.get('years_of_experience', 0)}. "
            f"Headline: {profile.get('headline', '')}. Summary: {profile.get('summary', '')}. "
            + " ".join(jobs_summary)
        )
        
        # 3. Project and work description narrative
        projects_list = []
        for i, job in enumerate(career):
            if job["description"]:
                projects_list.append(
                    f"At {job['company']} as {job['title']}: {job['description']}"
                )
        projects_text = "Work achievements and projects: " + " ".join(projects_list) if projects_list else "No detailed work descriptions."
        
        return skills_text, traj_text, projects_text
        
    def embed_candidate(self, candidate: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the three separate embeddings for a candidate."""
        skills_txt, traj_txt, proj_txt = self.get_candidate_texts(candidate)
        
        skills_emb = self.embed_text(skills_txt)
        traj_emb = self.embed_text(traj_txt)
        proj_emb = self.embed_text(proj_txt)
        
        return skills_emb, traj_emb, proj_emb

    def embed_job_description(self, jd_data: Dict[str, Any], jd_raw_text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate three embeddings for the Job Description.
        These represent what we're looking for in skills, trajectory, and project narrative.
        """
        # 1. Skills text: what skills does the JD want?
        req_skills = ", ".join(jd_data.get("required_skills", []))
        pref_skills = ", ".join(jd_data.get("preferred_skills", []))
        jd_skills_text = f"Required skills: {req_skills}. Preferred skills: {pref_skills}."
        
        # 2. Trajectory text: what trajectory/experience is wanted?
        jd_traj_text = (
            f"Seniority and experience required: {jd_data.get('experience_min', 5)}-{jd_data.get('experience_max', 9)} years of experience. "
            f"Target role: {jd_data.get('title', 'AI Engineer')}. Ideal career trajectory: prior product-company experience, "
            f"hands-on production engineering, stability (no title-chasing, no job-hopping)."
        )
        
        # 3. Projects/Narrative text: what projects/achievements are wanted?
        # We can extract paragraphs containing "doing", "responsibilities", "production", "retrieval", etc.
        # Or just use the raw text for the narrative search, as it contains all details.
        jd_proj_text = f"Job responsibilities, projects, and work outcomes: {jd_raw_text}"
        
        skills_emb = self.embed_text(jd_skills_text)
        traj_emb = self.embed_text(jd_traj_text)
        proj_emb = self.embed_text(jd_proj_text)
        
        return skills_emb, traj_emb, proj_emb
