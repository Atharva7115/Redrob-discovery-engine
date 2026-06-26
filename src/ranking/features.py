from datetime import date
from typing import Dict, Any, List, Optional
from src.config import BEHAVIORAL_WEIGHTS
from src.ingestion.normalizer import clean_text, parse_date, is_consulting_firm, is_academic_or_research
from src.ranking.skill_graph import SkillAdjacencyGraph

# Initialize skill graph
skill_graph = SkillAdjacencyGraph()

class CandidateFeatureExtractor:
    """Extracts structural, trajectory, skill-adjacency, and behavioral features
    for a candidate against a Job Description (JD) to feed into the ranking engine.
    """
    
    @staticmethod
    def extract_features(
        candidate: Dict[str, Any], 
        jd_data: Dict[str, Any], 
        semantic_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Extract a dictionary of numerical features for a single candidate.
        All features are normalized or scaled to be suitable for ranking models.
        """
        features = {}
        
        # 1. Semantic Scores (Stage 1 Vector Similarities)
        if semantic_scores:
            features["sem_skills_sim"] = float(semantic_scores.get("skills", 0.0))
            features["sem_trajectory_sim"] = float(semantic_scores.get("trajectory", 0.0))
            features["sem_projects_sim"] = float(semantic_scores.get("projects", 0.0))
        else:
            features["sem_skills_sim"] = 0.5
            features["sem_trajectory_sim"] = 0.5
            features["sem_projects_sim"] = 0.5
            
        # 2. Basic Profile Features
        profile = candidate.get("profile", {})
        yoe = float(profile.get("years_of_experience", 0.0))
        features["years_of_experience"] = yoe
        
        # Experience fit: 5-9 years is the sweet spot. We compute a bell-curve style score.
        # Max score of 1.0 when yoe is between 5 and 9. Penalize slightly outside.
        if yoe >= 5.0 and yoe <= 9.0:
            features["yoe_fit_score"] = 1.0
        elif yoe < 5.0:
            features["yoe_fit_score"] = max(0.2, yoe / 5.0) # Down-weight junior
        else:
            # Over-senior is fine, but we scale it down slightly
            features["yoe_fit_score"] = max(0.6, 1.0 - (yoe - 9.0) * 0.05)
            
        # 3. Career Trajectory Features
        career = candidate.get("career_history", [])
        num_jobs = len(career)
        features["num_jobs"] = float(num_jobs)
        
        # Average tenure in months
        total_tenure_months = sum(job.get("duration_months", 0) for job in career)
        avg_tenure = total_tenure_months / max(1, num_jobs)
        features["avg_tenure_months"] = avg_tenure
        
        # Job hopping frequency: Switching companies every 1.5 years (18 months) or less is a negative signal
        # E.g., if avg tenure is < 18 months, job_hop_penalty is high.
        features["job_hop_penalty"] = 1.0 if avg_tenure < 18.0 else 0.0
        
        # Is currently employed
        features["is_currently_employed"] = 1.0 if any(job.get("is_current", False) for job in career) else 0.0
        
        # Check if entire career is at IT consulting services firms (e.g. TCS, Infosys, Wipro, etc.)
        all_consulting = True
        has_any_job = False
        for job in career:
            has_any_job = True
            if not is_consulting_firm(job.get("company", "")):
                all_consulting = False
                break
        features["only_consulting_services"] = 1.0 if (all_consulting and has_any_job) else 0.0
        
        # Check if entire career is in academic or research-only settings
        all_academic = True
        for job in career:
            if not is_academic_or_research(job.get("title", ""), job.get("description", "")):
                all_academic = False
                break
        features["only_academic_research"] = 1.0 if (all_academic and has_any_job) else 0.0
        
        # Promotion velocity: Check if their job titles show growth (e.g., junior -> lead)
        has_promotion = False
        titles = [clean_text(job.get("title", "")) for job in career]
        # Reverse list so it goes chronological (oldest to newest)
        titles.reverse()
        senior_keywords = ["senior", "lead", "principal", "head", "architect", "manager", "director"]
        junior_keywords = ["junior", "intern", "associate", "trainee", "fresher", "member"]
        
        had_junior = False
        for t in titles:
            if any(jk in t for jk in junior_keywords):
                had_junior = True
            if had_junior and any(sk in t for sk in senior_keywords):
                has_promotion = True
                break
        features["has_promotion_velocity"] = 1.0 if has_promotion else 0.0
        
        # 4. Skill Adjacency Features
        cand_skills = candidate.get("skills", [])
        jd_required = jd_data.get("required_skills", [])
        jd_preferred = jd_data.get("preferred_skills", [])
        
        direct_matches = 0
        adjacent_score = 0.0
        
        has_embeddings_retrieval = False
        has_vector_db = False
        has_ltr_or_fine_tune = False
        
        for cand_s in cand_skills:
            s_name = cand_s["name"].lower().strip()
            
            # Check for core domains
            if any(kw in s_name for kw in ["embedding", "retrieval", "sentence-transformer", "hybrid search"]):
                has_embeddings_retrieval = True
            if any(kw in s_name for kw in ["faiss", "qdrant", "milvus", "pinecone", "weaviate", "vector"]):
                has_vector_db = True
            if any(kw in s_name for kw in ["learning-to-rank", "ltr", "fine-tuning", "lora", "qlora", "peft"]):
                has_ltr_or_fine_tune = True
                
            # Direct match check against JD required
            if s_name in jd_required:
                direct_matches += 1
                
            # Adjacency check
            max_sim = 0.0
            for jd_s in jd_required + jd_preferred:
                sim = skill_graph.get_similarity(s_name, jd_s)
                if sim > max_sim:
                    max_sim = sim
            
            # Scale by proficiency and endorsements
            prof_multiplier = {"beginner": 0.5, "intermediate": 0.8, "advanced": 1.0, "expert": 1.2}.get(
                cand_s["proficiency"], 0.8
            )
            # Add small bonus for endorsements
            endorse_bonus = min(0.2, cand_s["endorsements"] * 0.01)
            
            adjacent_score += max_sim * (prof_multiplier + endorse_bonus)
            
        features["direct_skills_match_count"] = float(direct_matches)
        features["adjacent_skills_match_score"] = adjacent_score
        features["has_embeddings_retrieval"] = 1.0 if has_embeddings_retrieval else 0.0
        features["has_vector_db"] = 1.0 if has_vector_db else 0.0
        features["has_ltr_or_fine_tune"] = 1.0 if has_ltr_or_fine_tune else 0.0
        
        # 5. Behavioral Platform Signals (`redrob_signals`)
        signals = candidate.get("redrob_signals", {})
        
        # Profile completeness (0.0 to 1.0)
        features["profile_completeness"] = float(signals.get("profile_completeness_score", 100.0)) / 100.0
        
        # Last active date: calculate months since last active
        last_active_s = signals.get("last_active_date", "")
        # Ref date is 2026-06-25
        ref_date = date(2026, 6, 25)
        try:
            last_act = parse_date(last_active_s)
            act_diff_months = (ref_date.year - last_act.year) * 12 + (ref_date.month - last_act.month)
        except:
            act_diff_months = 12 # assume inactive if error
        features["months_since_last_active"] = float(act_diff_months)
        
        # Stated availability signals
        features["open_to_work"] = 1.0 if signals.get("open_to_work_flag", False) else 0.0
        features["recruiter_response_rate"] = float(signals.get("recruiter_response_rate", 0.0))
        features["avg_response_time_hours"] = float(signals.get("avg_response_time_hours", 24.0))
        features["notice_period_days"] = float(signals.get("notice_period_days", 60.0))
        features["willing_to_relocate"] = 1.0 if signals.get("willing_to_relocate", False) else 0.0
        features["github_activity_score"] = float(signals.get("github_activity_score", -1.0))
        features["interview_completion_rate"] = float(signals.get("interview_completion_rate", 1.0))
        features["offer_acceptance_rate"] = float(signals.get("offer_acceptance_rate", -1.0))
        
        return features

    @staticmethod
    def compute_composite_heuristics_score(features: Dict[str, float]) -> float:
        """Compute a highly tuned heuristic composite score (0.0 to 1.0) using the extracted features.
        This serves as a weak-supervision label and a very robust fallback when LightGBM weights
        or LLM APIs are unavailable.
        """
        # A. Semantic Search Score (40%)
        # Skills semantic similarity is weighted higher
        semantic_score = (
            features["sem_skills_sim"] * 0.45 +
            features["sem_trajectory_sim"] * 0.30 +
            features["sem_projects_sim"] * 0.25
        )
        
        # B. Experience & Role Fit (20%)
        # Direct experience fit + promotion velocity
        experience_score = (
            features["yoe_fit_score"] * 0.70 +
            features["has_promotion_velocity"] * 0.30
        )
        
        # C. Skill Graph Match Score (25%)
        # Direct matches + adjacent matches + specific key skills check
        key_skills_match = (
            features["has_embeddings_retrieval"] * 0.40 +
            features["has_vector_db"] * 0.40 +
            features["has_ltr_or_fine_tune"] * 0.20
        )
        # Normalize skill score
        skill_match_score = min(1.0, (features["direct_skills_match_count"] * 0.15 + features["adjacent_skills_match_score"] * 0.05 + key_skills_match * 0.40))
        
        # D. Trajectory & Company Fit (15%)
        # Stability + product company experience
        trajectory_score = 1.0
        # Penalties:
        if features["job_hop_penalty"] > 0:
            trajectory_score -= 0.3 # Title-chaser penalty
        if features["only_consulting_services"] > 0:
            trajectory_score -= 0.5 # Consulting-only penalty (explicitly NOT wanted in JD)
        if features["only_academic_research"] > 0:
            trajectory_score -= 0.5 # Pure research-only penalty (explicitly NOT wanted)
        trajectory_score = max(0.0, trajectory_score)
        
        # Combine base scores (Weighted sum = 1.0)
        base_score = (
            semantic_score * 0.40 +
            experience_score * 0.20 +
            skill_match_score * 0.25 +
            trajectory_score * 0.15
        )
        
        # E. Behavioral Signals Modifier (Multiplicative adjustments)
        # Available and active candidates get a boost, inactive or ghosting candidates get a penalty
        modifier = 1.0
        
        # Active looking boost
        if features["open_to_work"] > 0:
            modifier *= BEHAVIORAL_WEIGHTS["open_to_work_multiplier"]
            
        # Inactivity penalty: last active > 6 months
        if features["months_since_last_active"] > BEHAVIORAL_WEIGHTS["last_active_max_months"]:
            # Graduated penalty down to floor
            over_months = features["months_since_last_active"] - BEHAVIORAL_WEIGHTS["last_active_max_months"]
            penalty = max(BEHAVIORAL_WEIGHTS["last_active_inactive_penalty"], 1.0 - over_months * 0.02)
            modifier *= penalty
            
        # Recruiter response rate penalty: if response rate < 30%
        resp_rate = features["recruiter_response_rate"]
        if resp_rate < BEHAVIORAL_WEIGHTS["min_recruiter_response_rate"]:
            penalty = max(BEHAVIORAL_WEIGHTS["response_rate_penalty_floor"], 0.7 + resp_rate)
            modifier *= penalty
            
        # Notice period: prefer sub-30. If notice period > 60 days, penalize slightly.
        notice = features["notice_period_days"]
        if notice > BEHAVIORAL_WEIGHTS["notice_period_preferred_days"]:
            # Graduated penalty for long notice
            penalty = max(BEHAVIORAL_WEIGHTS["notice_period_penalty_floor"], 1.0 - (notice - 30) * 0.001)
            modifier *= penalty
            
        # Interview completion rate (completion rate < 80% means they ghost, heavy penalty!)
        completion = features["interview_completion_rate"]
        if completion < 0.8:
            penalty = max(BEHAVIORAL_WEIGHTS["interview_ghosting_penalty_floor"], completion)
            modifier *= penalty
            
        # Final score = base * modifier
        final_score = base_score * modifier
        
        # Constrain to exactly 0.0 to 1.0
        return float(max(0.0, min(1.0, final_score)))
