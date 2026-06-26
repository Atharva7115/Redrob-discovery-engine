import os
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import lightgbm as lgb
from src.config import LTR_MODEL_PATH
from src.ranking.features import CandidateFeatureExtractor

class LearningToRankModel:
    """Manages training, loading, and inference of the LightGBM Learning-to-Rank (LTR) model."""
    
    def __init__(self, model_path: str = str(LTR_MODEL_PATH)):
        self.model_path = model_path
        self.ranker: Optional[lgb.LGBMRanker] = None
        self.feature_names = [
            "sem_skills_sim", "sem_trajectory_sim", "sem_projects_sim",
            "years_of_experience", "yoe_fit_score", "num_jobs",
            "avg_tenure_months", "job_hop_penalty", "is_currently_employed",
            "only_consulting_services", "only_academic_research", "has_promotion_velocity",
            "direct_skills_match_count", "adjacent_skills_match_score",
            "has_embeddings_retrieval", "has_vector_db", "has_ltr_or_fine_tune",
            "profile_completeness", "months_since_last_active", "open_to_work",
            "recruiter_response_rate", "avg_response_time_hours", "notice_period_days",
            "willing_to_relocate", "github_activity_score", "interview_completion_rate",
            "offer_acceptance_rate"
        ]
        
    def load_model(self) -> bool:
        """Load the trained LightGBM model from disk."""
        if not os.path.exists(self.model_path):
            return False
            
        try:
            # LightGBM models can be saved/loaded via text file
            self.ranker = lgb.Booster(model_file=self.model_path)
            return True
        except Exception as e:
            print(f"Failed to load LightGBM model: {e}")
            return False
            
    def predict_score(self, features: Dict[str, float]) -> float:
        """Predict the relevance score for a candidate based on their features.
        Falls back to the composite heuristic score if the LightGBM model is not loaded.
        """
        # If LightGBM model is loaded, use it to predict the LTR score
        if self.ranker is not None:
            try:
                # Prepare input feature vector in correct order
                x = np.array([[features[f] for f in self.feature_names]], dtype=np.float32)
                # Booster.predict returns scores
                score = float(self.ranker.predict(x)[0])
                # Sigmoid to map score to 0-1 range
                return float(1.0 / (1.0 + np.exp(-score)))
            except Exception as e:
                # Fallback on error
                pass
                
        # Fallback to the highly tuned heuristic score
        return CandidateFeatureExtractor.compute_composite_heuristics_score(features)
        
    def train(self, X: np.ndarray, y: np.ndarray, group: np.ndarray):
        """Train the LightGBM LambdaMART ranking model.
        - X: Feature matrix of shape (n_samples, n_features)
        - y: Relevance labels (e.g. 0 to 4)
        - group: Query group sizes (e.g. [size_group1, size_group2, ...])
        """
        # Create LGBMRanker
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[5, 10, 50],
            boosting_type="gbdt",
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        
        # Fit ranker
        ranker.fit(
            X, y,
            group=group,
            feature_name=self.feature_names
        )
        
        # Save booster to file
        ranker.booster_.save_model(self.model_path)
        self.ranker = ranker.booster_
        print(f"LightGBM LTR model trained and saved successfully to {self.model_path}")
