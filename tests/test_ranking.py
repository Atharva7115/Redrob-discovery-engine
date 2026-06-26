import pytest
from src.ranking.skill_graph import SkillAdjacencyGraph
from src.ranking.features import CandidateFeatureExtractor

@pytest.fixture
def skill_graph():
    return SkillAdjacencyGraph()

def test_skill_graph_similarity(skill_graph):
    # Identical skills
    assert skill_graph.get_similarity("qdrant", "qdrant") == 1.0
    assert skill_graph.get_similarity("Python", "  python ") == 1.0
    
    # Adjacent skills in the same cluster (vector_search)
    assert skill_graph.get_similarity("qdrant", "milvus") == 0.75
    assert skill_graph.get_similarity("pinecone", "faiss") == 0.75
    
    # Adjacent skills in the same cluster (nlp_embeddings)
    assert skill_graph.get_similarity("embeddings", "sentence-transformers") == 0.75
    
    # Adjacent skills in the same cluster (llm_fine_tuning)
    assert skill_graph.get_similarity("lora", "qlora") == 0.75
    
    # Substring match fallback
    assert skill_graph.get_similarity("fine-tuning", "fine-tuning llms") == 0.6
    
    # Unrelated skills
    assert skill_graph.get_similarity("qdrant", "photoshop") == 0.0
    assert skill_graph.get_similarity("python", "salesforce") == 0.0

def test_feature_extraction(clean_candidate):
    jd_data = {
        "required_skills": ["python", "embeddings", "faiss"],
        "preferred_skills": ["fine-tuning", "lora"],
        "experience_min": 5.0,
        "experience_max": 9.0
    }
    
    sims = {
        "skills": 0.8,
        "trajectory": 0.7,
        "projects": 0.75
    }
    
    features = CandidateFeatureExtractor.extract_features(clean_candidate, jd_data, sims)
    
    # Verify features are extracted correctly
    assert features["sem_skills_sim"] == 0.8
    assert features["sem_trajectory_sim"] == 0.7
    assert features["sem_projects_sim"] == 0.75
    
    assert features["years_of_experience"] == 6.0
    assert features["yoe_fit_score"] == 1.0  # 6.0 is in 5-9 sweet spot
    
    assert features["is_currently_employed"] == 1.0
    assert features["only_consulting_services"] == 0.0  # Acme Corp is not consulting
    assert features["only_academic_research"] == 0.0
    
    # Verify skills overlap
    assert features["direct_skills_match_count"] >= 1.0  # Python matches
    assert features["has_embeddings_retrieval"] == 0.0  # NLP is there, but no embeddings keyword in mock skills list
    
    # Verify composite score is reasonable
    h_score = CandidateFeatureExtractor.compute_composite_heuristics_score(features)
    assert 0.0 <= h_score <= 1.0
    assert h_score > 0.5  # High quality candidate should score high
