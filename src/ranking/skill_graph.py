import json
import os
from typing import Dict, Set, List, Optional
from src.config import SKILL_GRAPH_PATH

# Define domain clusters of adjacent/related skills
SKILL_CLUSTERS = {
    "vector_search": {
        "faiss", "qdrant", "milvus", "pinecone", "weaviate", "opensearch", "elasticsearch",
        "hybrid search", "vector search", "information retrieval", "ir", "bm25", "dense retrieval",
        "search engines", "retrieval systems", "valves", "chromadb"
    },
    "nlp_embeddings": {
        "nlp", "natural language processing", "embeddings", "sentence-transformers", "bert",
        "bge", "e5", "word2vec", "text embeddings", "transformers", "text classification",
        "semantic search", "named entity recognition", "ner", "topic modeling"
    },
    "llm_fine_tuning": {
        "llms", "large language models", "fine-tuning", "lora", "qlora", "peft", "prompt engineering",
        "langchain", "llamaindex", "gpt", "claude", "rlhf", "dpo", "rag", "retrieval-augmented generation"
    },
    "ml_ranking": {
        "machine learning", "ml", "applied ml", "ranking", "learning-to-rank", "ltr", "lambdarank",
        "lightgbm", "xgboost", "scikit-learn", "sklearn", "statistical modeling", "classification",
        "regression", "deep learning", "neural networks", "pytorch", "tensorflow", "keras"
    },
    "data_infra": {
        "python", "sql", "spark", "pyspark", "airflow", "kafka", "data pipelines", "backend",
        "data engineering", "aws", "gcp", "docker", "kubernetes", "fastapi", "flask", "django",
        "snowflake", "databricks", "dbt", "pandas", "numpy"
    }
}

class SkillAdjacencyGraph:
    """Manages a skill ontology / adjacency graph to compute semantic skill matches."""
    
    def __init__(self):
        self.clusters = SKILL_CLUSTERS
        # Build a reverse lookup mapping: skill -> set of cluster_names
        self.skill_to_clusters: Dict[str, Set[str]] = {}
        for cluster_name, skills in self.clusters.items():
            for skill in skills:
                if skill not in self.skill_to_clusters:
                    self.skill_to_clusters[skill] = set()
                self.skill_to_clusters[skill].add(cluster_name)
                
    def get_similarity(self, skill_a: str, skill_b: str) -> float:
        """Compute similarity between two skill names (0.0 to 1.0).
        1.0 if identical.
        0.75 if they belong to the same cluster.
        0.0 otherwise.
        """
        s_a = skill_a.lower().strip()
        s_b = skill_b.lower().strip()
        
        if s_a == s_b:
            return 1.0
            
        # Check if they share any cluster
        clusters_a = self.skill_to_clusters.get(s_a, set())
        clusters_b = self.skill_to_clusters.get(s_b, set())
        
        if clusters_a & clusters_b:
            return 0.75
            
        # Substring matching fallback (e.g. "fine-tuning llms" vs "fine-tuning")
        if s_a in s_b or s_b in s_a:
            return 0.6
            
        return 0.0

    def get_adjacent_skills(self, skill: str) -> List[str]:
        """Return list of skills adjacent to the given skill."""
        s = skill.lower().strip()
        adjacent = set()
        clusters = self.skill_to_clusters.get(s, set())
        for c in clusters:
            adjacent.update(self.clusters[c])
        if s in adjacent:
            adjacent.remove(s)
        return list(adjacent)
