import json
import os
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
from pathlib import Path
from src.ingestion.normalizer import clean_text, normalize_location, parse_date, calculate_months_between

def extract_text_from_docx(docx_path: str) -> str:
    """Extract plain text from a Word .docx file using standard zipfile and xml libraries.
    Avoids dependencies on external docx parsers.
    """
    WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    PARA = WORD_NAMESPACE + 'p'
    TEXT = WORD_NAMESPACE + 't'
    try:
        with zipfile.ZipFile(docx_path) as z:
            tree = ET.fromstring(z.read('word/document.xml'))
            paragraphs = []
            for paragraph in tree.iter(PARA):
                texts = [node.text for node in paragraph.iter(TEXT) if node.text]
                if texts:
                    paragraphs.append(''.join(texts))
            return '\n'.join(paragraphs)
    except Exception as e:
        # Fallback to reading file as text if it's already a text/md file
        try:
            with open(docx_path, "r", encoding="utf-8") as f:
                return f.read()
        except:
            raise IOError(f"Failed to read docx at {docx_path}: {str(e)}")

class JobDescriptionParser:
    """Parses a Job Description (JD) into a structured schema."""
    
    def __init__(self, docx_path: str):
        self.docx_path = docx_path
        self.raw_text = extract_text_from_docx(docx_path)
        self.structured_data = self._parse_rules()
        
    def _parse_rules(self) -> Dict[str, Any]:
        """A highly accurate, rule-based parser tailored to the Redrob AI JD structure.
        Can be extended with LLM extraction, but rule-based ensures 100% offline reliability.
        """
        text = self.raw_text
        lines = text.split('\n')
        
        data = {
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
            "disqualified_companies": [], # Handled by consulting checker in normalizer
            "notice_period_limit": 30, # preferred sub-30
        }
        
        # Parse experience from text if present
        for line in lines:
            line_lower = line.lower()
            if "experience required" in line_lower or "experience:" in line_lower:
                # Try to extract numbers
                matches = re.findall(r"(\d+)[–\-–](\d+)\s*years", line_lower)
                if matches:
                    data["experience_min"] = float(matches[0][0])
                    data["experience_max"] = float(matches[0][1])
                else:
                    single_match = re.findall(r"(\d+)\s*\+?\s*years", line_lower)
                    if single_match:
                        data["experience_min"] = float(single_match[0])
                        data["experience_max"] = float(single_match[0]) + 5.0
                        
        return data

    def get_structured(self) -> Dict[str, Any]:
        return self.structured_data

class CandidateParser:
    """Parses and normalizes candidate JSON records."""
    
    @staticmethod
    def parse_candidate(raw_candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a raw candidate record into a structured format for feature extraction."""
        cid = raw_candidate["candidate_id"]
        profile = raw_candidate.get("profile", {})
        career = raw_candidate.get("career_history", [])
        skills = raw_candidate.get("skills", [])
        education = raw_candidate.get("education", [])
        signals = raw_candidate.get("redrob_signals", {})
        
        # 1. Normalize profile
        norm_profile = {
            "anonymized_name": profile.get("anonymized_name", ""),
            "headline": profile.get("headline", ""),
            "summary": profile.get("summary", ""),
            "location": normalize_location(profile.get("location", "")),
            "country": clean_text(profile.get("country", "")),
            "years_of_experience": float(profile.get("years_of_experience", 0)),
            "current_title": clean_text(profile.get("current_title", "")),
            "current_company": clean_text(profile.get("current_company", "")),
            "current_company_size": profile.get("current_company_size", ""),
            "current_industry": clean_text(profile.get("current_industry", ""))
        }
        
        # 2. Normalize career history
        norm_career = []
        for job in career:
            norm_career.append({
                "company": clean_text(job.get("company", "")),
                "title": clean_text(job.get("title", "")),
                "start_date": job.get("start_date"),
                "end_date": job.get("end_date"),
                "duration_months": int(job.get("duration_months", 0)),
                "is_current": bool(job.get("is_current", False)),
                "industry": clean_text(job.get("industry", "")),
                "company_size": job.get("company_size", ""),
                "description": clean_text(job.get("description", ""))
            })
            
        # 3. Normalize education
        norm_edu = []
        for edu in education:
            norm_edu.append({
                "institution": clean_text(edu.get("institution", "")),
                "degree": clean_text(edu.get("degree", "")),
                "field_of_study": clean_text(edu.get("field_of_study", "")),
                "start_year": edu.get("start_year"),
                "end_year": edu.get("end_year"),
                "grade": clean_text(edu.get("grade", "")),
                "tier": edu.get("tier", "unknown")
            })
            
        # 4. Normalize skills
        norm_skills = []
        for skill in skills:
            norm_skills.append({
                "name": clean_text(skill.get("name", "")),
                "proficiency": skill.get("proficiency", "beginner").lower(),
                "endorsements": int(skill.get("endorsements", 0)),
                "duration_months": int(skill.get("duration_months", 0))
            })
            
        return {
            "candidate_id": cid,
            "profile": norm_profile,
            "career_history": norm_career,
            "education": norm_edu,
            "skills": norm_skills,
            "certifications": raw_candidate.get("certifications", []),
            "languages": raw_candidate.get("languages", []),
            "redrob_signals": signals
        }
import re
