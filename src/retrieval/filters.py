from datetime import date
from typing import Dict, Any, List, Tuple
from src.config import MIN_EXPERIENCE_YEARS, ALLOWED_COUNTRIES, DISQUALIFIED_TITLES
from src.ingestion.normalizer import parse_date

def is_honeypot_duration_anomaly(candidate: Dict[str, Any]) -> bool:
    """Check if the candidate has an impossible job duration in their career history.
    (E.g. claiming 166 months at a job that started in 2023).
    """
    ref_date = date(2026, 6, 25)
    career = candidate.get("career_history", [])
    
    for job in career:
        start_s = job.get("start_date")
        end_s = job.get("end_date")
        dur = job.get("duration_months", 0)
        
        if start_s:
            try:
                start_dt = parse_date(start_s)
                # If it is the current job (end_date is null), use the reference date
                end_dt = parse_date(end_s) if end_s else ref_date
                
                # Start date cannot be after end date
                if start_dt > end_dt:
                    return True
                    
                # Calculate maximum physically possible elapsed months
                elapsed_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 2
                
                # If stated duration is longer than the physical time the job existed by more than 12 months,
                # and the stated duration is substantial (e.g., > 12 months), it's a physical contradiction.
                if dur > elapsed_months + 12 and dur > 12:
                    return True
            except Exception:
                pass
    return False

def is_honeypot_skill_anomaly(candidate: Dict[str, Any]) -> bool:
    """Check if the candidate claims expert/advanced proficiency in skills but has 0 months of experience.
    Honeypots in the dataset have exactly 5 such skills.
    """
    skills = candidate.get("skills", [])
    expert_zero_count = 0
    
    for s in skills:
        dur = s.get("duration_months", 0)
        prof = s.get("proficiency", "").lower()
        if prof in ["expert", "advanced"] and dur == 0:
            expert_zero_count += 1
            
    # If a candidate has 3 or more expert/advanced skills with 0 months of experience, it's a honeypot
    return expert_zero_count >= 3

def is_honeypot(candidate: Dict[str, Any]) -> bool:
    """Combines all honeypot checks. Returns True if candidate is a honeypot."""
    return is_honeypot_duration_anomaly(candidate) or is_honeypot_skill_anomaly(candidate)

def apply_hard_filters(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """Apply Stage 0 hard filters.
    Returns (passed, reason_for_failure).
    """
    # 1. Filter out Honeypots immediately
    if is_honeypot(candidate):
        return False, "Honeypot: Profile contains impossible contradictions"
        
    profile = candidate.get("profile", {})
    
    # 2. Years of experience floor
    yoe = float(profile.get("years_of_experience", 0))
    if yoe < MIN_EXPERIENCE_YEARS:
        return False, f"Experience floor: Has {yoe} years, requires at least {MIN_EXPERIENCE_YEARS}"
        
    # 3. Location / Visa sponsorship filter
    # The JD states: "Pune/Noida, India (Hybrid) | Open to relocation | Outside India: case-by-case, but we don't sponsor work visas"
    # So we filter out candidates outside India who are unwilling to relocate.
    country = profile.get("country", "").lower().strip()
    willing_relocate = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)
    
    # If country is not India, they are outside India and require visa sponsorship (not sponsored)
    # We filter them out unless they are in India.
    if country and country != "india":
        # Check if they are in India despite the country field, or if they have a clear path
        # If they are outside India, we filter them.
        return False, f"Visa restriction: Located in {profile.get('country')}, no visa sponsorship available"
        
    # 4. Filter out Disqualified Titles
    headline = profile.get("headline", "").lower()
    current_title = profile.get("current_title", "").lower()
    
    for title in DISQUALIFIED_TITLES:
        title_lower = title.lower()
        if title_lower in headline or title_lower in current_title:
            return False, f"Disqualified title: Has title/headline matching '{title}'"
            
    return True, ""

def run_stage_0(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run all candidates through Stage 0 filters.
    Returns only candidates that passed.
    """
    passed_candidates = []
    for c in candidates:
        passed, _ = apply_hard_filters(c)
        if passed:
            passed_candidates.append(c)
    return passed_candidates
