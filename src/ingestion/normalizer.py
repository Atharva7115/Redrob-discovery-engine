import re
from datetime import datetime, date

def clean_text(text: str) -> str:
    """Lowercase, strip, and remove excessive whitespace from text."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def normalize_location(location: str) -> str:
    """Normalize location strings to a canonical lowercase representation."""
    if not location:
        return "unknown"
    loc = clean_text(location)
    # Map common variations to canonical cities
    if "bengaluru" in loc or "bangalore" in loc:
        return "bangalore"
    if "noida" in loc or "delhi" in loc or "gurgaon" in loc or "ncr" in loc or "ghaziabad" in loc or "faridabad" in loc:
        return "noida" # Noida/Delhi NCR
    if "pune" in loc:
        return "pune"
    if "hyderabad" in loc:
        return "hyderabad"
    if "mumbai" in loc or "navi mumbai" in loc or "thane" in loc:
        return "mumbai"
    if "chennai" in loc:
        return "chennai"
    if "kolkata" in loc:
        return "kolkata"
    return loc

def parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format, returning a date object.
    Falls back to a default date if parsing fails.
    """
    if not date_str:
        return date(2026, 6, 25) # Default reference date
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        try:
            # Try YYYY-MM format
            return datetime.strptime(date_str.strip(), "%Y-%m").date()
        except ValueError:
            # Fallback
            return date(2026, 6, 25)

def calculate_months_between(start_date_str: str, end_date_str: str = None) -> int:
    """Calculate the number of months between two date strings."""
    start = parse_date(start_date_str)
    # If end date is not provided, use the reference date (June 25, 2026)
    end = parse_date(end_date_str) if end_date_str else date(2026, 6, 25)
    
    months = (end.year - start.year) * 12 + (end.month - start.month)
    return max(0, months)

def is_consulting_firm(company_name: str) -> bool:
    """Check if a company is a known IT consulting services firm (TCS, Infosys, etc.)."""
    if not company_name:
        return False
    name = clean_text(company_name)
    consulting_firms = [
        "tcs", "tata consultancy services", "infosys", "wipro", "accenture", 
        "cognizant", "cts", "capgemini", "tech mahindra", "mindtree", 
        "mphasis", "l&t", "larsen & toubro", "ltimindtree", "hcl", 
        "hcltech", "cognizant technology solutions"
    ]
    for firm in consulting_firms:
        # Match as full words or exact string
        if firm == name or re.search(r'\b' + re.escape(firm) + r'\b', name):
            return True
    return False

def is_academic_or_research(role_title: str, description: str = "") -> bool:
    """Check if a role is purely academic or research-only with no production engineering."""
    title = clean_text(role_title)
    desc = clean_text(description)
    
    academic_keywords = ["academic", "postdoc", "phd student", "professor", "lecturer", "research fellow", "teaching assistant"]
    for kw in academic_keywords:
        if kw in title:
            # Double check that it's not "research engineer" or "applied research" which can be production
            if "engineer" not in title and "developer" not in title:
                return True
    
    # Check if description points to academic lab only
    if "academic research" in desc or "university lab" in desc:
        if "engineer" not in title and "developer" not in title:
            return True
            
    return False
