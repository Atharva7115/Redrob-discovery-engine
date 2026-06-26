# System Architecture — 3-Stage Funnel Pipeline

This document describes the 3-stage funnel architecture of the **Semantic Candidate Ranking Engine**, designed to rank 100,000 candidates efficiently, accurately, and within the 5-minute CPU-only sandbox constraint.

---

## Architectural Diagram

```mermaid
graph TD
    A[Raw Candidate Pool <br> N = 100,000] --> B[STAGE 0: Hard Filters & Honeypots]
    
    subgraph STAGE 0: Fast O(N) Filtering
        B --> B1["Honeypot Filter <br> (Job Duration & Skill Anomalies)"]
        B --> B2["Experience Floor <br> (>= 3 Years)"]
        B --> B3["Relocation & Visa Check <br> (India / No Sponsorship)"]
    end
    
    B1 -->|Filter Out| Discard[Discarded Candidates]
    B2 -->|Filter Out| Discard
    B3 -->|Filter Out| Discard
    
    B -->|Passed Pool <br> N1 ~ 95,000| C[STAGE 1: Semantic Retrieval & LTR]
    
    subgraph STAGE 1: Hybrid Retrieval & Learning-to-Rank
        C --> C1["Multi-Vector FAISS Search <br> (Skills, Trajectory, Projects)"]
        C --> C2["Feature Extraction <br> (Tenure, Job-hopping, Consulting, Signals)"]
        C --> C3["LightGBM LambdaMART Ranker <br> (Weak-Supervision trained)"]
    end
    
    C1 --> C3
    C2 --> C3
    
    C3 -->|Top-50 Shortlist| D[STAGE 2: LLM Deep Re-rank]
    
    subgraph STAGE 2: Refined Alignment & Explainability
        D --> E{Is Cached?}
        E -->|Yes| F[Read On-Disk JSON Cache]
        E -->|No| G{Is LLM API Online?}
        G -->|Yes| H["Run Parallel GPT-4o-mini <br> (Return Score, Strengths, Gaps, Reason)"]
        G -->|No| I["Run Local Recruiter-Grade Reasoner <br> (Fact-based offline fallback)"]
        H --> J[Write to On-Disk Cache]
    end
    
    F --> K[Rank & Tiebreak]
    I --> K
    J --> K
    
    scored_candidates["Stage 1 Filler Candidates <br> (Ranks 51-100)"] --> K
    
    K --> L["Final Recruiter shortlist <br> (Top 100 CSV & XLSX)"]
```

---

## Funnel Stages Deep Dive

### Stage 0: Hard Filters (Rule-based, Instant)
* **Goal:** Eliminate unqualified candidates and malicious trap profiles before running any expensive computations.
* **Honeypot Detection:** Scans for impossible profiles (e.g., job duration exceeding physical time elapsed since start date, or claiming expert skills with 0 months of experience). Eliminating these immediately prevents submission disqualification.
* **Operational Constraints:** Checks for minimum experience (e.g., $\ge 3$ years) and ensures no visa sponsorship is required (focusing on candidates in India or willing to relocate).
* **Complexity:** $O(N)$ running in **milliseconds** over the entire 100,000 pool.

### Stage 1: Semantic Retrieval & LTR (Fast Similarity + Tree Ensemble)
* **Goal:** Retrieve the best semantic matches and score them using a multi-factor ranking model.
* **Multi-Vector FAISS Indexes:** We generate separate semantic embeddings for (a) skills and tech stack, (b) career trajectory timeline, and (c) project work descriptions. Standardizing them to unit length allows FAISS `IndexFlatIP` to perform exact **cosine similarity** matching. We search all three indexes and combine them using dynamic weights derived from the JD's emphasis.
* **Feature Extractor:** Extracts structural features like tenure, promotions, job-hopping frequency, and behavioral platform signals (notice period, response rate, activity).
* **LightGBM Ranker:** A LambdaMART ranking model trained via weak-supervision to combine similarities, trajectory flags, and behavioral signals into a single calibrated score.
* **Complexity:** $O(N_1 \log K)$ running in **seconds** (FAISS search is sub-second; LTR prediction is sub-second).

### Stage 3: LLM Deep Re-rank (High-Fidelity Alignment & Explanations)
* **Goal:** Run deep semantic reasoning over the top candidate profiles to produce the final ranking and human-readable justifications.
* **LLM Prompts:** Feeds a detailed candidate profile summary, LTR scores, and the JD to `gpt-4o-mini`, returning structured JSON with a refined score, strengths, gaps, and a 1-2 sentence recruiter-grade justification.
* **Caching & Sandbox Compliance:** Since the hackathon ranking script must run **completely offline**, calling live LLM APIs at runtime is prohibited. To solve this:
  1. We pre-compute and cache LLM verdicts during development/indexing.
  2. At runtime, the re-ranker checks a local, thread-safe JSON cache.
  3. If a candidate is not in the cache, it falls back to a **local recruiter-grade reasoner**. This local engine analyzes the candidate's exact profile facts and dynamically constructs a non-templated, highly specific, and factually accurate justification.
* **Filler Logic:** To produce exactly 100 candidates for the submission while keeping LLM calls capped at 50, we re-rank the top 50 with the LLM, and fill positions 51-100 with the next best LTR candidates, generating their justifications locally.
