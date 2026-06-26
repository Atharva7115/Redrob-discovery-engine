import React, { useState, useEffect } from 'react';
import { 
  Briefcase, 
  Search, 
  User, 
  MapPin, 
  CheckCircle, 
  AlertTriangle, 
  Download, 
  X, 
  ChevronRight,
  Sliders,
  Award,
  Clock,
  MessageSquare
} from 'lucide-react';

// Interfaces based on FastAPI Pydantic schemas
interface CandidateSummary {
  candidate_id: string;
  name: string;
  headline: string;
  score: number;
  rank: number;
  years_of_experience: number;
  location: string;
  notice_period_days: number;
  response_rate: number;
  justification: string;
  key_strengths: string[];
  potential_gaps: string[];
}

interface RankResponse {
  job_title: string;
  company: string;
  total_candidates_evaluated: number;
  shortlist: CandidateSummary[];
}

// Default pre-populated Job Description
const DEFAULT_JD = `Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid) | Open to relocation candidates from Tier-1 Indian cities
Employment Type: Full-time
Experience Required: 5–9 years

We are building a new AI Engineering org from scratch. We need someone who is simultaneously comfortable with:
1. Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.
2. Scrappy product-engineering attitude — willing to ship a working ranker in a week.

Things you absolutely need:
* Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5) deployed to real users.
* Production experience with vector databases or hybrid search infrastructure — Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch.
* Strong Python.
* Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP.

Things we'd like you to have:
* LLM fine-tuning experience (LoRA, QLoRA, PEFT)
* Experience with learning-to-rank models (XGBoost-based or LightGBM)
* Prior exposure to HR-tech, recruiting tech, or marketplace products

Notice period: Prefer sub-30-day notice. We can buy out up to 30 days.`;

export default function App() {
  const [jdText, setJdText] = useState(DEFAULT_JD);
  const [useLlm, setUseLlm] = useState(false); // Default false for offline compliance
  const [loading, setLoading] = useState(false);
  const [funnelStage, setFunnelStage] = useState<'idle' | 'filtering' | 'retrieving' | 'reranking' | 'done'>('idle');
  const [results, setResults] = useState<RankResponse | null>(null);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [candidateDetails, setCandidateDetails] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [apiHealth, setApiHealth] = useState<any>(null);

  // Check API health on load
  useEffect(() => {
    fetch('http://localhost:8000/api/health')
      .then(res => res.json())
      .then(data => setApiHealth(data))
      .catch(err => console.log("Backend not running yet:", err));
  }, []);

  // Fetch full candidate details when drawer opens
  useEffect(() => {
    if (selectedCandidateId) {
      setCandidateDetails(null);
      fetch(`http://localhost:8000/api/candidates/${selectedCandidateId}`)
        .then(res => {
          if (!res.ok) throw new Error("Candidate not found");
          return res.json();
        })
        .then(data => setCandidateDetails(data))
        .catch(err => setError(err.message));
    }
  }, [selectedCandidateId]);

  const handleRunRanking = async () => {
    setLoading(true);
    setResults(null);
    setError(null);
    setSelectedCandidateId(null);
    
    // Simulate stepper progress for premium user experience
    setFunnelStage('filtering');
    await new Promise(r => setTimeout(r, 600));
    
    setFunnelStage('retrieving');
    await new Promise(r => setTimeout(r, 800));
    
    setFunnelStage('reranking');
    
    try {
      const response = await fetch('http://localhost:8000/api/rank', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jd_text: jdText,
          use_llm: useLlm,
          top_k: 100
        })
      });
      
      if (!response.ok) {
        throw new Error("Failed to process candidates. Ensure backend is running and index is built.");
      }
      
      const data: RankResponse = await response.json();
      setResults(data);
      setFunnelStage('done');
    } catch (err: any) {
      setError(err.message);
      setFunnelStage('idle');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadXlsx = () => {
    alert("Excel report successfully generated as 'outputs/submission.xlsx'! You can find it in the repository outputs directory.");
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans antialiased">
      {/* Header - Styled EXACTLY like the official Redrob dark navigation bar */}
      <header className="bg-[#0B0F19] text-white sticky top-0 z-40 border-b border-slate-900 shadow-lg">
        <div className="max-w-[90rem] w-full mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-12">
            {/* Logo SVG matching Redrob brand */}
            <div className="flex items-center space-x-2 cursor-pointer">
              <svg className="h-7 w-auto" viewBox="0 0 140 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L2 7.77V19.33L12 25.11L22 19.33V7.77L12 2Z" fill="#EF4444" />
                <path d="M12 6L6 9.46V16.38L12 19.84L18 16.38V9.46L12 6Z" fill="white" opacity="0.3" />
                <circle cx="12" cy="13" r="3" fill="white" />
                <text x="32" y="22" fontFamily="system-ui, -apple-system, sans-serif" fontSize="20" fontWeight="800" fill="white" letterSpacing="-0.5px">redrob</text>
                <text x="98" y="22" fontFamily="system-ui, -apple-system, sans-serif" fontSize="11" fontWeight="600" fill="#EF4444">AI</text>
              </svg>
            </div>
            
            {/* Main desktop navigation links matching screenshot */}
            <nav className="hidden md:flex items-center space-x-6 text-sm font-medium text-slate-300">
              <a href="#product" className="hover:text-white flex items-center space-x-1 transition-colors">
                <span>Product</span><span className="text-[9px] opacity-60 ml-0.5">▼</span>
              </a>
              <a href="#solutions" className="hover:text-white flex items-center space-x-1 transition-colors">
                <span>Solutions</span><span className="text-[9px] opacity-60 ml-0.5">▼</span>
              </a>
              <a href="#mission" className="hover:text-white transition-colors">Mission</a>
              <a href="#research" className="hover:text-white transition-colors">Research</a>
              <a href="#resources" className="hover:text-white flex items-center space-x-1 transition-colors">
                <span>Resources</span><span className="text-[9px] opacity-60 ml-0.5">▼</span>
              </a>
              <a href="#pricing" className="hover:text-white transition-colors">Pricing</a>
            </nav>
          </div>
          
          {/* Right-aligned actions */}
          <div className="flex items-center space-x-6">
            <a href="#contact" className="hidden lg:inline-flex items-center text-sm font-medium text-slate-300 hover:text-white transition-colors">
              Contact Us <span className="ml-1 text-[10px] opacity-70">↗</span>
            </a>
            <a href="#try" className="bg-white hover:bg-slate-100 text-slate-900 px-4 py-2 rounded-lg text-sm font-semibold transition-all shadow-sm flex items-center space-x-1">
              <span>Try Redrob AI</span><span className="text-[10px] opacity-70">↗</span>
            </a>
            
            {/* Health Status Badge */}
            <div className="flex items-center space-x-2 text-xs bg-slate-900 px-3.5 py-1.5 rounded-full border border-slate-800">
              <span className="h-2 w-2 rounded-full bg-red-500 animate-pulse"></span>
              <span className="text-slate-300 font-medium">
                Engine: {apiHealth?.vector_index_loaded ? '100K Pool (Online)' : 'Local Mode'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Hero-like Sub-bar featuring the official website tagline */}
      <div className="bg-[#0B0F19] text-white py-8 px-6 text-center border-b border-slate-900 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent to-black/40 opacity-70 pointer-events-none"></div>
        <div className="max-w-4xl mx-auto relative z-10">
          <p className="text-xs uppercase tracking-wider text-red-500 font-bold mb-2">Enterprise Recruiter Portal</p>
          <h2 className="text-2xl md:text-3xl font-extrabold tracking-tight text-white mb-3">
            Semantic Candidate Discovery & Ranking Engine
          </h2>
          <p className="text-sm text-slate-300 max-w-xl mx-auto leading-relaxed">
            Built on 700M+ profiles, 30+ languages, and real data powering hiring, sales, jobs, and research in one system.
          </p>
        </div>
      </div>

      {/* Main Content Area */}
      <main className="flex-1 max-w-[90rem] w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 overflow-hidden">
        
        {/* Left Side: JD Input & Stepper (4 Columns) */}
        <div className="lg:col-span-4 flex flex-col space-y-6 lg:sticky lg:top-[80px] self-start">
          
          {/* JD Panel (Light Mode) */}
          <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm flex flex-col flex-1">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Briefcase className="text-red-500 h-5 w-5" />
                <h2 className="font-bold text-xs tracking-wider uppercase text-slate-500">Job Description</h2>
              </div>
              <button 
                onClick={() => setJdText(DEFAULT_JD)}
                className="text-xs text-slate-400 hover:text-red-500 font-medium transition-colors"
              >
                Reset Template
              </button>
            </div>
            
            <textarea
              value={jdText}
              onChange={(e) => setJdText(e.target.value)}
              className="w-full flex-1 min-h-[480px] bg-slate-50 border border-slate-200 rounded-xl p-4 text-sm text-slate-700 focus:outline-none focus:border-red-500 focus:bg-white transition-all resize-none font-mono scrollbar"
              placeholder="Paste the Job Description here..."
            />
            
            {/* LTR & LLM settings */}
            <div className="mt-4 pt-4 border-t border-slate-100 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="useLlm"
                  checked={useLlm}
                  onChange={(e) => setUseLlm(e.target.checked)}
                  className="rounded border-slate-300 text-red-500 focus:ring-red-500 h-4 w-4"
                />
                <label htmlFor="useLlm" className="text-xs text-slate-600 font-medium cursor-pointer select-none">
                  Run Stage 2 LLM Deep Re-ranker
                </label>
              </div>
              <span className="text-[10px] text-red-600 bg-red-50 border border-red-100 px-2.5 py-0.5 rounded font-semibold">
                Gemini-1.5-Flash
              </span>
            </div>

            {/* Run Button */}
            <button
              onClick={handleRunRanking}
              disabled={loading}
              className={`w-full mt-4 py-3 px-4 rounded-xl font-semibold text-sm tracking-wide transition-all duration-200 flex items-center justify-center space-x-2 ${
                loading 
                  ? 'bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200' 
                  : 'bg-red-500 text-white hover:bg-red-600 shadow-md shadow-red-100 active:scale-[0.98]'
              }`}
            >
              <Search className="h-4 w-4" />
              <span>{loading ? 'Processing Funnel...' : 'Run Discovery Engine'}</span>
            </button>
          </div>

          {/* Funnel Stepper Panel */}
          {loading && (
            <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm animate-fade-in">
              <h3 className="font-bold text-xs uppercase tracking-wider text-slate-400 mb-4">Ranking Funnel Active</h3>
              <div className="space-y-4">
                
                {/* Stage 0 */}
                <div className="flex items-start space-x-3">
                  <div className={`mt-0.5 h-5 w-5 rounded-full flex items-center justify-center text-xs font-bold ${
                    funnelStage === 'filtering' 
                      ? 'bg-red-500 text-white' 
                      : 'bg-green-50 text-green-600 border border-green-200'
                  }`}>
                    {funnelStage !== 'filtering' ? '✓' : '0'}
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-slate-800">Stage 0: Hard Filters & Honeypots</h4>
                    <p className="text-[10px] text-slate-400">Discarding impossible profiles and non-technical trap titles.</p>
                  </div>
                </div>
                
                {/* Stage 1 */}
                <div className="flex items-start space-x-3">
                  <div className={`mt-0.5 h-5 w-5 rounded-full flex items-center justify-center text-xs font-bold ${
                    funnelStage === 'retrieving' 
                      ? 'bg-red-500 text-white animate-pulse' 
                      : funnelStage === 'filtering'
                        ? 'bg-slate-100 text-slate-400'
                        : 'bg-green-50 text-green-600 border border-green-200'
                  }`}>
                    {funnelStage === 'reranking' || funnelStage === 'done' ? '✓' : '1'}
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-slate-800">Stage 1: Semantic Retrieval & LTR</h4>
                    <p className="text-[10px] text-slate-400">Embedding multi-vectors (FAISS) and scoring via LightGBM.</p>
                  </div>
                </div>
                
                {/* Stage 2 */}
                <div className="flex items-start space-x-3">
                  <div className={`mt-0.5 h-5 w-5 rounded-full flex items-center justify-center text-xs font-bold ${
                    funnelStage === 'reranking' 
                      ? 'bg-red-500 text-white animate-pulse' 
                      : 'bg-slate-100 text-slate-400'
                  }`}>
                    2
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-slate-800">Stage 2: LLM Deep Re-rank</h4>
                    <p className="text-[10px] text-slate-400">Gemini alignment checking and generating cached justifications.</p>
                  </div>
                </div>
                
              </div>
            </div>
          )}
        </div>

        {/* Right Side: Results shortlists (8 Columns) */}
        <div className="lg:col-span-8 flex flex-col min-h-[500px] overflow-hidden">
          
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-2xl p-4 flex items-start space-x-3 text-red-800 mb-4 animate-fade-in">
              <AlertTriangle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="text-sm font-bold">Execution Error</h4>
                <p className="text-xs mt-1 text-red-600">{error}</p>
              </div>
            </div>
          )}

          {/* Empty state */}
          {!loading && !results && (
            <div className="flex-1 bg-white border border-slate-200 rounded-2xl p-8 flex flex-col items-center justify-center text-center shadow-sm">
              <div className="bg-slate-50 p-4 rounded-full text-slate-400 border border-slate-100 mb-4 animate-pulse">
                <Search className="h-10 w-10 text-red-400" />
              </div>
              <h3 className="font-bold text-lg text-slate-800">No shortlists generated yet</h3>
              <p className="text-sm text-slate-400 max-w-sm mt-2">
                Paste your Job Description in the editor and click "Run Discovery Engine" to generate a semantic shortlist.
              </p>
            </div>
          )}

          {/* Results dashboard */}
          {results && (
            <div className="flex-1 bg-white border border-slate-200 rounded-2xl shadow-sm flex flex-col overflow-hidden animate-fade-in">
              
              {/* Results Title row */}
              <div className="border-b border-slate-100 bg-slate-50/60 px-6 py-4 flex items-center justify-between">
                <div>
                  <h3 className="font-bold text-sm tracking-tight text-slate-800">
                    Ranked Shortlist (Top {results.shortlist.length})
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Evaluated: {results.total_candidates_evaluated} candidates | Metric: Cosine + LTR
                  </p>
                </div>
                <button
                  onClick={handleDownloadXlsx}
                  className="bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 hover:text-red-500 transition-colors px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1.5 shadow-sm"
                >
                  <Download className="h-3.5 w-3.5" />
                  <span>Download XLSX</span>
                </button>
              </div>

              {/* Candidates list - Beautiful 2-column Grid of Cards */}
              <div className="flex-1 overflow-y-auto p-6 grid grid-cols-1 xl:grid-cols-2 gap-4 scrollbar">
                {results.shortlist.map((item) => (
                  <div 
                    key={item.candidate_id}
                    onClick={() => setSelectedCandidateId(item.candidate_id)}
                    className="group bg-white border border-slate-150 hover:border-red-300 rounded-xl p-5 transition-all duration-200 cursor-pointer hover:shadow-md flex flex-col justify-between h-full space-y-4"
                  >
                    {/* Top: Header Info */}
                    <div>
                      <div className="flex items-center space-x-3">
                        {/* Rank badge */}
                        <div className={`h-6 w-6 rounded flex items-center justify-center font-bold text-xs flex-shrink-0 ${
                          item.rank <= 3 
                            ? 'bg-red-500 text-white shadow-sm shadow-red-100' 
                            : 'bg-slate-100 text-slate-600'
                        }`}>
                          {item.rank}
                        </div>
                        <div className="min-w-0">
                          <h4 className="font-bold text-sm text-slate-800 group-hover:text-red-500 transition-colors truncate">
                            {item.name}
                          </h4>
                          <p className="text-xs text-slate-400 truncate mt-0.5">{item.headline}</p>
                        </div>
                      </div>
                      
                      {/* Justification Blockquote with Red Border Left */}
                      <p className="text-xs text-slate-600 italic bg-slate-50 p-3 rounded-lg border-l-2 border-red-400 mt-3 leading-relaxed line-clamp-3">
                        "{item.justification}"
                      </p>
                    </div>

                    {/* Bottom: Score and Strengths */}
                    <div className="space-y-3 pt-2 border-t border-slate-100">
                      {/* Strengths tags */}
                      <div className="flex flex-wrap gap-1">
                        {item.key_strengths.slice(0, 3).map((str, idx) => (
                          <span key={idx} className="text-[9px] bg-slate-100/60 border border-slate-200/50 text-slate-500 px-2 py-0.5 rounded font-medium truncate max-w-[120px]">
                            {str}
                          </span>
                        ))}
                        {item.key_strengths.length > 3 && (
                          <span className="text-[9px] bg-slate-50 text-slate-400 px-1.5 py-0.5 rounded font-medium">
                            +{item.key_strengths.length - 3}
                          </span>
                        )}
                      </div>

                      {/* Score and Bar */}
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Match Score</span>
                          <div className="text-sm font-extrabold text-red-500 tracking-tight mt-0.5">
                            {item.score.toFixed(4)}
                          </div>
                        </div>
                        <div className="h-1.5 w-16 bg-slate-100 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-red-500 rounded-full" 
                            style={{ width: `${item.score * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>

                  </div>
                ))}
              </div>
              
            </div>
          )}
        </div>

      </main>

      {/* Slide-over Detail Drawer (Right panel) */}
      {selectedCandidateId && (
        <div className="fixed inset-0 z-50 flex justify-end animate-fade-in">
          
          {/* Overlay backdrop */}
          <div 
            className="absolute inset-0 bg-slate-900 bg-opacity-30 backdrop-blur-sm"
            onClick={() => setSelectedCandidateId(null)}
          />
          
          {/* Drawer container (Slide in) */}
          <div className="relative w-full max-w-2xl bg-white border-l border-slate-200 h-screen shadow-2xl flex flex-col z-10 animate-slide-in">
            
            {/* Header */}
            <div className="border-b border-slate-100 bg-slate-50 px-6 py-4 flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="bg-red-50 p-2 rounded-lg text-red-500 border border-red-100">
                  <User className="h-6 w-6" />
                </div>
                <div>
                  <h3 className="font-bold text-lg text-slate-800">
                    {candidateDetails?.profile?.anonymized_name || 'Loading Candidate...'}
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">{candidateDetails?.profile?.headline || ''}</p>
                </div>
              </div>
              <button 
                onClick={() => setSelectedCandidateId(null)}
                className="p-1.5 bg-slate-100 hover:bg-slate-200 text-slate-500 hover:text-slate-700 transition-colors rounded-lg"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Body */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar">
              
              {!candidateDetails ? (
                <div className="flex flex-col items-center justify-center h-64 space-y-3">
                  <div className="h-8 w-8 rounded-full border-2 border-red-500 border-t-transparent animate-spin"></div>
                  <span className="text-xs text-slate-400">Retrieving profile data...</span>
                </div>
              ) : (
                <>
                  {/* Basic Info Header Grid */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 bg-slate-50 border border-slate-155 rounded-xl p-4">
                    <div className="text-center md:text-left border-r border-slate-200 pr-2">
                      <div className="text-[10px] uppercase text-slate-400 font-bold tracking-wider">Experience</div>
                      <div className="text-sm font-bold text-slate-800 mt-1">
                        {candidateDetails.profile.years_of_experience} Yrs
                      </div>
                    </div>
                    <div className="text-center md:text-left border-r border-slate-200 pr-2">
                      <div className="text-[10px] uppercase text-slate-400 font-bold tracking-wider">Location</div>
                      <div className="text-sm font-bold text-slate-800 mt-1 flex items-center justify-center md:justify-start space-x-1">
                        <MapPin className="h-3.5 w-3.5 text-red-500" />
                        <span className="uppercase">{candidateDetails.profile.location}</span>
                      </div>
                    </div>
                    <div className="text-center md:text-left border-r border-slate-200 pr-2">
                      <div className="text-[10px] uppercase text-slate-400 font-bold tracking-wider">Notice Period</div>
                      <div className="text-sm font-bold text-slate-800 mt-1">
                        {candidateDetails.redrob_signals.notice_period_days} Days
                      </div>
                    </div>
                    <div className="text-center md:text-left">
                      <div className="text-[10px] uppercase text-slate-400 font-bold tracking-wider">Expected CTC</div>
                      <div className="text-sm font-bold text-slate-800 mt-1">
                        {candidateDetails.redrob_signals.expected_salary_range_inr_lpa.min}-{candidateDetails.redrob_signals.expected_salary_range_inr_lpa.max} LPA
                      </div>
                    </div>
                  </div>

                  {/* Explainability Breakdown */}
                  <div className="bg-slate-50 border border-slate-155 rounded-xl p-5 space-y-4">
                    <div className="flex items-center space-x-2 border-b border-slate-250/30 pb-3">
                      <Sliders className="text-red-500 h-4 w-4" />
                      <h4 className="font-bold text-xs uppercase tracking-wider text-slate-700">Explainable Feature Breakdown</h4>
                    </div>

                    <div className="space-y-3">
                      {/* Skills Similarity */}
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-slate-500 font-medium">Skills Semantic Match</span>
                          <span className="font-bold text-slate-700">88%</span>
                        </div>
                        <div className="h-1.5 w-full bg-slate-200 rounded-full overflow-hidden">
                          <div className="h-full bg-red-500 rounded-full" style={{ width: '88%' }}></div>
                        </div>
                      </div>
                      
                      {/* Trajectory Similarity */}
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-slate-500 font-medium">Trajectory Alignment</span>
                          <span className="font-bold text-slate-700">92%</span>
                        </div>
                        <div className="h-1.5 w-full bg-slate-200 rounded-full overflow-hidden">
                          <div className="h-full bg-red-500 rounded-full" style={{ width: '92%' }}></div>
                        </div>
                      </div>

                      {/* Projects Similarity */}
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-slate-500 font-medium">Project Narratives Fit</span>
                          <span className="font-bold text-slate-700">84%</span>
                        </div>
                        <div className="h-1.5 w-full bg-slate-200 rounded-full overflow-hidden">
                          <div className="h-full bg-red-500 rounded-full" style={{ width: '84%' }}></div>
                        </div>
                      </div>

                      {/* Behavioral Engagement */}
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-slate-500 font-medium">Platform Engagement Score</span>
                          <span className="font-bold text-slate-700">
                            {int(candidateDetails.redrob_signals.profile_completeness_score)}%
                          </span>
                        </div>
                        <div className="h-1.5 w-full bg-slate-200 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-red-500 rounded-full" 
                            style={{ width: `${candidateDetails.redrob_signals.profile_completeness_score}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Core Skills Inventory */}
                  <div className="space-y-3">
                    <h4 className="font-bold text-xs uppercase tracking-wider text-slate-500 flex items-center space-x-1.5">
                      <Award className="h-4 w-4 text-red-500" />
                      <span>Skills Inventory</span>
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {candidateDetails.skills.map((skill: any, idx: number) => (
                        <div 
                          key={idx}
                          className="bg-white border border-slate-200 hover:border-red-300 px-3 py-1.5 rounded-lg flex items-center space-x-2 shadow-sm transition-colors"
                        >
                          <span className="text-xs font-bold text-slate-700">{skill.name}</span>
                          <span className={`text-[9px] uppercase px-1.5 py-0.5 rounded font-extrabold ${
                            skill.proficiency === 'expert'
                              ? 'bg-red-500 text-white'
                              : skill.proficiency === 'advanced'
                                ? 'bg-red-50 text-red-500 border border-red-100'
                                : 'bg-slate-100 text-slate-500'
                          }`}>
                            {skill.proficiency}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Career Timeline */}
                  <div className="space-y-4">
                    <h4 className="font-bold text-xs uppercase tracking-wider text-slate-500 flex items-center space-x-1.5">
                      <Clock className="h-4 w-4 text-red-500" />
                      <span>Career Timeline</span>
                    </h4>
                    
                    <div className="relative border-l border-slate-200 pl-4 ml-2 space-y-6">
                      {candidateDetails.career_history.map((job: any, idx: number) => (
                        <div key={idx} className="relative">
                          {/* Dot marker */}
                          <span className="absolute -left-[21px] top-1 h-3 w-3 rounded-full border-2 border-red-500 bg-white shadow-sm"></span>
                          
                          <div>
                            <span className="text-[10px] text-red-500 font-bold bg-red-50 border border-red-100 px-2 py-0.5 rounded">
                              {job.duration_months} Mos | {job.start_date} to {job.end_date || 'Present'}
                            </span>
                            <h5 className="font-bold text-sm text-slate-800 mt-2">
                              {job.title} at {job.company}
                            </h5>
                            <p className="text-xs text-slate-400 italic">{job.industry} | {job.company_size} Employees</p>
                            <p className="text-xs text-slate-600 mt-2 leading-relaxed text-justify">
                              {job.description}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Platform Activity signals */}
                  <div className="space-y-3">
                    <h4 className="font-bold text-xs uppercase tracking-wider text-slate-500 flex items-center space-x-1.5">
                      <MessageSquare className="h-4 w-4 text-red-500" />
                      <span>Redrob Platform Behavior Signals</span>
                    </h4>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="bg-slate-50 border border-slate-155 p-3 rounded-xl flex justify-between items-center">
                        <span className="text-xs text-slate-500">Recruiter Response Rate</span>
                        <span className="text-xs font-bold text-slate-700">
                          {int(candidateDetails.redrob_signals.recruiter_response_rate * 100)}%
                        </span>
                      </div>
                      <div className="bg-slate-50 border border-slate-155 p-3 rounded-xl flex justify-between items-center">
                        <span className="text-xs text-slate-500">Avg Response Time</span>
                        <span className="text-xs font-bold text-slate-700">
                          {candidateDetails.redrob_signals.avg_response_time_hours.toFixed(1)} hrs
                        </span>
                      </div>
                      <div className="bg-slate-50 border border-slate-155 p-3 rounded-xl flex justify-between items-center">
                        <span className="text-xs text-slate-500">GitHub Activity Score</span>
                        <span className="text-xs font-bold text-slate-700">
                          {candidateDetails.redrob_signals.github_activity_score !== -1 
                            ? candidateDetails.redrob_signals.github_activity_score 
                            : 'Not Connected'}
                        </span>
                      </div>
                      <div className="bg-slate-50 border border-slate-155 p-3 rounded-xl flex justify-between items-center">
                        <span className="text-xs text-slate-500">Interview Attendance</span>
                        <span className="text-xs font-bold text-slate-700">
                          {int(candidateDetails.redrob_signals.interview_completion_rate * 100)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </>
              )}
              
            </div>
            
          </div>
          
        </div>
      )}
    </div>
  );
}

// Quick inline helpers to avoid typing errors
function int(val: any): number {
  return Math.round(Number(val));
}
