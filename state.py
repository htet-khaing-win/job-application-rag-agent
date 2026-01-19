from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class GraphState(BaseModel):
    job_description: str
    is_valid_jd: bool = False
    cleaned_jd: str = ""
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    candidate_summary: str = ""
    resume_summary: str = ""
    cover_letter: str = ""
    critique_feedback: str = ""
    needs_rewrite: bool = False
    grading_feedback: str = ""
    needs_refinement: bool = False
    refinement_count: int = 0 # Tracks the times Agent rewrite the letter based on critique feedback
    error_type: str = ""            
    error_message: str = ""         
    is_fallback: bool = False       
    final_response: str = "" 
    rewrite_count: int = 0  # Tracks the times Agent re-query Pinecone due to jd and resume mismatch
    company_name: str = ""
    company_research: str = ""
    company_research_success: bool = False
    verification_log: str = ""
    vector_relevance_score: Optional[float] = None   # Pinecone similarity (0–100)
    llm_relevance_score: int = 0           # LLM judgment (0–100)
    verified_skills: List[str] = Field(default_factory=list)  # Simple list for compatibility
    unverified_skills: List[str] = Field(default_factory=list)
    verified_skills_detailed: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict) 
    refinement_edits: List[Dict[str, str]] = Field(default_factory=list)  # NEW
    refinement_count: int = 0