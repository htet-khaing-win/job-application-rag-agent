from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
from operator import or_

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
    needs_refinement: Annotated[bool, or_] = False 
    refinement_count: int = 0
    error_type: str = ""            
    error_message: str = ""         
    is_fallback: bool = False       
    final_response: str = "" 
    rewrite_count: int = 0
    company_name: str = ""
    company_research: str = ""
    company_research_success: bool = False
    vector_relevance_score: Optional[float] = None
    verified_skills: List[str] = Field(default_factory=list)
    unverified_skills: List[str] = Field(default_factory=list)
    verified_skills_detailed: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict) 
    refinement_edits: List[Dict[str, str]] = Field(default_factory=list)
    verification_score: int = 100
    joined_context: bool = False
    ready_to_submit: bool = False
