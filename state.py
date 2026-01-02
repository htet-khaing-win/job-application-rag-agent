from typing import List, Dict, Any
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
    relevance_score: float = 0
    needs_rewrite: bool = False
    grading_feedback: str = ""
    needs_refinement: bool = False
    refinement_count: int = 0
    error_type: str = ""            
    error_message: str = ""         
    is_fallback: bool = False       
    final_response: str = ""
