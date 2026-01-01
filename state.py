from typing import Annotated, List
from operator import add
from pydantic import BaseModel, Field

class GraphState(BaseModel):
    job_description: str
    cleaned_jd: str = ""
    retrieved_chunks: Annotated[List[str], add] = Field(default_factory=list)
    candidate_summary: str = ""
    resume_summary: str = ""
    cover_letter: str = "   "
    critique_feedback: str = ""
    relevance_score: int = 0
    needs_rewrite: bool = False
    grading_feedback: str = ""
    needs_refinement: bool = False
    refinement_count: int = 0
