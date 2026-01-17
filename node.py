from langgraph.graph import StateGraph, END, START
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from state import GraphState
import json
import re
from tavily import TavilyClient

load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# INGEST JOB DESCRIPTION (Entry Node)
def ingest_jd_node(state: GraphState, llm) -> dict:

    validation_prompt = f"""
    ROLE: Input Validator
    
    TASK: Determine if this text is a job description or job-related query.
    
    INPUT: {state.job_description}
    
    CRITERIA:
    - Contains job requirements, qualifications, or responsibilities
    - Mentions a role title or position
    - Describes company needs or hiring context
    
    OUTPUT: Return ONLY one word: VALID or INVALID
    """
    validation_response = llm.invoke(validation_prompt)
    if "INVALID" in validation_response.content.upper():
        return {
            "is_valid_jd": False,
            "error_type": "invalid_input",
            "error_message": "This request doesn't appears to be a job description. Please paste a complete job posting."
        }
    
    cleaning_prompt = f"""Extract from this job description:
    - Core Requirements: must-have skills/qualifications
    - Key Responsibilities: primary duties
    - Target Keywords: critical ATS terms (comma-separated)

    Remove generic HR fluff. Focus on technical requirements and measurable outcomes.

    {state.job_description}"""
    
    response = llm.invoke(cleaning_prompt)
    return {
        "cleaned_jd": response.content,
        "is_valid_jd": True
    }


def research_company_node(state: GraphState, llm) -> dict:
    """
    Uses Tavily AI to fetch real-time company information. This runs AFTER user confirms the company name.
    
    Purpose:
        - Fetches company mission, values, recent news
        - Provides grounding for personalized cover letter opening
        - Reduces generic "I'm excited about [Company]" statements
    """
    
    company_name = state.company_name
    
    try:
        # Tavily search query
        search_query = f"{company_name} company mission values recent news 2024 2025"
        
        search_results = tavily_client.search(
            query=search_query,
            search_depth="advanced",
            max_results=5
        )
        
        # Extract relevant information
        research_text = f"Company: {company_name}\n\n"
        
        for result in search_results.get('results', []):
            research_text += f"Source: {result.get('title', 'N/A')}\n"
            research_text += f"{result.get('content', '')}\n\n"
        
        # Use LLM to synthesize research into usable summary
        synthesis_prompt = f"""Synthesize into 100-150 words:
        - Company Overview: 2-3 mission/values
        - Recent Developments: 1-2 items from past 6mo
        - Culture Insights: unique aspects

        {research_text}"""
        
        synthesis_response = llm.invoke(synthesis_prompt)
        
        return {
            "company_research": synthesis_response.content,
            "company_research_success": True
        }
        
    except Exception as e:
        print(f"Tavily Research Error: {str(e)}")
        return {
            "company_research": f"Unable to fetch current information about {company_name}.",
            "company_research_success": False
        }

def verify_claims_node(state: GraphState, llm) -> dict:
    """
    Guardrail Node - Validates that candidate_summary only contains 
    verifiable information from retrieved_chunks.
    
    Purpose:
        - Prevents hallucinated skills/qualifications
        - Cross-references summary against source chunks
        - Filters out unsupported claims
    """
    # precondition check
    if not state.candidate_summary.strip():
        return {
            "candidate_summary": "",
            "verification_log": "No candidate summary provided for verification.",
            "error_type": "verification_failed",
            "error_message": "Candidate summary was empty before verification."
        }
    
    # Extract actual text from retrieved chunks for verification
    source_text = "\n".join([
        chunk["text"] for chunk in state.retrieved_chunks
    ])
    
    verification_prompt = f"""Validate EVERY claim in Candidate Summary against Source Resume Data.

    RULES:
    - No inference (e.g., "used Python" is not equate to "expert Python")
    - No unstated qualifications or experience
    - No technology extrapolation
    - Cite specific projects as evidence 

    SOURCE RESUME DATA: {source_text}

    CANDIDATE SUMMARY: {state.candidate_summary}

    OUTPUT:
    Verified Claims:
    - [Claim]: [Source quote]

    Removed Claims:
    - [Claim]: [Reason]

    Final Verified Summary:
    [Rewrite using ONLY verified claims]"""
    
    response = llm.invoke(verification_prompt)
    
    # Extract the "Final Verified Summary" section
    content = response.content
    if "Final Verified Summary:" in content:
        verified_summary = content.split("Final Verified Summary:")[-1].strip()
    else:
        # Fallback: use the entire response if format isn't followed
        verified_summary = content
    
    MIN_SUMMARY_LENGTH = 100  # characters, deliberately conservative

    if len(verified_summary) < MIN_SUMMARY_LENGTH:
        return {
            "candidate_summary": verified_summary,
            "verification_log": content,
            "error_type": "verification_failed",
            "error_message": ("Most candidate claims could not be verified against the uploaded resume evidence.")
        }
    
    return {
        "candidate_summary": verified_summary,
        "verification_log": content
    }

def fallback_handler_node(state: GraphState, llm) -> dict:
    """
    Handles invalid inputs and severe mismatches.
    
    Purpose:
        - Provides helpful error messages
        - Guides user back to valid JD input
        - Prevents hallucinated cover letters
    
    Triggers:
        - state.is_valid_jd == False (invalid input)
        - state.error_type == "no_resumes" (empty database)
        - state.llm_relevance_score < 50 after 2 rewrites (severe mismatch)
    """
    
    if state.error_type == "invalid_input":
        message = f"""
    I couldn't identify this as a job description. 

    To generate a cover letter, please paste:
    - A complete job posting
    - The job requirements section
    - Or describe the role you're applying for

    """
    
    elif state.error_type == "no_resumes":
        message = """
    No resumes found in the system.

    Please upload your resume first using the file upload feature, then paste the job description again.
    """
    
    elif state.llm_relevance_score < 50:
        message = f"""
    I couldn't find a strong match between your resumes and this job description (relevance: {state.llm_relevance_score}%).

    Suggestions:
    - Upload a more relevant resume for this role
    - Ensure the job description is complete and specific
    - Try a different position that better matches your background

    Would you like to try a different job description?
    """
    
    else:
        # Generic fallback for unexpected errors
        message = """
    I specialize in generating cover letters based on job descriptions.

    Please paste a job description, and I'll create a tailored cover letter using your uploaded resume(s).
    """
    
    return {
        "final_response": message,
        "is_fallback": True
    }

# GRADE RETRIEVAL (The Critic)
def grade_retrieval_node(state: GraphState, llm) -> dict:
    """
    The Critic - Evaluates quality of retrieved resume matches.
    
    FIXED: Now only generates a 0-100 score. Routing decision is made purely in Python.
    """
    
    prompt = f"""Score resume-to-job match (0-100):
    - 90-100: Exceptional, all key requirements met
    - 70-89: Good, most requirements met
    - 50-69: Moderate, some gaps
    - 30-49: Weak, significant gaps
    - 0-29: Poor match

    Focus on what IS present, not missing.

    JOB REQUIREMENTS: {state.cleaned_jd}

    RESUME CHUNKS: {state.retrieved_chunks}

    Return strict JSON only (no markdown):
    {{"score": <int>, "reasoning": "<2-3 sentences>"}}"""
    
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    # Remove common markdown artifacts
    content = content.replace("```json", "").replace("```", "").strip()
    
    try:
        data = json.loads(content)
        score = int(data.get("score", 0))
        reasoning = data.get("reasoning", "No reasoning provided.")
        
        # Validation
        if not (0 <= score <= 100):
            print(f"Warning: Score {score} out of range, clamping to 0-100")
            score = max(0, min(100, score))
            
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print(f"Raw LLM Output: {content[:200]}")
        
        # Fallback regex extraction
        score_match = re.search(r'"score":\s*(\d+)', content)
        score = int(score_match.group(1)) if score_match else 0
        
        reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', content)
        reasoning = reasoning_match.group(1) if reasoning_match else "Parsing error occurred."
    
    print(f" Grading Result: Score={score}")
    
    return {
        "llm_relevance_score": score,
        "grading_feedback": reasoning
    }


# GENERATE SUMMARY (The Analyst)
def generate_summary_node(state: GraphState, llm) -> dict:
    """
    The Analyst - Creates a bridging narrative between candidate and role.
    
    Purpose:
        - Maps candidate achievements to job "Must-Haves"
        - Synthesizes best-fit resume data with JD requirements
        - Produces evidence-based talking points for cover letter
    
    Input: 
        - state.cleaned_jd - Target job requirements
        - state.retrieved_chunks - Top candidate resume data
    Output: 
        - state.candidate_summary - Structured match analysis
    """
    prompt = f"""Create candidate-job match analysis:
    - Key Qualifications Match: map top 5 achievements to job requirements (with metrics)
    - Standout Achievements: top 3 with quantifiable results
    - Value Proposition: 2-3 sentences on unique fit, skill gaps

    Third person, professional tone.

    JOB REQUIREMENTS: {state.cleaned_jd}

    RESUME DATA: {state.retrieved_chunks}"""

    response = llm.invoke(prompt)
    return {"candidate_summary": response.content}


# WRITE COVER LETTER (The Copywriter)
def write_cover_letter_node(state: GraphState, llm) -> dict:
    """
    The Copywriter - Drafts the initial cover letter with STRICT grounding.
    """
    
    # Extract verifiable facts for reference
    source_evidence = "\n".join([
        f"- {chunk['text']}" for chunk in state.retrieved_chunks
    ])
    
    prompt = f"""Write ATS-optimized cover letter (300-350 words):

    CONSTRAINTS (STRICT):
    - Use ONLY skills/experience from Verified Qualifications below
    - No invention/assumption of qualifications
    - If unmatched requirement: express willingness to learn OR skip
    - Every metric must trace to Resume Evidence
    - No forbidden phrases: "Source:", "Job Requirements:", "According to", parenthetical citations

    STRUCTURE:
    1. Opening: reference {state.company_name if hasattr(state, 'company_name') else '[Company Name]'} using company research
    2. Body 1: match 2-3 strengths to job requirements (with evidence)
    3. Body 2: problem-solving fit with concrete examples
    4. Closing: strong call-to-action

    FORMAT: Active voice, 2-4 metrics, mirror 5-7 JD keywords, use [{state.company_name if hasattr(state, 'company_name') else 'Company Name'}] placeholder.

    VERIFIED QUALIFICATIONS (Ground Truth): {state.candidate_summary}

    RESUME EVIDENCE: {source_evidence}

    JOB REQUIREMENTS: {state.cleaned_jd}

    COMPANY RESEARCH: {state.company_research if hasattr(state, 'company_research') else '[Research pending]'}

    Return cover letter body only."""
    
    response = llm.invoke(prompt)
    return {"cover_letter": response.content}


# CRITIQUE LETTER (The Hiring Manager)
def critique_letter_node(state: GraphState, llm) -> dict:
    """
    The Hiring Manager - Reviews letter for quality and authenticity.
    
    Purpose:
        - Identifies generic "AI-sounding" phrases
        - Checks for specific evidence backing claims
        - Provides actionable feedback for refinement
    
    Input: 
        - state.cover_letter - Draft letter
        - state.cleaned_jd - Job requirements
    Output: 
        - state.critique_feedback - Structured improvement suggestions
        - state.needs_refinement - Boolean flag for rewrite loop
    """
    prompt = f"""Critique cover letter against job requirements:

    CRITERIA:
    - Specificity: concrete examples/metrics?
    - Originality: avoid clichés ("passionate", "team player")?
    - Relevance: addresses top 3 requirements?
    - Tone: confident not arrogant?
    - ATS: critical keywords natural?
    - Hallucination: only verified resume claims?

    COVER LETTER: {state.cover_letter}

    JOB REQUIREMENTS: {state.cleaned_jd}

    OUTPUT:
    Strengths: 2-3 positives
    Issues to Fix: numbered, with line references
    Recommended Changes: specific rewrites
    Ready to Submit?: YES/NO"""
    
    response = llm.invoke(prompt)
    
    # Determine if refinement is needed
    needs_refinement = "NO" in response.content.split("Ready to Submit?")[-1]
    
    return {
        "critique_feedback": response.content,
        "needs_refinement": needs_refinement
    }


# REFINE LETTER (The Editor)

def refine_letter_node(state: GraphState, llm) -> dict:
    """
    The Editor - Rewrites letter based on critique feedback.
    
    Purpose:
        - Implements specific improvements from critique
        - Maintains original strengths while fixing issues
        - Iterates up to 2-3 times for polish
    
    Input: 
        - state.cover_letter - Current draft
        - state.critique_feedback - Specific improvement requests
    Output: 
        - state.cover_letter - Refined version (overwrites)
        - state.refinement_count - Iteration tracker
    """
    current_count = state.refinement_count + 1
    
    prompt = f"""Revise cover letter (iteration {current_count}/3):
    - Address EVERY "Issues to Fix"
    - Implement "Recommended Changes" exactly
    - Preserve strengths, structure, word count
    - Ensure natural flow

    CURRENT DRAFT: {state.cover_letter}

    CRITIQUE: {state.critique_feedback}

    Return revised letter only. No commentary."""
    response = llm.invoke(prompt)
    
    return {
        "cover_letter": response.content,
        "refinement_count": current_count
    }

def rewrite_query_node(state: GraphState, llm) -> dict:
    """Uses LLM to improve the search query based on critic feedback."""
    prompt = f"""Previous search failed. Rewrite query for better vector DB matching using technical keywords and core requirements.

    ORIGINAL: {state.cleaned_jd}

    FEEDBACK: {state.grading_feedback}"""

    response = llm.invoke(prompt)
    return {
        "cleaned_jd": response.content,
        "rewrite_count": state.rewrite_count + 1 
    }
