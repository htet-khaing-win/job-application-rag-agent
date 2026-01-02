from langgraph.graph import StateGraph, END, START
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from state import GraphState


# INGEST JOB DESCRIPTION (Entry Node)
def ingest_jd_node(state: GraphState, llm) -> GraphState:

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
    if validation_prompt == "INVALID":
        return {
            **state,
            "is_valid_jd": False,
            "error_type": "invalid_input",
            "error_message": "This request doesn't appears to be a job description. Please paste a complete job posting."
        }
    
    cleaning_prompt = f"""
    ROLE: You are an expert HR Document Analyst with 10+ years of experience parsing job descriptions.
    
    TASK: Clean and extract the essential components from this Job Description.
    
    INPUT - Job Description: {state.job_description}
    
    INSTRUCTIONS:
    1. Remove generic HR language ("competitive salary", "great culture", etc.)
    2. Extract MUST-HAVE technical skills and qualifications
    3. Identify key responsibilities and measurable outcomes
    4. List industry-specific keywords for ATS optimization
    5. Organize output into clear sections
    
    OUTPUT FORMAT:
    Core Requirements: [List of must-have skills and qualifications]
    Key Responsibilities: [Primary job duties]
    Target Keywords: [Comma-separated list of critical terms for matching]
    """
    
    response = llm.invoke(cleaning_prompt)
    return {
        **state,
        "cleaned_jd": response.content,
        "is_valid_jd": True
    }


# GRADE RETRIEVAL (The Critic)
def grade_retrieval_node(state: GraphState, llm) -> GraphState:
    """
    The Critic - Evaluates quality of retrieved resume matches.
    
    Purpose:
        - Scores each retrieved chunk against Job Description requirements
        - Determines if results meet quality threshold (e.g., >70% relevance)
        - Triggers rewrite_query if quality is insufficient
    
    Input: 
        - state.cleaned_jd - Target job requirements
        - state.retrieved_chunks - Candidate resume data
    Output: 
        - state['relevance_score'] - Overall match quality (0-100)
        - state['needs_rewrite'] - Boolean flag for query refinement
    """
    prompt = f"""
    ROLE: You are a Technical Recruiter with expertise in candidate-role matching.
    
    TASK: Grade the relevance of retrieved resume content against job requirements.
    
    INPUT - Job Requirements: {state.cleaned_jd}
    
    INPUT - Retrieved Resume Chunks: {state.retrieved_chunks}
    
    INSTRUCTIONS:
    1. Score each chunk on relevance (0-100)
    2. Check for critical skill gaps
    3. Calculate overall match percentage
    4. Determine if results are sufficient to proceed (threshold: 70%)
    
    OUTPUT FORMAT:
    Score: [0-100]
    Needs Rewrite: [YES/NO]
    Reasoning: [Brief explanation]
    """
    response = llm.invoke(prompt)
    
    # Parse response to extract score and rewrite flag
    content = response.content
    score = 75  # TODO: Extract from LLM response
    needs_rewrite = "NO" in content.upper()
    
    return {
        "relevance_score": score,
        "needs_rewrite": needs_rewrite,
        "grading_feedback": content
    }


# GENERATE SUMMARY (The Analyst)
def generate_summary_node(state: GraphState, llm) -> GraphState:
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
    prompt = f"""
    ROLE: You are a Career Coach and Resume Writer with 10+ years of experience.
    
    TASK: Create a comprehensive candidate-job match analysis.
    
    INPUT - Job Requirements: {state.cleaned_jd}
    
    INPUT - Candidate Resume Data: {state.retrieved_chunks}
    
    INSTRUCTIONS:
    1. Map candidate's top 5 achievements to specific job requirements
    2. Highlight quantifiable results (metrics, percentages, revenue impact)
    3. Identify unique value propositions (what sets candidate apart)
    4. Note any skill gaps and how to address them
    5. Write in third person, professional tone
    
    OUTPUT FORMAT:
    Key Qualifications Match: Requirement and Candidate Evidence mapping
    Standout Achievements: Top 3 accomplishments with metrics
    Value Proposition: 2-3 sentences on unique fit
    """
    response = llm.invoke(prompt)
    return {"candidate_summary": response.content}


# WRITE COVER LETTER (The Copywriter)
def write_cover_letter_node(state: GraphState, llm) -> GraphState:
    """
    The Copywriter - Drafts the initial cover letter.
    
    Purpose:
        - Transforms candidate summary into persuasive narrative
        - Uses impact-driven language and standard business format
        - Incorporates ATS-friendly keywords from Job Description
    
    Input: 
        - state.candidate_summary - Match analysis
        - state.cleaned_jd - Job requirements
    Output: 
        - state.cover_letter - Draft cover letter text
    """
    prompt = f"""
    ROLE: You are a Senior Career Consultant specializing in Tech industry applications.
    
    TASK: Write a compelling, ATS-optimized cover letter.
    
    INPUT - Candidate Summary: {state.candidate_summary}
    
    INPUT - Job Requirements: {state.cleaned_jd}
    
    INSTRUCTIONS:
    1. Opening: Hook with specific company research or mutual connection or offer solution for specific problem mentioned in the Job Description
    2. Body Paragraph 1: Demonstrate fit, solution and unique value relevant to Job Requirements and Preferred Skill sets
    3. Body Paragraph 2: Address top 2-3 technical requirements with evidence 
    4. Closing: Strong call-to-action requesting to review the Resume for more details
    
    REQUIREMENTS:
    - Use active voice and strong verbs (led, achieved, delivered)
    - Include 3-5 specific metrics/achievements
    - Mirror 5-7 keywords from job description
    - Keep to 300-350 words (3-4 paragraphs)
    - Use placeholders: [Company Name], [Hiring Manager Name], [Your Name]
    
    OUTPUT FORMAT:
    Return ONLY the cover letter body text. No preamble.
    """
    response = llm.invoke(prompt)
    return {"cover_letter": response.content}


# CRITIQUE LETTER (The Hiring Manager)
def critique_letter_node(state: GraphState, llm) -> GraphState:
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
    prompt = f"""
    ROLE: You are a Hiring Manager reviewing a cover letter for authenticity and impact.
    
    TASK: Critique this cover letter against the job requirements.
    
    INPUT - Cover Letter: {state.cover_letter}
    
    INPUT - Job Requirements: {state.cleaned_jd}
    
    EVALUATION CRITERIA:
    1. Specificity: Are claims backed by concrete examples/metrics?
    2. Originality: Does it avoid clichés like "passionate," "team player"?
    3. Relevance: Does it address the top 3 job requirements directly?
    4. Tone: Is it confident but not arrogant?
    5. ATS Optimization: Does it include critical keywords naturally?
    
    OUTPUT FORMAT:
    Strengths: 2-3 specific positives
    Issues to Fix: Numbered list of concrete problems with line references
    Recommended Changes: Specific rewrites or additions needed
    Ready to Submit?: [YES/NO]
    """
    response = llm.invoke(prompt)
    
    # Determine if refinement is needed
    needs_refinement = "NO" in response.content.split("Ready to Submit?")[-1]
    
    return {
        "critique_feedback": response.content,
        "needs_refinement": needs_refinement
    }


# REFINE LETTER (The Editor)

def refine_letter_node(state: GraphState, llm) -> GraphState:
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
    
    prompt = f"""
    ROLE: You are a Professional Editor specializing in business correspondence.
    
    TASK: Revise this cover letter based on the provided critique.
    
    INPUT - Current Draft: {state.cover_letter}
    
    INPUT - Critique Feedback: {state.critique_feedback}
    
    INSTRUCTIONS:
    1. Address EVERY issue listed in "Issues to Fix"
    2. Implement "Recommended Changes" exactly as specified
    3. Preserve the strengths identified in the critique
    4. Maintain the original structure and word count
    5. Ensure natural flow after edits
    
    REFINEMENT ITERATION: {current_count}/3
    
    OUTPUT FORMAT:
    Return ONLY the revised cover letter text. No commentary.
    """
    response = llm.invoke(prompt)
    
    return {
        "cover_letter": response.content,
        "refinement_count": current_count
    }

