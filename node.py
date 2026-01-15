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
        "cleaned_jd": response.content,
        "is_valid_jd": True
    }


# def extract_company_name_node(state: GraphState, llm) -> dict:
#     """
#     Attempts to extract company name from JD, but marks as requiring user confirmation.
#     """
    
#     prompt = f"""
#     ROLE: You are an HR document parser.

#     TASK: Extract the company name from this job description.

#     INPUT:
#     {state.cleaned_jd}

#     INSTRUCTIONS:
#     1. Look for explicit company identification (e.g., "About [Company]", "Join [Company]")
#     2. If no company name is found, return "UNKNOWN"
#     3. Return ONLY the company name, no explanation

#     OUTPUT: [Company Name] or UNKNOWN
#     """
    
#     response = llm.invoke(prompt)
#     extracted_name = response.content.strip().replace('"', '')
    
#     # Mark as requiring confirmation
#     return {
#         "suggested_company_name": extracted_name,
#         "needs_company_confirmation": True
#     }


def research_company_node(state: GraphState, llm) -> dict:
    """
    Uses Tavily AI to fetch real-time company information. This runs AFTER user confirms the company name.
    
    Purpose:
        - Fetches company mission, values, recent news
        - Provides grounding for personalized cover letter opening
        - Reduces generic "I'm excited about [Company]" statements
    """
    
    company_name = state.company_name
    
    # if not company_name or company_name == "UNKNOWN":
    #     return {
    #         "company_research": "No company information available.",
    #         "company_research_success": False
    #     }
    
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
        synthesis_prompt = f"""
    ROLE: You are a Company Research Analyst.

    TASK: Synthesize this raw company research into actionable insights for a cover letter.

    RAW RESEARCH DATA:
    {research_text}

    INSTRUCTIONS:
    1. Extract 2-3 key company values or mission statements
    2. Identify 1-2 recent achievements, initiatives, or news items
    3. Note any unique company culture aspects
    4. Keep output concise (100-150 words)

    OUTPUT FORMAT:
    Company Overview: [2-3 sentences on mission/values]
    Recent Developments: [1-2 notable items from past 6 months]
    Culture Insights: [Any relevant workplace culture notes]
    """
        
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
    
    verification_prompt = f"""
    ROLE: You are a Fact-Checker for resume verification.

    TASK: Validate that EVERY claim in the Candidate Summary is directly supported by the Source Resume Data.

    SOURCE RESUME DATA (Ground Truth):
    {source_text}

    CANDIDATE SUMMARY TO VERIFY:
    {state.candidate_summary}

    INSTRUCTIONS:
    1. Extract each specific claim about skills, experience, and achievements
    2. When mentioning skills, experience, cite specific projects from the resume that demonstrate those skills (e.g., 'demonstrated expertise in [Project Name]').
    3. Mark claims as VERIFIED or UNSUPPORTED
    4. If a claim cannot be directly traced to source data, it must be removed

    STRICT RULES:
    - Do NOT infer skills (e.g., if resume says "used Python", don't claim "expert in Python")
    - Do NOT add qualifications not explicitly stated
    - Do NOT assume years of experience beyond what's documented
    - Do NOT extrapolate technologies (e.g., if they used x, don't claim y unless stated)

    OUTPUT FORMAT:
    Verified Claims:
    - [Claim 1]: [Supporting quote from source]
    - [Claim 2]: [Supporting quote from source]

    Removed Claims (Unsupported):
    - [Claim X]: [Reason for removal]

    Final Verified Summary:
    [Rewrite the candidate summary using ONLY verified claims]
    """
    
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
        - state.relevance_score < 50 after 2 rewrites (severe mismatch)
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
    
    elif state.relevance_score < 50:
        message = f"""
    I couldn't find a strong match between your resumes and this job description (relevance: {state.relevance_score}%).

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
    
    prompt = f"""
    ROLE: You are a Technical Recruiter with expertise in candidate-role matching.

    TASK: Score the relevance of retrieved resume content against job requirements.

    JOB REQUIREMENTS:
    {state.cleaned_jd}

    RETRIEVED RESUME CHUNKS:
    {state.retrieved_chunks}

    INSTRUCTIONS:
    1. Evaluate how well the resume chunks collectively match the job requirements
    2. Assign a score from 0-100 where:
    - 90-100: Exceptional match, all key requirements covered with strong evidence
    - 70-89: Good match, most requirements covered
    - 50-69: Moderate match, some requirements met but gaps exist
    - 30-49: Weak match, significant gaps in qualifications
    - 0-29: Poor match, candidate lacks most requirements

    3. Provide brief reasoning explaining the score

    CRITICAL: Focus on what IS present in the resume data, not what's missing.

    OUTPUT FORMAT (strict JSON):
    {{
        "score": <integer 0-100>,
        "reasoning": "<2-3 sentence explanation>"
    }}

    Do not add markdown formatting, code blocks, or any other text outside the JSON object.
    """
    
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
        "relevance_score": score,
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
def write_cover_letter_node(state: GraphState, llm) -> dict:
    """
    The Copywriter - Drafts the initial cover letter with STRICT grounding.
    """
    
    # Extract verifiable facts for reference
    source_evidence = "\n".join([
        f"- {chunk['text']}" for chunk in state.retrieved_chunks
    ])
    
    prompt = f"""
    ROLE: You are a Senior Career Consultant specializing in Tech industry applications.

    TASK: Write a compelling, ATS-optimized cover letter.

    VERIFIED CANDIDATE QUALIFICATIONS (Ground Truth - Use ONLY This):
    {state.candidate_summary}

    SOURCE RESUME EVIDENCE (Reference for Verification):
    {source_evidence}

    JOB REQUIREMENTS:
    {state.cleaned_jd}

    COMPANY RESEARCH (if available):
    {state.company_research if hasattr(state, 'company_research') else '[Research pending]'}

    CRITICAL CONSTRAINTS:
    1. You may ONLY reference skills, experience, and achievements explicitly stated in "Verified Candidate Qualifications"
    2. Do NOT invent or assume any qualifications not present in the source data
    3. If a job requirement cannot be matched to candidate qualifications, either:
    a) Express enthusiasm to learn/develop that skill, OR
    b) Skip mentioning it entirely
    4. Every metric or achievement must have a direct quote source from resume evidence

    LETTER STRUCTURE:
    1. Opening: Reference {state.company_name if hasattr(state, 'company_name') else '[Company Name]'} specifically using company research insights
    2. Body Paragraph 1: Match top 2-3 candidate strengths to job requirements (with evidence)
    3. Body Paragraph 2: Demonstrate problem-solving fit with concrete examples
    4. Closing: Strong call-to-action referencing specific role title

    FORMATTING REQUIREMENTS:
    - Use active voice and strong verbs (led, achieved, delivered)
    - Include 2-4 specific metrics/achievements from verified data only
    - Mirror 5-7 keywords from job description naturally
    - Keep to 300-350 words (3-4 paragraphs)
    - Use placeholders: [{state.company_name if hasattr(state, 'company_name') else 'Company Name'}]

    VERIFICATION CHECK:
    Before finalizing, ensure every claim can be traced back to "Source Resume Evidence".

    OUTPUT FORMAT:
    Return ONLY the cover letter body text. No preamble or commentary.
    """
    
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

def rewrite_query_node(state: GraphState, llm) -> dict:
    """Uses LLM to improve the search query based on critic feedback."""
    prompt = f"""
    The previous search for resumes failed. 
    Original Query: {state.cleaned_jd}
    Critic Feedback: {state.grading_feedback}
    
    Task: Rewrite the search query to be more effective at finding relevant resume chunks in a vector database.
    Focus on technical keywords and core requirements.
    """
    response = llm.invoke(prompt)
    return {
        "cleaned_jd": response.content,
        "rewrite_count": state.rewrite_count + 1 
    }
