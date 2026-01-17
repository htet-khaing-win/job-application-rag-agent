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
        - Separates skills with project evidence vs. skills only listed
    """
    # Precondition check
    if not state.candidate_summary.strip():
        return {
            "candidate_summary": "",
            "verification_log": "No candidate summary provided for verification.",
            "verified_skills": [],
            "unverified_skills": [],
            "verified_skills_detailed": {"with_evidence": [], "no_evidence": []},
            "error_type": "verification_failed",
            "error_message": "Candidate summary was empty before verification."
        }
    
    # Extract actual text from retrieved chunks for verification
    source_text = "\n".join([
        chunk["text"] for chunk in state.retrieved_chunks
    ])
    
    verification_prompt = f"""You are a fact-checker validating a candidate summary against resume evidence.

        VALIDATION RULES: 
        ACCEPT if skill appears ANYWHERE in resume:
        - In Skills Section: "verified_skills, verified_skills, verified_skills" -> Accept all three
        - In Experience Section: "Built pipeline with verified_skills_detailed" -> Accept with evidence
        - Classification:
            * WITH EVIDENCE: Skill used in project -> Allow specific claims with metrics
            * WITHOUT EVIDENCE: Skill only in skills list -> Allow generic mention only

        REJECT only if:
        - Technology not mentioned anywhere in resume
        - Exaggerated expertise levels without evidence (expert, senior, advanced)

        SOURCE RESUME DATA: {source_text}

        CANDIDATE SUMMARY TO VERIFY: {state.candidate_summary}

        OUTPUT FORMAT (strict JSON):
        {{
        "verified_skills_with_evidence": [
            {{"skill": "verified_skills_detailed", "evidence": "Built RAG pipeline using verified_skills_detailed for ...", "has_project": true}},
            {{"skill": "verified_skills_detailed", "evidence": "Developed backend services using verified_skills_detailed", "has_project": true}}
        ],
        "verified_skills_no_evidence": [
            {{"skill": "verified_skills_no_evidence", "source": "Listed in Skills section", "has_project": false}},
            {{"skill": "verified_skills_no_evidence", "source": "Mentioned in resume", "has_project": false}}
        ],
        "unverified_skills": [
            {{"skill": "unverified_skills", "reason": "unverified_skills mentioned but no expertise level stated"}}
        ],
        "removed_claims": [
            {{"claim": "unverified_skills", "reason": "No unverified_skills mentioned in resume"}}
        ],
        "verified_summary": "Rewritten summary using verified skills with specific evidence for skills with projects, and brief mentions for skills without project evidence. Professional tone, grounded in resume facts."
        }}

        IMPORTANT: Be generous with verification if evidence exists, but strict about exaggerations and absent technologies.

        Return ONLY the JSON object."""
    
    response = llm.invoke(verification_prompt)
    content = response.content.strip()
    
    # Parse JSON response
    content = content.replace("```json", "").replace("```", "").strip()
    
    try:
        data = json.loads(content)
        verified_summary = data.get("verified_summary", "")
        
        # Get the new categorized skill lists
        skills_with_evidence = data.get("verified_skills_with_evidence", [])
        skills_no_evidence = data.get("verified_skills_no_evidence", [])
        unverified_skills_raw = data.get("unverified_skills", [])
        removed_claims = data.get("removed_claims", [])
        
        # Build verified_skills list (simple strings for compatibility)
        verified_skills = []
        for item in skills_with_evidence:
            if isinstance(item, dict):
                verified_skills.append(item.get("skill", ""))
        
        for item in skills_no_evidence:
            if isinstance(item, dict):
                verified_skills.append(item.get("skill", ""))
        
        # Build unverified_skills list (simple strings)
        unverified_skills = []
        for item in unverified_skills_raw:
            if isinstance(item, dict):
                unverified_skills.append(item.get("skill", ""))
            else:
                unverified_skills.append(str(item))
        
        # Store the detailed breakdown for cover letter node
        verified_skills_detailed = {
            "with_evidence": skills_with_evidence,
            "no_evidence": skills_no_evidence
        }
        
        MIN_SUMMARY_LENGTH = 100
        
        # Enhanced logging
        if removed_claims:
            print(f" Removed {len(removed_claims)} unverified claims:")
            for claim in removed_claims[:3]:  # Show first 3
                print(f"  - {claim.get('claim', 'Unknown')}")
        
        if skills_with_evidence:
            print(f" Skills with evidence: {len(skills_with_evidence)}")
            for skill in skills_with_evidence[:3]:
                print(f"  - {skill.get('skill', 'Unknown')}: {skill.get('evidence', '')[:60]}...")
        
        if skills_no_evidence:
            skill_names = [s.get('skill', '') for s in skills_no_evidence]
            print(f" Skills without project evidence: {', '.join(skill_names[:5])}")
        
        if unverified_skills:
            print(f" Unverified skills: {', '.join(unverified_skills[:5])}")
        
        # CRITICAL: Always return the verified summary, never the original
        if len(verified_summary) < MIN_SUMMARY_LENGTH:
            return {
                "candidate_summary": verified_summary,  # Still use verified even if short
                "verification_log": json.dumps(data, indent=2),
                "verified_skills": verified_skills,
                "unverified_skills": unverified_skills,
                "verified_skills_detailed": verified_skills_detailed,
                "error_type": "verification_failed",
                "error_message": f"Only {len(verified_skills)} skills could be verified. Consider uploading a more relevant resume."
            }
        
        return {
            "candidate_summary": verified_summary,  # ENFORCED: Only verified content
            "verification_log": json.dumps(data, indent=2),
            "verified_skills": verified_skills,  # Simple strings: ["Python", "Pinecone"]
            "unverified_skills": unverified_skills,  # Simple strings
            "verified_skills_detailed": verified_skills_detailed  # Rich format with evidence flags
        }
        
    except json.JSONDecodeError as e:
        print(f" JSON Parse Error in verification: {e}")
        print(f"Raw LLM Output: {content[:300]}")
        
        # Fallback: Extract verified summary with regex
        if "verified_summary" in content.lower():
            match = re.search(r'"verified_summary":\s*"([^"]+)"', content)
            if match:
                verified_summary = match.group(1)
                return {
                    "candidate_summary": verified_summary,
                    "verification_log": content,
                    "verified_skills": [],
                    "unverified_skills": [],
                    "verified_skills_detailed": {"with_evidence": [], "no_evidence": []}
                }
        
        # Last resort: Return error
        return {
            "candidate_summary": "",
            "verification_log": content,
            "verified_skills": [],
            "unverified_skills": [],
            "verified_skills_detailed": {"with_evidence": [], "no_evidence": []},
            "error_type": "verification_failed",
            "error_message": "Verification system malfunction. Please retry."
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
        - state.error_type == "hallucination_critical" (unverifiable claims)
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
        
    elif state.error_type == "hallucination_critical":
        message = f"""
        Quality Check Failed: The system detected unverified skills in the generated letter.

        Issue: {state.error_message}

        Solutions:
        1. Upload a more relevant resume with experience in the required technologies
        2. Try a different job posting that better matches your background
        3. Ensure your resume clearly describes your project experience
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
    Enhanced with multi-layer hallucination prevention.
    """
    
    if state.vector_relevance_score is None or state.vector_relevance_score <= 0:
        raise RuntimeError(
            "Invariant violated: cover letter generation attempted without valid retrieval score."
        )

    # Build skill lists for explicit constraint
    skills_with_evidence = state.verified_skills_detailed.get("with_evidence", [])
    skills_no_evidence = state.verified_skills_detailed.get("no_evidence", [])

    # Create explicit evidence blocks
    skills_with_ev_text = "\n".join([
        f"- {s['skill']}: {s['evidence']}" for s in skills_with_evidence
    ]) if skills_with_evidence else "None"

    skills_no_ev_text = ", ".join([
        s['skill'] for s in skills_no_evidence
    ]) if skills_no_evidence else "None"

    unverified_skills_list = ", ".join(state.unverified_skills) if state.unverified_skills else "None"
    
    # Identify which unverified skills the job actually requires
    job_required_unverified = []
    if state.unverified_skills:
        jd_lower = state.cleaned_jd.lower()
        for skill in state.unverified_skills:
            if skill.lower() in jd_lower:
                job_required_unverified.append(skill)
    
    transfer_learning_guidance = ""
    if job_required_unverified:
        transfer_learning_guidance = f"""
    UNVERIFIED SKILLS REQUIRED BY JOB (Address via transfer learning only):
    {', '.join(job_required_unverified[:3])}

    TRANSFER LEARNING TEMPLATE:
    "While I haven't used [UNVERIFIED_SKILL] professionally, my experience with [VERIFIED_SKILL] demonstrates the same [COMPETENCY] required. For example, in [VERIFIED_PROJECT], I [ACHIEVEMENT], which prepared me to quickly adapt to new tools."

    DO NOT claim direct experience with these skills.
    """
    
    prompt = f"""You are writing a factual cover letter with zero tolerance for hallucination.

    SKILLS YOU CAN USE (with specific project details): {skills_with_ev_text}

    SKILLS YOU CAN MENTION (brief, generic mention only): {skills_no_ev_text}

    FORBIDDEN SKILLS (never claim direct experience):
    {unverified_skills_list}

    {transfer_learning_guidance}

    ABSOLUTE RULES:
    1. ONLY use skills from "SKILLS YOU CAN USE" section with their specific evidence
    2. For "SKILLS YOU CAN MENTION": Brief mention only, no elaboration
    3. For "FORBIDDEN SKILLS" required by job: Use transfer learning template ONLY
    4. If a skill is in FORBIDDEN list, you CANNOT claim proficiency
    5. Every technical claim must cite the EXACT project name from verified evidence
    6. Forbidden phrases: "proficient in", "extensive experience in", "expert in"

    STRUCTURE (300-350 words):

    Paragraph 1 - Opening (40-50 words):
    - Hook with company-specific insight from research
    - State position + ONE metric from verified evidence

    Paragraph 2 - Skills Match (120-140 words):
    - Pick 2-3 skills from "SKILLS YOU CAN USE" section
    - Include exact project names and metrics
    - Mirror job keywords naturally

    Paragraph 3 - Growth/Adaptability (60-80 words):
    - IF job requires FORBIDDEN skills: Use transfer learning (max 2 skills)
    - IF no FORBIDDEN skills: Expand on verified achievements

    Paragraph 4 - Closing (40-50 words):
    - Call-to-action referencing company initiative
    - Company name: [{state.company_name or 'Company Name'}]

    YOUR ONLY SOURCE OF TRUTH:

    VERIFIED CANDIDATE SUMMARY: {state.candidate_summary}

    JOB REQUIREMENTS: {state.cleaned_jd}

    COMPANY RESEARCH: {state.company_research if hasattr(state, 'company_research') else '[Research pending]'}

    OUTPUT REQUIREMENTS:
    - Return cover letter body only (no "Dear Hiring Manager")
    - First person, active voice, professional tone
    - Include 2-4 specific metrics
    - No bullet points, no parenthetical citations

    CRITICAL: If you cannot find verified evidence for a skill, DO NOT mention it."""
    
    response = llm.invoke(prompt)
    letter_content = response.content.strip()
    
    # POST-GENERATION HALLUCINATION CHECK
    hallucination_detected = False
    hallucinated_skills = []
    
    if state.unverified_skills:
        print("\nRunning hallucination detection...")
        
        for skill in state.unverified_skills:
            skill_pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            
            if re.search(skill_pattern, letter_content.lower()):
                match = re.search(skill_pattern, letter_content.lower())
                if match:
                    start = max(0, match.start() - 50)
                    end = min(len(letter_content), match.end() + 50)
                    context = letter_content[start:end].lower()
                    
                    transfer_phrases = [
                        "haven't used", "haven't worked with", "new to", 
                        "eager to learn", "while my experience is in",
                        "while i haven't", "although i haven't",
                        "prepared me to", "ready to learn", "demonstrates the same",
                        "looking forward to learning"
                    ]
                    
                    has_valid_context = any(phrase in context for phrase in transfer_phrases)
                    
                    if not has_valid_context:
                        hallucination_detected = True
                        hallucinated_skills.append(skill)
                        print(f"  HALLUCINATION: '{skill}' claimed without transfer learning")
                        print(f"  Context: ...{context}...")
    
    # ENFORCEMENT: Block hallucinated content
    if hallucination_detected:
        print(f"\nCOVER LETTER REJECTED")
        print(f"Reason: {len(hallucinated_skills)} hallucinated skill(s) detected")
        print(f"Skills: {', '.join(hallucinated_skills)}\n")
        
        # AUTO-FIX: Remove sentences containing hallucinations
        cleaned_letter = letter_content
        removed_sentences = []
        
        for skill in hallucinated_skills:
            sentences = cleaned_letter.split('.')
            filtered_sentences = []
            
            for sentence in sentences:
                skill_pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if not re.search(skill_pattern, sentence.lower()):
                    filtered_sentences.append(sentence)
                else:
                    removed_sentences.append(sentence.strip())
            
            cleaned_letter = '.'.join(filtered_sentences)
        
        if removed_sentences:
            print("Removed sentences:")
            for sent in removed_sentences[:3]:
                print(f"  - {sent[:100]}...")
        
        word_count = len(cleaned_letter.split())
        if word_count < 200:
            return {
                "cover_letter": "",
                "error_type": "hallucination_critical",
                "error_message": f"Cover letter contained unverified claims about: {', '.join(hallucinated_skills)}. Auto-fix resulted in insufficient content ({word_count} words). Please upload a more relevant resume or try a different job posting."
            }
        
        print(f"Auto-fixed: {word_count} words remaining\n")
        letter_content = cleaned_letter
    else:
        print("Hallucination check passed - all claims verified\n")
    
    return {"cover_letter": letter_content}


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
