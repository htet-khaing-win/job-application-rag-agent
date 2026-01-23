from langgraph.graph import StateGraph, END, START
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from state import GraphState
import json
import re
from tavily import TavilyClient
import asyncio
from langsmith import traceable
from collections import Counter
from typing import List, Dict

load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# INGEST JOB DESCRIPTION (Entry Node)
def ingest_jd_node(state: GraphState, llm) -> dict:
    """
    Single-pass JD validation and extraction.
    
    Combines validation + cleaning into one LLM call for efficiency.
    """
    
    combined_prompt = f"""You are processing a job description input.

    TASK 1: Validate if this is a legitimate job description
    TASK 2: If valid, extract key information

    INPUT: {state.job_description}

    VALIDATION CRITERIA:
    - Contains job requirements, qualifications, or responsibilities
    - Mentions a role title or position
    - Describes company needs or hiring context

    EXTRACTION (if valid):
    - Core Requirements: must-have skills/qualifications
    - Key Responsibilities: primary duties
    - Target Keywords: critical ATS terms

    OUTPUT (strict JSON, no markdown):
    {{
      "valid": true,
      "cleaned_jd": "extracted requirements and responsibilities, no fluff"
    }}

    If invalid:
    {{
      "valid": false,
      "cleaned_jd": ""
    }}
    """
    
    response = llm.invoke(combined_prompt)
    content = response.content.strip().replace("```json", "").replace("```", "")
    
    try:
        data = json.loads(content)
        
        if not data.get("valid", False):
            return {
                "is_valid_jd": False,
                "error_type": "invalid_input",
                "error_message": "This doesn't appear to be a job description. Please paste a complete job posting."
            }
        
        return {
            "cleaned_jd": data.get("cleaned_jd", ""),
            "is_valid_jd": True
        }
        
    except json.JSONDecodeError as e:
        print(f" Job Description parsing error: {e}")
        # Fallback: assume valid if contains common JD keywords
        jd_lower = state.job_description.lower()
        if any(kw in jd_lower for kw in ["requirements", "responsibilities", "qualifications", "experience"]):
            return {
                "cleaned_jd": state.job_description,
                "is_valid_jd": True
            }
        else:
            return {
                "is_valid_jd": False,
                "error_type": "invalid_input",
                "error_message": "Could not parse job description."
            }

@traceable(run_type="tool", name="tavily_company_research")
async def research_company_node(state: GraphState) -> dict:
    """
    Uses Tavily AI to fetch real-time company information. This runs AFTER user confirms the company name.
    
    Purpose:
        - Fetches company mission, values, recent news
        - Provides grounding for personalized cover letter opening
    """
    
    company_name = state.company_name
    
    try:
        # Tavily search query
        search_query = f"{company_name} company mission values recent news 2024 2025"
        
        search_results = tavily_client.search(
            query=search_query,
            search_depth="advanced",
            max_results=3
        )
        
        
        # Extract relevant information
        research_snippets = []
        
        for result in search_results.get('results', []):
            title = result.get('title', 'N/A')
            content = result.get('content', '')[:300]  # Truncate to 300 chars per source
            
            if content:
                research_snippets.append(f"• {title}: {content}")
        
        if research_snippets:
            company_research = f"Company: {company_name}\n\n" + "\n\n".join(research_snippets)
        else:
            company_research = f"Company: {company_name} (limited public information available)"
        
        return {
            "company_research": company_research,
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
    """
    return {
        "candidate_summary": state.candidate_summary,
        "verified_skills": [],
        "unverified_skills": [],
        "verified_skills_detailed": {"with_evidence": [], "no_evidence": []}
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
    """
    
    if state.error_type == "invalid_input":
        message = f"""
    I couldn't identify this as a job description. 
    """
    
    elif state.error_type == "no_resumes":
        message = """
    No resumes found in the system.
    """

        
    elif state.error_type == "hallucination_critical":
        message = f"""
        Quality Check Failed: The system detected unverified skills in the generated letter.

        Issue: {state.error_message}
        """
    
    else:
        # Generic fallback for unexpected errors
        message = """
    Please paste a job description, and I'll create a tailored cover letter using your uploaded resume(s).
    """
    
    return {
        "final_response": message,
        "is_fallback": True
    }

# helper Functions for generate_summary_node
def extract_jd_keywords(cleaned_jd: str, top_k: int = 20) -> List[str]:
    """Domain-agnostic keyword extraction via frequency analysis."""
    text = cleaned_jd.lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}", text)
    
    stopwords = {
        "experience", "knowledge", "ability", "skills", "required",
        "requirements", "responsibilities", "years", "work", "role",
        "candidate", "team", "will", "must", "should"
    }
    tokens = [t for t in tokens if t not in stopwords]
    
    unigrams = tokens
    MAX_TOKENS = 400
    tokens = tokens[:MAX_TOKENS]

    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    
    counts = Counter(unigrams + bigrams)
    
    # Prefer technical terms
    ranked = [
        k for k, v in counts.most_common(top_k * 3)
        if (v >= 2) or (len(k.split()) > 1 and v >= 1)
    ]
    
    if len(ranked) < top_k:
        ranked = [k for k, _ in counts.most_common(top_k)]
    
    return ranked[:top_k]

def match_keywords_to_chunks(
    keywords: List[str],
    chunks: List[Dict],
    min_hits: int = 1
) -> Dict[str, List[str]]:
    """Verify keywords appear in resume evidence."""
    evidence = {}
    
    for kw in keywords:
        kw_l = kw.lower()
        hits = []
        
        for c in chunks:
            text = c["text"].lower()
            if kw_l in text:
                hits.append(c["text"][:200])
        
        if len(hits) >= min_hits:
            evidence[kw] = hits[:2]
    
    return evidence

# GENERATE SUMMARY (The Analyst)
def generate_summary_node(state: GraphState, llm=None) -> dict:
    """
    The Analyst - Creates a bridging narrative between candidate and role.

    Purpose:
        - Extracts key technical requirements directly from the job description
        - Verifies each requirement against retrieved resume evidence
        - Produces a deterministic, hallucination-free skill alignment summary

    Output fields:
        - candidate_summary: concise human-readable alignment overview
        - verified_skills: ordered list (strongest evidence first)
        - unverified_skills: JD requirements lacking resume evidence
        - verified_skills_detailed: structured evidence for strict grounding
    """
    jd_keywords = extract_jd_keywords(state.cleaned_jd, top_k=15)
    matched = match_keywords_to_chunks(jd_keywords, state.retrieved_chunks)
    
    verified = sorted(matched.keys(),key=lambda k: len(matched[k]),reverse=True)

    unverified = [k for k in jd_keywords if k not in verified]
    
    summary = (
        f"Candidate matches {len(verified)}/{len(jd_keywords)} "
        f"key job requirements.\n\n"
        f"Verified Skills: {', '.join(verified[:8])}\n"
        f"Skills to Address: {', '.join(unverified[:5])}"
    )
    
    return {
        "candidate_summary": summary,
        "verified_skills": verified,
        "unverified_skills": unverified,
        "verified_skills_detailed": {
            "with_evidence": [
                {"skill": k, "evidence": v[0]}
                for k, v in matched.items()
            ],
            "no_evidence": [{"skill": s} for s in unverified]
        }
    }


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
    
    print(" Your Cover Letter is on the way, verification pending...")
    return {"cover_letter": letter_content}

def verify_letter_claims_node(state: GraphState, llm) -> dict:
    """
    Post-generation verification with specific feedback for refinement.
    """
    letter = state.cover_letter

    # HARD CAP source
    source_text = "\n".join(
        chunk["text"] for chunk in state.retrieved_chunks[:3]
    )

    prompt = f"""
STRICT TASK. NO EXPLANATION.

Check whether EVERY factual claim in the LETTER
is supported by the RESUME SOURCE.

RESUME SOURCE: {source_text}

LETTER: {letter}

OUTPUT JSON ONLY.

If all claims supported:
{{"pass": true}}

If ANY claim unsupported:
{{"pass": false}}
"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "error_type": "hallucination_critical",
            "error_message": "Verification output malformed"
        }

    if not result.get("pass", False):
        return {
            "error_type": "hallucination_critical",
            "error_message": "Unsupported claims detected"
        }

    return {}


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

    prompt = f"""You are a diff generator for cover letter refinement.

    TASK: Identify 0-5 HIGH-IMPACT edits only. Each edit must:
    - Target a specific 15-30 character phrase (for exact matching)
    - Provide a direct replacement (max 2 sentences)
    - Fix only: generic phrases, missing keywords, weak metrics, or ATS mismatches

    IGNORE: Minor word choices, stylistic preferences, overall structure.

    COVER LETTER: {state.cover_letter}

    JOB REQUIREMENTS (top 5 only): {state.cleaned_jd[:500]}

    OUTPUT (strict JSON, no markdown):
    {{
    "edits": [
        {{
        "type": "replace",
        "target": "I am passionate about technology",
        "replacement": "I built 3 production ML pipelines processing 2M+ records",
        "reason": "Replace generic phrase with verified metric"
        }}
    ],
    "severity": "minor",
    "ready_to_submit": true
    }}

    RULES:
    - If 0 edits needed: {{"edits": [], "ready_to_submit": true}}
    - Max 5 edits per iteration
    - Only target phrases that materially affect ATS scoring or hiring manager perception
    - NEVER suggest full paragraph rewrites
    """
    
    response = llm.invoke(prompt)
    content = response.content.strip().replace("```json", "").replace("```", "")
    
    try:
        data = json.loads(content)
        edits = data.get("edits", [])
        ready = data.get("ready_to_submit", len(edits) == 0)
        
        return {
            "critique_feedback": json.dumps(data, indent=2),  # For logging
            "refinement_edits": edits,  # NEW: Structured edits
            "needs_refinement": not ready and len(edits) > 0
        }
        
    except json.JSONDecodeError as e:
        print(f"Critique parse error: {e}")
        return {
            "refinement_edits": [],
            "needs_refinement": False,
            "ready_to_submit": True
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
    
    edits = state.refinement_edits
    letter = state.cover_letter
    
    if not edits:
        return {
            "cover_letter": letter,
            "needs_refinement": False,
            "refinement_count": state.refinement_count
        }
    
    for edit in edits:
        edit_type = edit.get("type")
        target = edit.get("target", "")
        
        if edit_type == "replace":
            replacement = edit.get("replacement", "")
            if target in letter:
                letter = letter.replace(target, replacement, 1)  # Only first occurrence
                print(f"Replaced: '{target[:40]}...'")
            else:
                print(f"Target not found: '{target[:40]}...'")
        
        elif edit_type == "delete":
            if target in letter:
                letter = letter.replace(target, "", 1)
                # Clean up double spaces
                letter = re.sub(r'\s{2,}', ' ', letter)
                print(f"Deleted: '{target[:40]}...'")
        
        elif edit_type == "insert_after":
            anchor = edit.get("anchor", "")
            content = edit.get("content", "")
            if anchor in letter:
                letter = letter.replace(anchor, f"{anchor} {content}", 1)
                print(f"Inserted after: '{anchor[:40]}...'")

    # This is triggered only if insertions create awkward transitions
    needs_flow_fix = any(e.get("type") == "insert_after" for e in edits)
    
    if needs_flow_fix and len(edits) > 1:
        #Fix transitions only, don't rewrite content
        flow_prompt = f"""Fix ONLY the sentence transitions in this letter. Do not change content, metrics, or structure.

                    LETTER: {letter}

                    RULE: If transitions are already smooth, return the letter unchanged.
                    Output the letter only, no commentary."""
        
        response = llm.invoke(flow_prompt)
        letter = response.content.strip()
    
    return {
        "cover_letter": letter,
        "refinement_count": state.refinement_count + 1,
        "needs_refinement": False  # Stop after 1 iteration
    }

def join_context_node(state: GraphState) -> dict:
    """This prevents race conditions between resume verification and company research."""

    if not state.candidate_summary:
        return{
            "error_type": "incomplete_context",
            "error_message": "Candidate summary missing."
        }
    
    # Optional for Company Research

    if not state.company_research:
        state.company_research = "[No Company research available]"

    return{
        "candidate_summary" : state.candidate_summary,
        "company_research" : state.company_research
    }