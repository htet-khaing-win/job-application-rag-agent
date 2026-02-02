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
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def sanitize_json_response(content: str) -> str:
    """Remove control characters and formatting that break JSON parsing."""
    content = content.strip().replace("```json", "").replace("```", "")
    # Remove ASCII control characters (0-31, 127-159)
    content = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', content)
    # Normalize multiple spaces
    content = re.sub(r' {2,}', ' ', content)
    return content.strip()

# INGEST JOB DESCRIPTION (Entry Node)
def ingest_jd_node(state: GraphState, llm) -> dict:
    """
    Single-pass JD validation and extraction.
    
    Combines validation + cleaning into one LLM call for efficiency.
    """
    
    combined_prompt = f"""You are processing a job description input.

    TASK 1: Validate if this is a legitimate job description and extract key information if valid

    INPUT: {state.job_description}

    VALIDATION CRITERIA:
    - Must be a Job Description

    EXTRACTION (if valid):
    - Core Requirements: must-have skills/qualifications

    OUTPUT (strict JSON, no markdown):
    {{
      "valid": true,
      "cleaned_jd": "extracted requirements and responsibilities"
    }}

    If invalid:
    {{
      "valid": false,
      "cleaned_jd": ""
    }}
    """
    
    response = llm.invoke(combined_prompt)
    content = sanitize_json_response(response.content)
    
    try:
        data = json.loads(content)
        
        if not data.get("valid", False):
            return {
                "is_valid_jd": False,
                "error_type": "invalid_input",
                "error_message": "This doesn't appear to be a job description. Please paste a complete job posting."
            }
        
        cleaned = data.get("cleaned_jd", "")

        if isinstance(cleaned, dict):
            # Flatten structured JD into deterministic text
            cleaned = "\n".join(
                f"{k}: {', '.join(v) if isinstance(v, list) else v}"
                for k, v in cleaned.items()
            )

        return {
            "cleaned_jd": cleaned,
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


def fallback_handler_node(state: GraphState) -> dict:
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
    """Extract keywords using TF-IDF."""
    
    stopwords = ENGLISH_STOP_WORDS.union({
        'experience', 'knowledge', 'ability', 'skills', 'skill', 'required',
        'requirements', 'requirement', 'responsibilities', 'responsibility',
        'years', 'year', 'work', 'working', 'role', 'candidate', 'team',
        'strong', 'good', 'excellent', 'proven', 'demonstrated',
        'develop', 'build', 'create', 'manage', 'lead', 'support',
        'including', 'highly', 'successfully', 'effectively'
    })
    
    vectorizer = TfidfVectorizer(
        max_features=top_k,
        stop_words=list(stopwords),
        ngram_range=(1, 2)
    )
    
    vectorizer.fit_transform([cleaned_jd])
    return list(vectorizer.get_feature_names_out())

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
def generate_summary_node(state: GraphState) -> dict:
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
    The Copywriter - Generates evidence-grounded cover letter.
    """
    
    if state.vector_relevance_score is None or state.vector_relevance_score <= 0:
        raise RuntimeError(
            "Invariant violated: cover letter generation attempted without valid retrieval score."
        )

    # Build concise skill evidence map
    skills_with_evidence = state.verified_skills_detailed.get("with_evidence", [])
    unverified_skills = state.unverified_skills[:3]  # Cap at 3 most important
    
    # Format as compact structure
    verified_block = "\n".join([
        f"• {s['skill']}: {s['evidence'][:120]}"
        for s in skills_with_evidence[:8]  # Top 8 only
    ])
    
    unverified_block = ", ".join(unverified_skills) if unverified_skills else "None"
    
    # Identify which unverified skills are actually in JD
    jd_lower = state.cleaned_jd.lower()
    critical_gaps = [s for s in unverified_skills if s.lower() in jd_lower]

    allowed_skills = [s["skill"] for s in skills_with_evidence[:8]] + unverified_skills
    allowed_skills_str = ", ".join(allowed_skills)

    prompt = f"""Write a factual cover letter body.

    VERIFIED SKILLS: {verified_block}

    UNVERIFIED SKILLS: {unverified_block}
    Critical gaps: {', '.join(critical_gaps) if critical_gaps else 'None'}

    COMPANY: {state.company_research}

    Do not mention skills outside this list: {allowed_skills_str}

    STRUCTURE:
    P1 (40-50w): Company hook + 1 metric
    P2 (120-140w): 3 verified skills with evidence + metrics
    P3 (60-80w): Address gaps via transferable skills OR expand verified achievements
    P4 (40-50w): Call-to-action + company reference

    RULES:
    - Use only skills from VERIFIED list with exact evidence
    - Unverified skills: "prepared to learn via [verified skill]"
    - Do not introduce any skills/tools not in ALLOWED SKILLS LIST
    - Company name: {state.company_name}
        Output cover letter body only."""
    
    response = llm.invoke(prompt)
    letter_content = response.content.strip()
    
    return {"cover_letter": letter_content}


# CRITIQUE LETTER (The Hiring Manager)
def critique_letter_node(state: GraphState, llm) -> dict:
    """The Hiring Manager - Reviews letter for quality and authenticity."""
    
    prompt = f"""You are a Hiring Manager who is expert in criticizing cover letters.

    TASK: Identify 0-5 HIGH-IMPACT edits only if necessary. Each edit must:
    - Target a specific 15-30 character phrase (for exact matching)
    - Provide a direct replacement (max 1 sentence)
    - Fix only: generic phrases, missing keywords

    COVER LETTER: {state.cover_letter}

    JOB REQUIREMENTS (top 5 only): {state.cleaned_jd[:500]}

    OUTPUT (strict JSON, no markdown):
    {{
      "edits": [
        {{
          "type": "replace",
          "target": "exact phrase",
          "replacement": "specific text",
          "reason": "brief"
        }}
      ],
      "severity": "minor",
      "ready_to_submit": true
    }}

    RULES:
    - If no edits needed: {{"edits": [], "ready_to_submit": true}}
    """
    
    response = llm.invoke(prompt)
    content = sanitize_json_response(response.content)
    
    try:
        data = json.loads(content)
        edits = data.get("edits", [])
        ready = data.get("ready_to_submit", len(edits) == 0)
        
        # Validate that suggested edits actually exist in the letter
        valid_edits = []
        for edit in edits:
            target = edit.get("target", "")
            if target in state.cover_letter:
                valid_edits.append(edit)
            else:
                print(f" Critique suggested invalid target: '{target[:40]}...'")

        return {
            "critique_feedback": json.dumps(data, indent=2),  # For logging
            "refinement_edits": valid_edits, 
            "needs_refinement": not ready and len(valid_edits) > 0  # Clear logic
        }
        
    except json.JSONDecodeError as e:
        print(f" Critique parse error: {e}")
        return {
            "refinement_edits": [],
            "needs_refinement": False,
            "critique_feedback": f"Parse error: {str(e)}"
        }


# REFINE LETTER (The Editor)
def refine_letter_node(state: GraphState, llm) -> dict:
    """
    The Editor - Rewrites letter based on critique feedback.
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
                letter = letter.replace(target, replacement, 1)
                print(f" Replaced: '{target[:40]}...'")
            else:
                print(f"Target not found: '{target[:40]}...'")
        
        elif edit_type == "delete":
            if target in letter:
                letter = letter.replace(target, "", 1)
                letter = re.sub(r'\s{2,}', ' ', letter)
                print(f"Deleted: '{target[:40]}...'")
        
        elif edit_type == "insert_after":
            anchor = edit.get("anchor", "")
            content = edit.get("content", "")
            if anchor in letter:
                letter = letter.replace(anchor, f"{anchor} {content}", 1)
                print(f"Inserted after: '{anchor[:40]}...'")

    needs_flow_fix = any(e.get("type") == "insert_after" for e in edits)
    
    if needs_flow_fix and len(edits) > 1:
        flow_prompt = f"""Fix ONLY the sentence transitions in this letter. Do not change content, metrics, or structure.

        LETTER: {letter}

        RULE: If transitions are already smooth, return the letter unchanged.
        Output the letter only, no commentary."""
        
        response = llm.invoke(flow_prompt)
        letter = response.content.strip()
    
    return {
        "cover_letter": letter,
        "refinement_count": state.refinement_count + 1,
        "needs_refinement": False
    }

def join_context_node(state: GraphState) -> dict:
    """This prevents race conditions between resume verification and company research."""

    if not state.candidate_summary:
        return {
            "error_type": "incomplete_context",
            "error_message": "Candidate summary missing."
        }
    
    updates = {}
    # Optional for Company Research

    if not state.company_research:
        state.company_research = "[No Company research available]"

    return updates