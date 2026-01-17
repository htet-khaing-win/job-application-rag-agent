from state import GraphState
from functools import partial
from langgraph.graph import StateGraph, END, START
from node import ingest_jd_node, grade_retrieval_node, generate_summary_node, write_cover_letter_node, critique_letter_node, refine_letter_node, fallback_handler_node, rewrite_query_node, verify_claims_node, research_company_node
from database import retrieve_resumes_node
from IPython.display import display, Image

def should_rewrite_query(state: GraphState) -> str:
    """
    If the retrieval quality is poor, route to rewrite_query, else continue.
    """
    MAX_REWRITES = 2
    ACCEPTABLE_THRESHOLD = 60
    current_score = state.llm_relevance_score
    current_rewrites = state.rewrite_count

    # If quality is acceptable, proceed
    if current_score  >= ACCEPTABLE_THRESHOLD:
        return "generate_summary"
    
    # If exhausted rewrites, trigger fallback
    if current_rewrites >= MAX_REWRITES:
        return "fallback_handler"
    
    # Otherwise, retry retrieval with adjusted query
    print(f"Score below threshold. Rewriting query (attempt {current_rewrites + 1}/{MAX_REWRITES})")
    return "rewrite_query"

def should_refine_letter(state: GraphState) -> str:
    """
    If critique failed and under iteration limit, route to refinement.
    """
    if not state.needs_refinement:
        return END
    
    if state.refinement_count >= 3:
        return END
    return "refine_letter"

def should_proceed_with_retrieval(state: GraphState) -> str:
    """
    Check if input was valid before retrieving.
    """
    if not state.is_valid_jd:
        return "fallback_handler"
    return "retrieve_resumes"

def needs_company_research(state: GraphState) -> str:
    """
    NEW: Routes to company research if name is confirmed.
    """
    if state.company_name and state.company_name != "UNKNOWN":
        print(f" Researching: {state.company_name}")
        return "research_company"
    
    print("  No company name provided. Skipping research.")
    return "retrieve_resumes"

def verification_guard(state: GraphState) -> str:
    if state.error_type in ["verification_failed", "hallucination_critical"]:
        return "fallback_handler"
    return "write_cover_letter"


def build_graph(generator_llm, critic_llm):
    """
    Constructs the LangGraph workflow for job application assistance.
    
    Design principle:
    - Generator (Mistral): Optimized for constrained, fluent generation
    - Critic (Qwen): Optimized for analytical fault-finding
    """

    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node("ingest_jd", partial(ingest_jd_node, llm=critic_llm))  
    workflow.add_node("research_company", partial(research_company_node, llm=critic_llm))  
    workflow.add_node("retrieve_resumes", partial(retrieve_resumes_node, llm=critic_llm))
    workflow.add_node("grade_retrieval", partial(grade_retrieval_node, llm=critic_llm))
    workflow.add_node("rewrite_query", partial(rewrite_query_node, llm=generator_llm))
    workflow.add_node("verify_claims", partial(verify_claims_node, llm=critic_llm))  
    workflow.add_node("generate_summary", partial(generate_summary_node, llm=generator_llm))
    workflow.add_node("write_cover_letter", partial(write_cover_letter_node, llm=generator_llm))
    workflow.add_node("critique_letter", partial(critique_letter_node, llm=critic_llm))
    workflow.add_node("refine_letter", partial(refine_letter_node, llm=generator_llm))
    workflow.add_node("fallback_handler", partial(fallback_handler_node, llm=critic_llm))

    # Edges
    workflow.add_edge(START, "ingest_jd")
    workflow.add_edge("ingest_jd", "research_company")

    workflow.add_edge("research_company", "retrieve_resumes")
    workflow.add_edge("retrieve_resumes", "grade_retrieval")

    workflow.add_conditional_edges(
        "grade_retrieval",
        should_rewrite_query,
        {
            "generate_summary": "generate_summary",
            "rewrite_query": "rewrite_query",  
            "fallback_handler": "fallback_handler"
        }
    )
    workflow.add_edge("rewrite_query", "retrieve_resumes")
    workflow.add_edge("generate_summary", "verify_claims")

    workflow.add_conditional_edges(
    "verify_claims",
    verification_guard,
    {
        "write_cover_letter": "write_cover_letter",
        "fallback_handler": "fallback_handler"
    }
)

    workflow.add_edge("write_cover_letter", "critique_letter")
    workflow.add_edge("refine_letter", "critique_letter")

    workflow.add_conditional_edges(
        "critique_letter",
        should_refine_letter,
        {
            "refine_letter": "refine_letter",
            END: END
        }
    )
    workflow.add_edge("fallback_handler", END)
    return workflow.compile()
