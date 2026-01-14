from state import GraphState
from functools import partial
from langgraph.graph import StateGraph, END, START
from node import ingest_jd_node, grade_retrieval_node, generate_summary_node, write_cover_letter_node, critique_letter_node, refine_letter_node, fallback_handler_node, rewrite_query_node
from database import retrieve_resumes_node
from IPython.display import display, Image

def should_rewrite_query(state: GraphState) -> str:
    """
    If the retrieval quality is poor, route to rewrite_query, else continue.
    """
    MAX_REWRITES = 2
    # If quality is acceptable, proceed
    if not state.needs_rewrite:
        return "generate_summary"
    
    # If exhausted rewrites, trigger fallback
    if state.rewrite_count >= MAX_REWRITES:
        return "fallback_handler"
    
    # Otherwise, retry retrieval with adjusted query
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
    workflow.add_node("fallback_handler", partial(fallback_handler_node, llm=critic_llm))
    workflow.add_node("rewrite_query", partial(rewrite_query_node, llm=generator_llm))
    workflow.add_node("retrieve_resumes", partial(retrieve_resumes_node, llm=critic_llm))
    workflow.add_node("grade_retrieval", partial(grade_retrieval_node, llm=critic_llm))
    workflow.add_node("generate_summary", partial(generate_summary_node, llm=generator_llm))
    workflow.add_node("write_cover_letter", partial(write_cover_letter_node, llm=generator_llm))
    workflow.add_node("critique_letter", partial(critique_letter_node, llm=critic_llm))
    workflow.add_node("refine_letter", partial(refine_letter_node, llm=generator_llm))

    # Edges
    workflow.add_edge(START, "ingest_jd")
    workflow.add_conditional_edges(
        "ingest_jd",
        should_proceed_with_retrieval,
        {
            "retrieve_resumes": "retrieve_resumes",
            "fallback_handler": "fallback_handler"
        }
    )
    workflow.add_edge("retrieve_resumes", "grade_retrieval")
    workflow.add_conditional_edges(
        "grade_retrieval",
        should_rewrite_query,
        {
            "rewrite_query": "rewrite_query",  
            "generate_summary": "generate_summary",
            "fallback_handler": "fallback_handler"
        }
    )
    workflow.add_edge("generate_summary", "write_cover_letter")
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
