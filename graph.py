from state import GraphState
from functools import partial
from langgraph.graph import StateGraph, END, START
from node import ingest_jd_node, generate_summary_node, write_cover_letter_node, critique_letter_node, refine_letter_node, fallback_handler_node, research_company_node, join_context_node
from database import retrieve_resumes_node
import asyncio

async def should_refine_letter(state: GraphState) -> str:
    """
    Route to refinement if critique identified issues and we haven't exceeded iteration limit.
    """
    # Check iteration limit first
    if state.refinement_count >= 1:
        return END
    
    # If no edits suggested, nothing to refine
    if not state.refinement_edits:
        return END
    
    # If needs_refinement flag is explicitly False, critic approved
    if not state.needs_refinement:
        return END
    
    return "refine_letter"


async def route_after_retrieval(state: GraphState) -> str:
    # Yield control to event loop - ensures state updates are committed
    await asyncio.sleep(0)
    
    # Defensive check for race condition
    if state.vector_relevance_score is None:
        print("  Routing: No vector score available")
        return "fallback_handler"
    
    score = state.vector_relevance_score
    print(f" Routing: Vector score = {score}")
    
    if score >= 5.0:
        return "generate_summary"
    
    print(f"  Score {score} below threshold (5.0)")
    return "fallback_handler"

def route_after_refine(state: GraphState) -> str:
    if state.refinement_count >= 1:
        return END
    return "critique_letter"

def build_graph(generator_llm, critic_llm):
    """
    Constructs the LangGraph workflow for job application assistance.
    """

    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node("ingest_jd", partial(ingest_jd_node, llm=critic_llm))  
    workflow.add_node("research_company", research_company_node)  
    workflow.add_node("retrieve_resumes", retrieve_resumes_node) 
    workflow.add_node("generate_summary", generate_summary_node)
    workflow.add_node("write_cover_letter", partial(write_cover_letter_node, llm=generator_llm))
    workflow.add_node("critique_letter", partial(critique_letter_node, llm=critic_llm))
    workflow.add_node("refine_letter", partial(refine_letter_node, llm=generator_llm))
    workflow.add_node("fallback_handler", fallback_handler_node)
    workflow.add_node("join_context", join_context_node)


    # Edges
    workflow.add_edge(START, "ingest_jd")
    workflow.add_edge("ingest_jd", "research_company")
    workflow.add_conditional_edges(
        "ingest_jd",
        lambda state: "retrieve_resumes" if state.is_valid_jd else "fallback_handler",
        {
            "retrieve_resumes": "retrieve_resumes",
            "fallback_handler": "fallback_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "retrieve_resumes",
        route_after_retrieval, 
        {
            "generate_summary": "generate_summary",
            "fallback_handler": "fallback_handler"
        }
    )

    workflow.add_edge("generate_summary", "join_context")
    workflow.add_edge("research_company", "join_context")

    workflow.add_edge("join_context", "write_cover_letter")
    workflow.add_edge("write_cover_letter", "critique_letter")

    workflow.add_conditional_edges(
        "critique_letter",
        should_refine_letter,
        {
            "refine_letter": "refine_letter",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "refine_letter",
        route_after_refine,
        {
            "critique_letter": "critique_letter",
            END: END
        }
    )

    workflow.add_edge("fallback_handler", END)
    return workflow.compile()
