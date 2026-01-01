from state import ChainState
from functools import partial
from langgraph.graph import StateGraph, END, START
from node import ingest_jd_node, grade_retrieval_node, generate_summary_node, write_cover_letter_node, critique_letter_node, refine_letter_node
from database import retrieve_resumes_node
from IPython.display import display, Image

def should_rewrite_query(state: ChainState) -> str:
    """
    If the retrieval quality is poor, route to rewrite_query, else continue.
    """
    if state.needs_rewrite:
        return "rewrite_query"
    
    return "generate_summary"

def should_refine_letter(state: ChainState) -> str:
    """
    If critique failed and under iteration limit, route to refinement.
    """
    if not state.needs_refinement:
        return END
    
    if state.refinement_count >= 4:
        return END
    return "refine_letter"


def build_graph(llm):
    """Constructs the LangGraph workflow for job application assistance."""

    workflow = StateGraph(ChainState)

    # Nodes
    workflow.add_node("ingest_jd", partial(ingest_jd_node, llm=llm))
    workflow.add_node("retrieve_resumes", partial(retrieve_resumes_node, llm=llm))
    workflow.add_node("grade_retrieval", partial(grade_retrieval_node, llm=llm))
    workflow.add_node("generate_summary", partial(generate_summary_node, llm=llm))
    workflow.add_node("write_cover_letter", partial(write_cover_letter_node, llm=llm))
    workflow.add_node("critique_letter", partial(critique_letter_node, llm=llm))
    workflow.add_node("refine_letter", partial(refine_letter_node, llm=llm))
    # workflow.add_node("should_rewrite", partial(should_rewrite_query, llm=llm))
    # workflow.add_node("should_refine", partial(should_refine_letter, llm=llm))

    # Edges
    workflow.add_edge(START, "ingest_jd")
    workflow.add_edge("ingest_jd", "retrieve_resumes")
    workflow.add_edge("retrieve_resumes", "grade_retrieval")
    workflow.add_conditional_edges(
        "grade_retrieval",
        should_rewrite_query,
        {
            "rewrite_query": "retrieve_resumes",  
            "generate_summary": "generate_summary"
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
    
    return workflow.compile()
