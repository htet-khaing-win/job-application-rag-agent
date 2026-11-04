import asyncio
from collections.abc import Iterable

from langgraph.graph import END, START, StateGraph

from state import GraphState
from subagents import (
    AnalystAgent,
    ContextJoinAgent,
    CriticAgent,
    EditorAgent,
    FactCheckerAgent,
    FallbackAgent,
    JDAnalystAgent,
    ResearchAgent,
    RetrieverAgent,
    Supervisor,
    WriterAgent,
)

VALID_ABLATIONS = frozenset({"fact_checker", "critique_refine", "research", "failover"})


def validate_ablate(ablate: Iterable[str]) -> frozenset:
    """Normalise and validate ablation names ("failover" is applied by the pipeline, not the graph)."""
    names = frozenset(ablate or ())
    unknown = names - VALID_ABLATIONS
    if unknown:
        raise ValueError(f"Unknown ablation(s): {sorted(unknown)}; valid: {sorted(VALID_ABLATIONS)}")
    return names

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

def route_after_ingest(state: GraphState):
    """Fan out to retrieval + research in parallel when the JD is valid."""
    if state.is_valid_jd:
        return ["retrieve_resumes", "research_company"]
    return "fallback_handler"


def route_after_join(state: GraphState) -> str:
    return "fallback_handler" if state.error_type in ("incomplete_context", "generation_failed") else "write_cover_letter"


def route_after_write(state: GraphState) -> str:
    return "fallback_handler" if state.error_type == "generation_failed" or not state.cover_letter.strip() else "verify_letter"


def build_graph(generator_llm=None, critic_llm=None, ablate: Iterable[str] = frozenset()):
    """
    Constructs the self-healing LangGraph workflow.

    Every node is a specialised sub-agent guarded by the Supervisor (retry -> degrade).
    By default each agent resolves its own model + API key from the active runtime profile
    (see runtime.py). Passing generator_llm / critic_llm pins models explicitly (tests, A/B).
    """
    ablate = validate_ablate(ablate)
    pinned = generator_llm is not None or critic_llm is not None
    gen, crit = generator_llm, critic_llm
    sup = Supervisor()

    agents = {
        "ingest_jd": JDAnalystAgent(crit if pinned else None),
        "research_company": ResearchAgent(disabled="research" in ablate),
        "retrieve_resumes": RetrieverAgent(),
        "generate_summary": AnalystAgent(),
        "join_context": ContextJoinAgent(),
        "write_cover_letter": WriterAgent(gen if pinned else None),
        "verify_letter": FactCheckerAgent(crit if pinned else None, passthrough="fact_checker" in ablate),
        "critique_letter": CriticAgent(crit if pinned else None),
        "refine_letter": EditorAgent(gen if pinned else None),
        "fallback_handler": FallbackAgent(),
    }

    workflow = StateGraph(GraphState)
    if "critique_refine" in ablate:  # ship the verified draft without critique / refinement
        agents.pop("critique_letter")
        agents.pop("refine_letter")
    for name, agent in agents.items():
        workflow.add_node(name, sup.guard(agent))

    workflow.add_edge(START, "ingest_jd")
    workflow.add_conditional_edges(
        "ingest_jd", route_after_ingest,
        ["retrieve_resumes", "research_company", "fallback_handler"],
    )
    workflow.add_conditional_edges(
        "retrieve_resumes", route_after_retrieval,
        {"generate_summary": "generate_summary", "fallback_handler": "fallback_handler"},
    )

    # join waits for BOTH branches (list form), unlike two separate edges which fire on either
    workflow.add_edge(["generate_summary", "research_company"], "join_context")
    workflow.add_conditional_edges(
        "join_context", route_after_join,
        {"write_cover_letter": "write_cover_letter", "fallback_handler": "fallback_handler"},
    )
    workflow.add_conditional_edges(
        "write_cover_letter", route_after_write,
        {"verify_letter": "verify_letter", "fallback_handler": "fallback_handler"},
    )
    if "critique_refine" in ablate:
        workflow.add_edge("verify_letter", END)
    else:
        workflow.add_edge("verify_letter", "critique_letter")
        workflow.add_conditional_edges(
            "critique_letter", should_refine_letter,
            {"refine_letter": "refine_letter", END: END},
        )
        workflow.add_conditional_edges(
            "refine_letter", route_after_refine,
            {"critique_letter": "critique_letter", END: END},
        )

    workflow.add_edge("fallback_handler", END)
    return workflow.compile()
