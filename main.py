# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from graph import build_graph

load_dotenv()

# generator_llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0.0, 
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     api_key = os.getenv("GEMINI_API_KEY"),
#     streaming=True,
#     verbose=True
# )

# critic_llm = ChatOpenAI(
#     model="gpt-4o-mini",     
#     temperature=0.0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     api_key=os.getenv("OPENAI_API_KEY"),
#     streaming=True,
#     verbose=True
# )

generator_llm = ChatOllama(
    model="mistral:7b-instruct",
    temperature=0.7, # To be creative
    num_ctx=8192,
    num_gpu=35,
)

# Critic: Optimized for structural analysis and fault-finding
critic_llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.0,      
    num_ctx=4096, # For latnecy Tradeoff
    num_gpu=35,
)

def main():
    """
    Orchestrator for the Job Application Assistant.
    
    This entry point manages:
    1. Robust Multi-line Input: Uses a 'DONE' sentinel to allow users to 
       paste complex JDs with empty lines without prematurely triggering the graph.
    2. State Initialization: Prepares a clean dictionary-based initial state 
       to comply with LangGraph's mapping requirements.
    3. Fallback Handling: Gracefully presents fallback suggestions if the 
       JD is invalid or no resume matches are found.
    4. Output Formatting: Displays the final cover letter, relevance scores, 
       and iteration metrics.
    """

    print("--- Job Application Assistant ---")
    print("Paste your Job Description below.")
    print("When finished, type 'DONE' on a new line and press Enter.")
    print("---------------------------------")

    # Multi-line input collection
    lines = []
    while True:
        try:
            line = input()
            # If the user types 'DONE', stop collecting
            if line.strip().upper() == "DONE":
                break
            lines.append(line)
        except EOFError:
            break
    
    job_description = "\n".join(lines)

    if not job_description.strip():
        print("You'll need to parse job description first")
        return
    
    company_input = input("Please enter the company name: ").strip()
    print(" Researching company information...")

    # Build Graph
    app = build_graph(generator_llm, critic_llm)
    
    # Initialize state
    initial_state = {
        "job_description" : job_description,
        "is_valid_jd": False,
        "cleaned_jd" : "",
        "retrieved_chunks" : [],
        "candidate_summary" : "",
        "resume_summary" : "",
        "cover_letter" : "",
        "critique_feedback" : "",
        "vector_relevance_score": None,
        "llm_relevance_score" : 0,
        "needs_rewrite" : False,
        "grading_feedback" : "",
        "needs_refinement" : False,
        "refinement_count" : 0,
        "error_type": "",
        "error_message": "",
        "is_fallback": False,
        "final_response": "",
        "rewrite_count": 0,
        "company_name": company_input,
        "company_research": "",
        "needs_company_confirmation": False,
        "company_research_success": False
    }   

    try:

        print("\n Analyzing job description...")
        result = app.invoke(
            initial_state, config = {
                "run_name": "job_application_run",
                "tags": ["langgraph", "async", "cover-letter"],
                "metadata": {
                    "company": company_input,
                    "has_research": bool(company_input),
                }
            }
        )

        # Check if fallback was triggered
        if result.get("is_fallback", False):
            print("FALLBACK RESPONSE: ")
            print(result.get("final_response", "An error occurred."))
            return
        
        print("Your Cover Letter is Ready")
        print("--------------------------------- \n")
        print(result["cover_letter"])
        print("\n ---------------------------------")
        print(f" Retrieval Score: {result['vector_relevance_score']}/100")
        print(f" Refinement Iterations: {result['refinement_count']}")


    except Exception as e:
        print(f" Error during processing: {str(e)}")

if __name__ == "__main__":
    main()