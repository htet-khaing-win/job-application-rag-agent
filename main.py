from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from graph import build_graph

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0, 
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key = os.getenv("API_KEY")
)

def main():
    """
    Entry point for the Job Application Assistant Agent.
    
    Workflow:
    1. Takes a job description as input
    2. Retrieves relevant resume chunks from Pinecone
    3. Generates tailored cover letter with iterative refinement
    4. Outputs polished, ATS-optimized application materials
    """

    app = build_graph(llm)
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    job_description = "\n".join(lines)

    if not job_description.strip():
        print("You'll need to parse job description first")
        return
    
    # Initialize state
    initial_state = {
        "job_description" : job_description,
        "cleaned_jd" : "",
        "retrieved_chunks" : [],
        "candidate_summary" : "",
        "resume_summary" : "",
        "relevance_score" : 0,
        "needs_rewrite" : False,
        "grading_feedback" : "",
        "cover_letter" : "",
        "critique_feedback" : "",
        "needs_refinement" : False,
        "refinement_count" : 0
    }   

    try:
        result = app.invoke(initial_state)
        print("Your Cover Letter is Ready")
        print(result["cover_letter"])
        print(f" Retrieval Score: {result['relevance_score']}/100")
        print(f" Refinement Iterations: {result['refinement_count']}")


    except Exception as e:
        print(f" Error during processing: {str(e)}")
        print("\n Debug Info:")
        print(f"- Check your API keys in .env file")
        print(f"- Verify Pinecone index 'resume-index' exists")
        print(f"- Ensure resume data is uploaded to Pinecone")

if __name__ == "__main__":
    main()