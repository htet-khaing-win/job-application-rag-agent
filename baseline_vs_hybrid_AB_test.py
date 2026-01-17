import sys
import os
import json
import time
import numpy as np
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from typing import List, Dict
from dataclasses import dataclass, asdict
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()


# INITIALIZATION & BM25 FITTING
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embeddings = OllamaEmbeddings(model="nomic-embed-text")

grading_llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.0,
    num_ctx=4096,
    num_gpu=35,
)

index = pc.Index("resume-index")

# BM25 LOADING LOGIC 

BM25_PATH = "utilities/fitted_bm25.json"

if os.path.exists(BM25_PATH):
    print(f" Loading custom fitted BM25 from {BM25_PATH}")
    bm25 = BM25Encoder()
    bm25.load(BM25_PATH)
else:
    print(" WARNING: No fitted BM25 found. Using default (cold) encoder.")
    print(" Quality improvement will be minimal until you fit the encoder on your resume corpus.")
    bm25 = BM25Encoder.default()

@dataclass
class TestResult:
    """Container for single test result"""
    method: str
    test_name: str
    namespace: str
    relevance_score: float
    avg_chunk_score: float
    top_chunk_score: float
    chunks_above_threshold: int
    retrieval_time_ms: float

# RETRIEVAL FUNCTIONS

def baseline_retrieve(jd: str, namespace: str, top_k: int = 5) -> tuple:
    """Pure semantic search (Baseline)"""
    start = time.time()
    
    query_vector = embeddings.embed_query(jd)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    
    latency = (time.time() - start) * 1000
    chunks = [
        {
            "text": m["metadata"]["text"],
            "score": round(m["score"], 4),
            "chunk_id": m["id"]
        }
        for m in results['matches']
    ]
    
    return chunks, latency

def hybrid_retrieve(jd: str, namespace: str, top_k: int = 5, alpha: float = 0.4) -> tuple:
    """Fitted Hybrid search (Dense + BM25)"""
    start = time.time()
    
    dense_query = embeddings.embed_query(jd)
    # Uses the fitted encoder to generate importance-weighted sparse vectors
    sparse_query = bm25.encode_queries(jd) 
    
    results = index.query(
        vector=dense_query,
        sparse_vector=sparse_query,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
        alpha=alpha  
    )
    
    latency = (time.time() - start) * 1000
    chunks = [
        {
            "text": m["metadata"]["text"],
            "score": round(m["score"], 4),
            "chunk_id": m["id"]
        }
        for m in results['matches']
    ]
    
    return chunks, latency

# EVALUATION & ANALYSIS

def normalize_scores(chunks: List[Dict]) -> List[Dict]:
    if not chunks: return chunks
    scores = [c["score"] for c in chunks]
    min_s, max_s = min(scores), max(scores)
    if min_s == max_s: return chunks
    for c in chunks:
        c["norm_score"] = (c["score"] - min_s) / (max_s - min_s)
    return chunks

def grade_retrieval(jd: str, chunks: List[Dict]) -> Dict:
    """LLM-based relevance grading"""
    prompt = f"""
    ROLE: Technical Recruiter evaluating resume-job match quality.
    JOB REQUIREMENTS:
    {jd}
    RETRIEVED CHUNKS:
    {json.dumps([{"text": c["text"], "score": c["score"]} for c in chunks], indent=2)}
    TASK: Rate overall relevance from 0-100 based on how well these chunks prove the candidate is a fit.
    OUTPUT (strict JSON):
    {{"score": <int>, "reasoning": "<brief explanation>"}}
    """
    
    try:
        response = grading_llm.invoke(prompt)
        content = response.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        return {"score": int(data.get("score", 0)), "reasoning": data.get("reasoning", "")}
    except Exception as e:
        return {"score": 0, "reasoning": f"Error: {str(e)}"}

def run_ab_test(test_cases: List[Dict], output_file: str = "hybrid_ab_results.json"):
    stats_info = index.describe_index_stats()
    namespaces = sorted(list(stats_info.get('namespaces', {}).keys()))
    
    if not namespaces:
        print("  No resumes found in index.")
        return
    
    results = []
    
    for test_case in test_cases:
        test_name = test_case["name"]
        jd = test_case["jd"]
        
        print(f"\n{'='*75}")
        print(f"TEST: {test_name}")
        print(f"{'='*75}")
        
        for namespace in namespaces:
            print(f"\nResume: {namespace}")
            
            # 1. Baseline Run
            print(" [1/2] Baseline retrieval...")
            chunks_b, lat_b = baseline_retrieve(jd, namespace)
            chunks_b = normalize_scores(chunks_b)
            score_b = grade_retrieval(jd, chunks_b)["score"]
            
            # Calculate additional metrics for Baseline
            norm_scores_b = [c.get("norm_score", 0) for c in chunks_b]
            avg_b = np.mean(norm_scores_b) if norm_scores_b else 0
            top_b = max(norm_scores_b) if norm_scores_b else 0
            count_b = sum(1 for s in norm_scores_b if s >= 0.7)
            
            res_b = TestResult(
                method="baseline", 
                test_name=test_name, 
                namespace=namespace, 
                relevance_score=score_b, 
                avg_chunk_score=avg_b, 
                top_chunk_score=top_b, 
                chunks_above_threshold=count_b, 
                retrieval_time_ms=lat_b
            )
            results.append(res_b)

            # 2. Hybrid Run
            print(" [2/2] Hybrid retrieval...")
            chunks_h, lat_h = hybrid_retrieve(jd, namespace)
            chunks_h = normalize_scores(chunks_h)
            score_h = grade_retrieval(jd, chunks_h)["score"]
            
            # Calculate additional metrics for Hybrid
            norm_scores_h = [c.get("norm_score", 0) for c in chunks_h]
            avg_h = np.mean(norm_scores_h) if norm_scores_h else 0
            top_h = max(norm_scores_h) if norm_scores_h else 0
            count_h = sum(1 for s in norm_scores_h if s >= 0.7)
            
            res_h = TestResult(
                method="hybrid", 
                test_name=test_name, 
                namespace=namespace, 
                relevance_score=score_h, 
                avg_chunk_score=avg_h, 
                top_chunk_score=top_h, 
                chunks_above_threshold=count_h, 
                retrieval_time_ms=lat_h
            )
            results.append(res_h)

            # LIVE TABLE PRINTING
            print(f"\n {'Metric':<30} {'Baseline':<15} {'Hybrid':<15} {'Δ':<10}")
            print(f" {'-'*70}")
            print(f" {'Relevance Score':<30} {score_b:<15} {score_h:<15} {score_h - score_b:+.1f}")
            print(f" {'Avg Chunk Score (Norm)':<30} {avg_b:<15.4f} {avg_h:<15.4f} {avg_h - avg_b:+.4f}")
            print(f" {'Top Chunk Score':<30} {top_b:<15.4f} {top_h:<15.4f} {top_h - top_b:+.4f}")
            print(f" {'High Quality Chunks':<30} {count_b:<15} {count_h:<15} {count_h - count_b:+}")
            print(f" {'Latency (ms)':<30} {lat_b:<15.1f} {lat_h:<15.1f} {lat_h - lat_b:+.1f}")

    # Save all results to JSON
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    return results

def analyze_results(results_file: str = "hybrid_ab_results.json"):
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    methods = ["baseline", "hybrid"]
    metrics = [
        ("relevance_score", "Relevance Score", "higher"),
        ("retrieval_time_ms", "Latency (ms)", "lower")
    ]
    
    print(f"\n{'='*70}\nFINAL ANALYSIS\n{'='*70}")
    
    for key, name, goal in metrics:
        b_vals = [r[key] for r in data if r["method"] == "baseline"]
        h_vals = [r[key] for r in data if r["method"] == "hybrid"]
        
        b_avg, h_avg = np.mean(b_vals), np.mean(h_vals)
        imp = ((h_avg - b_avg) / b_avg * 100) if goal == "higher" else ((b_avg - h_avg) / b_avg * 100)
        _, p_val = stats.ttest_ind(h_vals, b_vals, equal_var=False)
        
        print(f"{name}:")
        print(f"  Baseline: {b_avg:.2f} | Hybrid: {h_avg:.2f}")
        print(f"  Improvement: {imp:+.2f}% | p-value: {p_val:.4f}")
    
    print(f"{'='*70}\n")


# MAIN EXECUTION

if __name__ == "__main__":
    test_cases = [
        {
            "name": "Lead AI Engineer",
            "jd": """
            Role Overview:

            We are looking for an experienced Lead AI Engineer to lead the design and articulation of enterprise-grade AI solutions, with deep hands-on knowledge of Microsoft Azure AI and modern generative and agentic AI capabilities. This role focuses on understanding customer business needs, shaping AI-driven transformation strategies, presenting compelling solution designs, and supporting successful delivery execution.

            The ideal candidate combines strong technical expertise with consulting skills—able to guide customers, recommend prudent AI approaches, and communicate clear value propositions that align with business outcomes.

            Key Responsibilities:

            Customer Engagement & Requirements Discovery
            Lead customer workshops to understand goals, processes, challenges, and expected outcomes.
            Translate business requirements into structured AI use cases with clear feasibility, ROI, and prioritization.
            Conduct current-state assessments covering data readiness, integration needs, compliance, and security.
            AI Solution Architecture & Design
            Design end-to-end AI solutions using:
            Azure OpenAI Service, Azure AI Studio, Azure Machine Learning, Azure Cognitive Services, Azure Data Platform (ADLS, Fabric, Databricks, Synapse)
            Vector search, embeddings, and retrieval-augmented generation (RAG)
            Develop architecture diagrams, data flows, integration patterns, and system component definitions.
            Define solution options, trade-offs, and best-fit recommendations based on business value and risk.
            Ensure solutions adhere to enterprise standards for scalability, performance, privacy, and governance.
            Consulting & Thought Leadership
            Act as a trusted advisor to customers on AI adoption, roadmap planning, and responsible AI practices.
            Provide clear guidance on MLOps/LLMOps, data governance, security, and deployment strategies.
            Share knowledge on Azure AI capabilities, innovations, and Microsoft’s product roadmap.
            Proposal Development & Presales Support
            Prepare high-quality proposals, solution decks, and value proposition statements.
            Respond to RFP/RFI documents with accurate, clear, and competitive solution descriptions.
            Conduct customer presentations, demos, POCs, and technical deep dives.
            Partner with sales teams to articulate solution differentiation and business value.
            Delivery Handover & Execution Support
            Provide technical oversight during project initiation and design phases.
            Support delivery teams with clarifications, design governance, and risk mitigation.
            Participate in review sessions to ensure solution integrity throughout implementation.
            Reusable Assets & Knowledge Creation
            Create reusable frameworks, solution accelerators, demo environments, and reference architectures.
            Contribute to internal playbooks, best practices, and industry-specific AI templates.
            Stay updated with Azure AI advancements, enterprise AI trends, and generative AI patterns.
            Required Skills & Experience:

            Technical Skills

            7+ years in AI/ML, cloud architecture, or solution design; at least 3 years with Azure AI.
            Strong hands-on expertise with:
            Azure OpenAI (GPT models, embeddings, chat orchestration)
            Azure Machine Learning (pipelines, model registry, deployment)
            Cognitive Services (vision, language, speech)
            Azure Data services (Fabric, ADLS, Databricks)
            API integration, microservices, and serverless architectures
            Experience designing RAG architectures, conversational agents, document intelligence, predictive models, or enterprise automation solutions.
            Understanding of enterprise security, identity, compliance, and responsible AI guidelines.
            Consulting & Presales Skills

            Ability to translate technical concepts into business-friendly narratives.
            Strong presentation, negotiation, and client-facing communication skills.
            Experience in writing proposals, solution briefs, and architecture documents.
            Ability to guide CXO-level stakeholders and handle technical deep-dive discussions.
            Soft Skills

            Strategic thinker with strong problem-solving and analytical abilities.
            Confident communicator with excellent articulation skills.
            Team collaborator, proactive, detail-oriented, and customer-obsessed.
            Preferred Qualifications:

            Microsoft certifications:
            Azure AI Engineer Associate
            Azure Solutions Architect Expert
            Azure Data Engineer Associate
            Experience in enterprise data modernization, analytics, or cloud transformation.
            Exposure to domain-specific AI use cases (banking, insurance, retail, government, manufacturing).
            """
        },
        {
            "name": "Sales - AI Data Scientist",
            "jd": """
            Summary
            Imagine what you could do here. At Apple, new ideas have a way of becoming outstanding products, services, and customer experiences very quickly. Bring passion and dedication to your job, and there's no telling what you could accomplish.nnApple’s Sales organization generates the revenue needed to fuel our ongoing development of products and services.

            This, in turn, enriches the lives of hundreds of millions of people around the world. We are, in many ways, the face of Apple to our largest customers.nnApple's Decision Intelligence (DI) team is looking for a versatile individual who is passionate about crafting, implementing, and operating analytical solutions that have a direct and measurable impact on Apple Sales and its customers.

            Description

            As a DI Data Scientist, you will employ predictive modeling, data visualization, and statistical analysis techniques to build end-to-end solutions for internal collaborators, using sales performance data, market data, programs, external data, etc.nnThis role will operate in both capacities, to augment existing data solutions, as well as innovate and inventing data science projects, crafting analytic experiences that simplify data into insights and catalyze decision-making.nnAnalytics is a team sport, and in your role, you will be key in leading and influencing teams on the translation of business problems and questions into data science models.

            Minimum Qualifications

            We're looking for someone with an eagerness and ability to learn new skills and solve dynamic problems in an encouraging and expansive environment.n

            Familiarity with vector similarity search, RAG architectures, and LLM prompt evaluation.n

            Experience co-developing with software engineers in production environments.n

            Ability to lead development projects from start to finish.n

            Comfort with ambiguity. Ability to structure complex analysis through data analysis and strategy research.n

            Collaborate closely with business teams to deep dive into business performance and improve reporting dashboards on key operational metrics.n4+ years of experience in a Data Visualization, Data Science, Data Analysis, or Data Translation role, with a keen eye for design and attention to detail.n

            Applied knowledge of statistical data analysis, predictive modeling classification, Time Series techniques, sampling methods, multivariate analysis, hypothesis testing, and drift analysis.n

            Proficiency in SQL and experience with at least one major data analytics platform, such as Hadoop, Spark, or Snowflake.n

            Expertise with data visualization tools (such as Tableau, d3, plotly, etc.) for data analysis and presentation. Experience with Tableau Server, TabPy, and Extensions is a plus.n

            Proficiency in programming languages, tools, and frameworks like Python, Git, Notebooks, Dataiku, and Streamlit.n

            Knowledge of project management and productivity tools such as Wrike, Sketch. n

            Strong time management skills with the ability to collaborate across multiple teams.n

            Knowledge of best practices in data analysis, data visualization, and data science.n

            Able to balance competing priorities, long-term projects, and ad hoc requirements.n

            Ability to work in a fast-paced, dynamic, constantly evolving business environment.n

            Bachelors's degree in Computer Science, Statistics, Mathematics, Engineering, Economics, Applied Mathematics, Machine Learning, or a related field.

            Preferred Qualifications

            Experience with observability tools for LLMs (e.g., LangSmith, Truera, Weights u0026 Biases)n

            Proven experience working with LLMs and GenAI frameworks (LangChain, LlamaIndex, etc.)n

            Strong experience articulating and translating business questions into data solutions.n

            Communicate results and insights effectively to partners and senior leaders, as well as both technical and non-technical audiences.n

            Experience with anomaly detection and causal inference models.n

            Sound communication skills - adept at messaging domain and technical content, at a level appropriate for the audience. Strong ability to gain trust with stakeholders and senior leadership.n

            Familiarity with embedding, retrieval algorithms, agents, and data modeling for vector development graphs. n

            Advanced Degree (MS or Ph.D.) in Economics, Electrical Engineering, Statistics, Data Science, or a similar quantitative field.
            """
        },
        {
            "name": "Business Analyst",
            "jd": """
            Now we're looking for a Business Analyst, who's ready to jump in, learn fast, and help shape the next wave of human-AI experiences.

            What You'll Do

            Join project teams working on real client challenges (think: AI for nurse handovers, smarter cities, future banking).

            Help run discovery and design workshops mapping journeys, capturing requirements, and prototyping wireframes.

            Translate ideas into structured insights: business requirements, user stories, process flows.

            Explore GenAI tools, data, and design thinking methods to supercharge client solutions.

            Collaborate with strategists, designers, engineers, and client stakeholders.

            Contribute to the “big picture” while sweating the details slides, user flows, or that one killer insight.

            Work on impactful projects in Southeast Asia not just “slides in the corner,” but initiatives that touch citizens, patients, and customers at scale.

            Learn directly from senior strategists, designers, and AI innovators (your mentors are advising governments and Fortune 500s).

            Experiment with the latest Microsoft tech stack (Azure, OpenAI, Power Platform, Dynamics, Azure).

            Build a portfolio of strategy + design work that shows you can bridge business and technology.

            Get the energy of a startup with the backing of a global joint venture (Accenture + Microsoft).

            You will also,

            Work on innovations that matter.

            Learn the language of both boardrooms and design studios.

            See how strategy and tech actually move from whiteboard to world.



            Characteristics that can spell success for this role

            Would be great if you have Bachelor Degree in Computer Science or relevantly

            Min 3+ years of experiences, preferably working with Management Consulting environment to manage diversified industry projects

            Would be great if you have recent experience on AI projects

            A mindset of learn fast, add value, and leave your mark.

            Curiosity for how strategy, design, and tech collide.

            Strong analytical and communication skills you can untangle complexity and tell a crisp story.

            A comfort with ambiguity: you don’t wait for perfect instructions; you explore, test, and learn.

            Interest in digital experiences, GenAI, and how organizations adapt to disruption.

            Bonus: experience with wireframing (Figma, Miro, PowerPoint hacks welcome), process mapping, or research
            """
        }
    ]
    
    # Run test
    print("Starting A/B test...")
    results = run_ab_test(test_cases)
    
    # Analyze
    analyze_results()