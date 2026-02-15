import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from state import GraphState
from docx import Document
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import time
from privacy import PIIGuard
import sys
from pinecone_text.sparse import BM25Encoder
import re
import asyncio
from langsmith import traceable

load_dotenv()
pii_guard = PIIGuard()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
BM25_PATH = "utilities/fitted_bm25.json"

_bm25 = None
_index = None
_embeddings = None

def get_bm25():
    global _bm25
    if _bm25 is None:
        if os.path.exists(BM25_PATH):
            _bm25 = BM25Encoder()
            _bm25.load(BM25_PATH)
        else:
            _bm25 = BM25Encoder.default()
    return _bm25

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        # _embeddings = OllamaEmbeddings(model="nomic-embed-text")
        # _embeddings = GoogleGenerativeAIEmbeddings(
        #             model="models/text-embedding-001",
        #             google_api_key=os.getenv("GEMINI_API_KEY")
        #                 )
        _embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # 1536 dimensions
            api_key=os.getenv("OPENAI_API_KEY")
        )
    return _embeddings

SECTION_PATTERNS = {
    "EXPERIENCE": r"(WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|EMPLOYMENT|CAREER HISTORY)",
    "EDUCATION": r"(EDUCATION|ACADEMIC BACKGROUND|UNIVERSITY)",
    "SKILLS": r"(SKILLS|TECHNICAL SKILLS|CORE COMPETENCIES|TECHNOLOGIES)",
    "PROJECTS": r"(PROJECTS|PERSONAL PROJECTS|KEY PROJECTS)",
    "CERTIFICATIONS": r"(CERTIFICATIONS?|CERTIFICATES?|LICENSES)",
    "SUMMARY": r"(SUMMARY|PROFILE|ABOUT|OBJECTIVE)"
}

def split_by_sections(text: str):
    """Extract resume sections with fallback."""
    sections = []
    current_section = "OTHER"
    buffer = []

    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            buffer.append(line)
            continue
            
        upper = line.upper()
        matched = False

        for section, pattern in SECTION_PATTERNS.items():
            if re.search(pattern, upper):
                # Save previous section
                if buffer:
                    sections.append((current_section, "\n".join(buffer)))
                buffer = []
                current_section = section
                matched = True
                break

        if not matched:
            buffer.append(line)

    # Don't forget last section
    if buffer:
        sections.append((current_section, "\n".join(buffer)))

    return sections

def get_index(index_name: str, dimension: int = 1536):
    """
    Ensures the Pinecone vector infrastructure is ready.
    
    Purpose:
        - Prevents 'Index Not Found' errors by auto-provisioning.
        - Ensures dimension matches our Google Gemini Embedding model (1536).
        - Configures ServerlessSpec for cost-effective scaling.
    """
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name = index_name,
            dimension = dimension,
            metric = "dotproduct", # for hybrid search
            spec = ServerlessSpec(
                cloud="aws", 
                region="us-east-1"
            )
        )

        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"Index {index_name} is ready")

    else:
        desc = pc.describe_index(index_name)
        if desc.dimension != dimension:
            raise ValueError(f"Index dimension mismatch! Expected {dimension}, found {desc.dimension}")
        print(f" Connected to existing index: {index_name}")
    return pc.Index(index_name)

def get_pinecone_index():
    global _index
    if _index is None:
        # We hardcode the config here, or load it from env vars
        _index = get_index(index_name="resume-index", dimension=1536)
    return _index


# Helpers
def parse_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content: text += content + "\n"
    return text

def parse_docx(file_path):
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)

def clean_text(text):
    """
    Clean excessive whitespace while preserving document structure.
    
    - Converts multiple spaces to single space within lines
    - Preserves line breaks for section detection
    - Removes completely empty lines
    """
    lines = []
    for line in text.splitlines():
        cleaned_line = ' '.join(line.split())  # Multiple spaces to one
        if cleaned_line:  # Skip completely empty lines
            lines.append(cleaned_line)
    
    return '\n'.join(lines)

# Ingestion Flow 
def ingest_resume_to_pinecone(file_path):
    """
    Orchestrates the full document ingestion pipeline for the vector database.
    
    Purpose:
        - Automatically detects file format (PDF, DOCX, TXT) and extracts raw text.
        - Standardizes text by removing excessive whitespace and formatting artifacts.
        - Breaks down long resumes into semantically meaningful chunks to respect LLM context limits.
        - Transforms text chunks into high-dimensional vectors using Google Gemini Embeddings.
        - Upserts vectors into Pinecone with associated metadata for filtered retrieval.

    Process Flow:
        1. Parse: Extract content based on file extension.
        2. Clean: Anonymize and Normalize text for better embedding quality.
        3. Split: Segment text using SemanticChunker.
        4. Vectorize: Generate 1536-dimension embeddings for each segment.
        5. Storage: Batch upload to Pinecone with source tracking.

    Input: 
        - file_path (str): Absolute or relative path to the candidate's resume file.
        
    Side Effects: 
        - Updates the Pinecone 'resume-index' with new vector entries.
    """
    bm25 = get_bm25()
    index = get_pinecone_index()
    embeddings = get_embeddings()
    filename = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[-1].lower()

    #pdf
    if ext == ".pdf": text = parse_pdf(file_path)

    # docs
    elif ext == ".docx": text = parse_docx(file_path)

    # txt
    else: 
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    
    # Redact the sensitive infor before processing
    safe_text = pii_guard.redact_text(text)

    # Section detection
    sections = split_by_sections(safe_text)

    
    cleaned_sections = []

    for section_name, section_text in sections:
    # Clean excessive whitespace WITHIN lines, but preserve line breaks
        cleaned_text = re.sub(r'[ \t]+', ' ', section_text)  # Multiple spaces
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Multiple newlines
        cleaned_sections.append((section_name, cleaned_text.strip()))

    sections = cleaned_sections

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )
    
    chunks_with_meta = []
    for section_name, section_text in sections:
        if len(section_text.strip()) < 20:  # Skip empty sections
            continue
            
        section_chunks = splitter.split_text(section_text)
        
        for chunk in section_chunks:
            chunk_clean = chunk.strip()
            if len(chunk_clean) < 30:  # Filter junk
                continue
                
            chunks_with_meta.append({
                "text": chunk_clean,
                "section": section_name
            })

    # Extract text for embedding
    texts = [c["text"] for c in chunks_with_meta]
    # generate vectors (dense+sparse)
    dense_vectors = embeddings.embed_documents(texts)

    # generate the keyword weight for every chunk
    sparse_vectors = bm25.encode_documents(texts)
    
    # Cascading Delete (Deleting Chunks if the upload file has the same name)
    print(f" Checking for existing records of {filename}...")
    try:
        index.delete(delete_all=True, namespace=filename)
    except Exception: pass


    # Embedding & Upserting
    vectors = []

    for i, (chunk_obj, dense, sparse) in enumerate(zip(chunks_with_meta, dense_vectors, sparse_vectors)):
        vectors.append({
            "id": f"{filename}#{i}", # "#" symbol is there for ID to be more deterministic
            "values": dense,
            "sparse_values": sparse,
            "metadata": {
                "text": chunk_obj["text"],
                "section": chunk_obj["section"],
                "source": filename,
                "doc_type": "resume",
                "ingested_at": time.time() 
            }
        })
    
    index.upsert(vectors=vectors, namespace=filename)
    print(f" Successfully ingested: {filename}")

def list_stored_resumes():
    """
    Fetches a unique list of all resumes currently indexed in Pinecone.
    
    Returns:
        List[str]: A sorted list of unique filenames found in metadata.
    """
    index = get_pinecone_index()
    stats = index.describe_index_stats()
    namespaces = stats.get('namespaces', {})
    return sorted(list(namespaces.keys()))


def delete_resume_from_pinecone(filename: str):
    """
    Deletes all vector chunks associated with a specific filename.
    """
    index = get_pinecone_index()  # Get lazy-loaded instance
    try:
        index.delete(delete_all=True, namespace=filename)
        print(f"Deleted all chunks for {filename}")
        return True
    except Exception as e:
        print(f"Error deleting {filename}: {str(e)}")
        return False
    
def validate_resume_file(file_path):
    """
    Pre-flight check for file integrity and size.
    """
    valid_exts = ['.pdf','.docx','.txt']
    ext = os.path.splitext(file_path)[-1].lower()

    if ext not in valid_exts:
        return False, "Unsupported file format."
    
    # Check size < 5MB
    if os.path.getsize(file_path) > 5 * 1024 * 1024:
        return False, "File too large (Max 5MB)."
    
    return True, "Completed"

@traceable(run_type="tool", name="pinecone_retrieval")
async def retrieve_resumes_node(state: GraphState) -> dict:
    """
    The Researcher - Queries Pinecone vector database for relevant resume chunks.
    
    Purpose:
        - Embeds the cleaned JD into vector space
        - Retrieves top-K most relevant resume sections from Pinecone
        - Returns ranked candidate data for grading
    
    Input: state.cleaned_jd - Processed job description
    Output: state.retrieved_chunks - List of resume text chunks with scores
    
    """

    # Initialize lazy-loaded resources
    embeddings = get_embeddings()
    bm25 = get_bm25()
    index = get_pinecone_index()

    # Parallelize embedding generation
    dense_query, sparse_query, resume_list = await asyncio.gather(
        asyncio.to_thread(embeddings.embed_query, state.cleaned_jd),
        asyncio.to_thread(bm25.encode_queries, state.cleaned_jd),
        asyncio.to_thread(list_stored_resumes)
    )


    if not resume_list:
        print(" No resumes found in database.")
        return {
            "retrieved_chunks": [], 
            "error_type": "no_resumes",
            "grading_feedback": "Database is empty. Please upload resumes first.",
            "vector_relevance_score": 0.0
        }

    capped_list = resume_list[:2]
    # Helper function for Pinecone query
    async def query_namespace(namespace):
        return await asyncio.to_thread(
            index.query,
            vector=dense_query,
            sparse_vector=sparse_query,
            top_k=10,
            include_metadata=True,
            alpha=0.4,
            namespace=namespace
        )
    
    # Gather all queries in parallel
    results = await asyncio.gather(
        *[query_namespace(ns) for ns in capped_list],
        return_exceptions=True
    )
    
    all_matches = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error querying namespace {capped_list[i]}: {result}")
        else:
            all_matches.extend(result.get('matches', []))

    if not all_matches:
        return {
                "retrieved_chunks": [], 
                "vector_relevance_score": 0.0, 
                "grading_feedback": "No relevant matches."}

    # Aggregation per Resume (Diversity)
    sorted_matches = sorted(all_matches, key=lambda x: x['score'], reverse=True)
    final_matches = sorted_matches[:10]

    if not final_matches:
        print(" No matches passed the relevance threshold.")
        return {
            "retrieved_chunks": [], 
            "grading_feedback": "Mismatch: No resumes are sufficiently relevant to this Job Description.",
            "vector_relevance_score": 0.0 # Fallback Signal
        }

    retrieved_chunks = [
        {
            "text": m["metadata"]["text"],
            "source": m["metadata"]["source"],
            "score": round(m["score"], 4),
            "chunk_id": m["id"]
        }
        for m in final_matches 
    ]

    vector_score = round(final_matches[0]['score'] * 100, 2)
    print(f" Retrieved {len(retrieved_chunks)} relevant chunks.")

    return {
        "retrieved_chunks": retrieved_chunks, 
        "vector_relevance_score": vector_score,
    }

if __name__ == "__main__":
    # Check if a file path was actually provided in the terminal
    if len(sys.argv) > 1:
        resume_path = sys.argv[1]
        print(f"Starting ingestion for: {resume_path}")
        ingest_resume_to_pinecone(resume_path)
    else:
        print("Error: Please provide a file path // python database.py resume.pdf")