import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from state import GraphState
from docx import Document
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def get_index(index_name: str, dimension: int = 768):
    """
    Ensures the Pinecone vector infrastructure is ready.
    
    Purpose:
        - Prevents 'Index Not Found' errors by auto-provisioning.
        - Ensures dimension matches our Google Gemini Embedding model (768).
        - Configures ServerlessSpec for cost-effective scaling.
    """
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name = index_name,
            dimension = dimension,
            metric = "cosine",
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

index = get_index(index_name="resume-index", dimension=768)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("API_KEY") 
)


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
    return " ".join(text.split()).strip()

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
        2. Clean: Normalize text for better embedding quality.
        3. Split: Segment text using RecursiveCharacterTextSplitter (500 chars, 80 overlap).
        4. Vectorize: Generate 768-dimension embeddings for each segment.
        5. Storage: Batch upload to Pinecone with source tracking.

    Input: 
        - file_path (str): Absolute or relative path to the candidate's resume file.
        
    Side Effects: 
        - Updates the Pinecone 'resume-index' with new vector entries.
    """

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

    cleaned = clean_text(text)
    
    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(cleaned)
    
    # Cascading Delete (Deleting Chunks if the upload file has the same name)
    print(f" Checking for existing records of {filename}...")
    try:
        # deleting chunks using metadata filter
        index.delete(filter={"source": filename})
    except Exception as e:
        print(f" Clean-up note: {str(e)}")


    # Embedding & Upserting
    vectors = []
    embedded_chunks = embeddings.embed_documents(chunks)

    for i, (chunk, vector_val) in enumerate(zip(chunks, embedded_chunks)):
        vectors.append({
            "id": f"{filename}#{i}", # "#" symbol is there for ID to be more deterministic
            "values": vector_val,
            "metadata": {
                "text": chunk, 
                "source": filename,
                "doc_type": "resume",
                "ingested_at": time.time() 
            }
        })
    
    index.upsert(vectors=vectors)
    print(f" Successfully ingested: {filename}")

def list_stored_resumes():
    """
    Fetches a unique list of all resumes currently indexed in Pinecone.
    
    Returns:
        List[str]: A sorted list of unique filenames found in metadata.
    """
    stats = index.describe_index_stats()
    namespaces = stats.get('namespaces', {})
    return sorted(list(namespaces.keys()))


def delete_resume_from_pinecone(filename: str):
    """
    Deletes all vector chunks associated with a specific filename.
    """
    try:
        index.delete(delete_all = True, namespace = filename)
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


def retrieve_resumes_node(state: GraphState) -> GraphState:
    """
    The Researcher - Queries Pinecone vector database for relevant resume chunks.
    
    Purpose:
        - Embeds the cleaned JD into vector space
        - Retrieves top-K most relevant resume sections from Pinecone
        - Returns ranked candidate data for grading
    
    Input: state.cleaned_jd - Processed job description
    Output: state.retrieved_chunks - List of resume text chunks with scores
    
    """
    query_vector = embeddings.embed_query(state.cleaned_jd)
    all_matches = []
    resume_list = list_stored_resumes()

    for names in resume_list:
        resume = index.query(
            vector = query_vector,
            top_k = 10,
            include_metadata = True,
            namespace = names
        )
        all_matches.extend(resume['matches'])

    # Sort  top 5 based on score
    all_matches = sorted(all_matches, key=lambda x: x['score'], reverse=True)[:5]
 
    if not all_matches:
        print("No Matching Resumes found in any namespace.")
        return {**state, "retrieved_chunks": [], "grading_feedback": "No resumes found."}

    retrieved_chunks = [match['metadata']['text'] for match in all_matches]
    return {**state, "retrieved_chunks": retrieved_chunks}