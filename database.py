import os
from dotenv import load_dotenv
from pinecone import Pinecone, SeverlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from state import GraphState
from docx import Document
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("resume-index")

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
    results = index.query(
        vector=[0.0] * 768, 
        top_k=10000, 
        include_metadata=True,
        filter={"doc_type": "resume"}
    )
    
    if not results['matches']:
        return []
    
    unique_filenames = {match['metadata']['source'] for match in results['matches']}
    return sorted(list(unique_filenames))


def delete_resume_from_pinecone(filename: str):
    """
    Deletes all vector chunks associated with a specific filename.
    """
    try:
        index.delete(filter={"source": filename})
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
    results = index.query(
        vector = query_vector,
        top_k = 5,
        include_metadata = True,
        filter = {
            "doc_type": "resume"
        }
    )

    if not results['matches']:
        print("No Matching Resumes found in Pinecone.")
        return {"retrieved_chunks": [], "grading_feedback": "No resumes found in database."}

    retrieved_chunks = [match['metadata']['text'] for match in results['matches']]

    return {**state, "retrieved_chunks": retrieved_chunks}