import os
from state import GraphState
import time
import sys
import re
import asyncio
from langsmith import traceable
from runtime import get_runtime
from security import safe_filename, redact_secrets

BM25_PATH = "utilities/fitted_bm25.json"

_pii_guard = None
_bm25 = None
_bm25_mode = "fitted"

DEFAULT_SETTINGS = {"alpha": 0.4, "max_resumes": 10, "top_k": 10}
_QUERY_CONCURRENCY = 8


def set_bm25_mode(mode: str) -> None:
    """"fitted" (default; uses utilities/fitted_bm25.json if present) or "cold" (generic encoder).

    Evals should use "cold" when the fitted encoder was trained on the corpus being queried,
    otherwise sparse scores leak corpus statistics.
    """
    global _bm25, _bm25_mode
    if mode not in ("fitted", "cold"):
        raise ValueError("bm25 mode must be 'fitted' or 'cold'")
    if mode != _bm25_mode:
        _bm25 = None
    _bm25_mode = mode


def get_settings() -> dict:
    """Retrieval settings from the active runtime profile (alpha, max_resumes, top_k)."""
    merged = dict(DEFAULT_SETTINGS)
    try:
        extra = getattr(get_runtime(), "settings", None)
        if isinstance(extra, dict):
            merged.update({k: v for k, v in extra.items() if k in merged and v is not None})
    except Exception:  # noqa: BLE001 - no runtime (tests/scripts): defaults
        pass
    return merged


def _embedding_model_key() -> str:
    try:
        b = get_runtime().profile.roles["embeddings"][0]
        return f"{b.provider}:{b.model}"
    except Exception:  # noqa: BLE001
        return "default"


def select_fair_matches(matches_by_ns: dict, top_k: int) -> list:
    """Merge per-resume matches so no single resume can crowd the others out entirely.

    Half the slots (at most one per resume) go to each resume's best chunk, ranked by that
    chunk's score; the rest are filled globally by score. Output is sorted by score.
    """
    pool = [(ns, m) for ns, ms in matches_by_ns.items() for m in ms]
    if not pool:
        return []
    chosen, seen = [], set()
    bests = sorted(((ns, max(ms, key=lambda m: m["score"])) for ns, ms in matches_by_ns.items() if ms),
                   key=lambda t: t[1]["score"], reverse=True)
    for ns, m in bests[: max(1, top_k // 2)]:
        chosen.append(m)
        seen.add(m["id"])
    for ns, m in sorted(pool, key=lambda t: t[1]["score"], reverse=True):
        if len(chosen) >= top_k:
            break
        if m["id"] not in seen:
            chosen.append(m)
            seen.add(m["id"])
    return sorted(chosen, key=lambda m: m["score"], reverse=True)[:top_k]


def get_pii_guard():
    global _pii_guard
    if _pii_guard is None:
        from privacy import PIIGuard
        _pii_guard = PIIGuard()
    return _pii_guard


def get_bm25():
    global _bm25
    if _bm25 is None:
        from pinecone_text.sparse import BM25Encoder
        if _bm25_mode == "fitted" and os.path.exists(BM25_PATH):
            _bm25 = BM25Encoder()
            _bm25.load(BM25_PATH)
        else:
            _bm25 = BM25Encoder.default()
    return _bm25


def get_embeddings():
    """Embedding model bound to the 'embeddings' role in the active profile."""
    return get_runtime().embeddings()


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

_index_cache: dict = {}


def get_index(index_name: str, dimension: int = 1536):
    """
    Ensures the Pinecone vector infrastructure is ready (auto-provisions if missing).
    The Pinecone client is built from the vault-held key for the 'vector_store' role.
    """
    pc = get_runtime().pinecone()
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        from pinecone import ServerlessSpec
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="dotproduct",  # for hybrid search
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"Index {index_name} is ready")
    else:
        desc = pc.describe_index(index_name)
        if desc.dimension != dimension:
            raise ValueError(f"Index dimension mismatch! Expected {dimension}, found {desc.dimension}")
    return pc.Index(index_name)


def get_pinecone_index():
    rt = get_runtime()
    name = rt.index_name()
    cache_key = (name, id(rt.pinecone()))
    if cache_key not in _index_cache:
        _index_cache[cache_key] = get_index(index_name=name, dimension=1536)
    return _index_cache[cache_key]


# Helpers
def parse_pdf(file_path):
    import pdfplumber
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content: text += content + "\n"
    return text

def parse_docx(file_path):
    from docx import Document
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
    filename = safe_filename(os.path.basename(file_path))
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
    safe_text = get_pii_guard().redact_text(text)

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
    from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    filename = safe_filename(filename)
    index = get_pinecone_index()  # Get lazy-loaded instance
    try:
        index.delete(delete_all=True, namespace=filename)
        print(f"Deleted all chunks for {filename}")
        return True
    except Exception as e:
        print(f"Error deleting {filename}: {redact_secrets(e)}")
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

def hybrid_scale(dense, sparse, alpha: float):
    """Convex-combine dense and sparse query vectors: dense*alpha, sparse*(1-alpha).

    Pinecone's query() has no `alpha` argument (it would be ignored or rejected), so the
    documented approach is to scale the query vectors client-side. alpha=1 is dense-only.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1]")
    return ([float(v) * alpha for v in dense],
            {"indices": list(sparse["indices"]), "values": [float(v) * (1.0 - alpha) for v in sparse["values"]]})


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

    from cache import cached_embed_query
    cfg = get_settings()
    alpha, top_k = float(cfg["alpha"]), int(cfg["top_k"])
    max_resumes = max(1, int(cfg["max_resumes"]))

    # Parallelize embedding generation (query embeddings are cached per model+text)
    dense_query, sparse_query, resume_list = await asyncio.gather(
        asyncio.to_thread(cached_embed_query, embeddings, _embedding_model_key(), state.cleaned_jd),
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

    namespaces = resume_list[:max_resumes]
    sem = asyncio.Semaphore(_QUERY_CONCURRENCY)

    hybrid_dense, hybrid_sparse = hybrid_scale(dense_query, sparse_query, alpha)

    async def query_namespace(namespace):
        async with sem:
            return await asyncio.to_thread(
                index.query,
                vector=hybrid_dense,
                sparse_vector=hybrid_sparse,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )

    results = await asyncio.gather(*[query_namespace(ns) for ns in namespaces], return_exceptions=True)

    by_ns = {}
    for ns, result in zip(namespaces, results):
        if isinstance(result, Exception):
            print(f"Error querying namespace {ns}: {redact_secrets(result)}")
        else:
            by_ns[ns] = list(result.get('matches', []))  # type: ignore[union-attr]

    final_matches = select_fair_matches(by_ns, top_k)
    if not final_matches:
        return {
            "retrieved_chunks": [],
            "vector_relevance_score": 0.0,
            "grading_feedback": "No relevant matches.",
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
        from runtime import Runtime, Profile, activate_runtime
        from vault import KeyVault
        activate_runtime(Runtime(KeyVault(), Profile.load()))
        resume_path = sys.argv[1]
        print(f"Starting ingestion for: {resume_path}")
        ingest_resume_to_pinecone(resume_path)
    else:
        print("Error: Please provide a file path // python database.py resume.pdf")