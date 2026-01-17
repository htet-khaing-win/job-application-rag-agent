# Job Application RAG Agent

Multi-agent RAG system that generates personalized, ATS-optimized cover letters by matching candidate resumes to job descriptions using hybrid vector search, LLM orchestration, and privacy-first design.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Safety & Privacy](#safety--privacy)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

  ## Overview

I hate doing repetitive things. Especially writing cover letters when there's no guarantee a human will ever read them. I'm also skeptical about sharing my resume with LLMs, given the risks of leaking sensitive personal info, hallucinated qualifications and generic, AI-sounding output. So I automated the entire workflow. 

This project turns a tedious, error-prone process into a structured, verifiable system that generates personalized, ATS-friendly cover letters without hallucinating qualifications or leaking sensitive data.

If this saves you time during job hunting, that's a win in my book.

**Good Luck out there!**

---

## Key Features

### Privacy-First Architecture
- **Presidio PII Guard**: Automatically redacts names, emails, phone numbers, URLs, locations, and certificates before vectorization
- **Custom Burmese Name Recognition**: Extended NER model for multi-cultural name patterns
- **Local Processing**: All data processing runs on-device using Ollama LLMs

### Intelligent Document Processing
- **Multi-Format Support**: PDF, DOCX, TXT parsing
- **Section-Aware Chunking**: Detects resume sections (Experience, Education, Skills) for better retrieval
- **Hybrid Search**: Combines dense embeddings (Nomic) with sparse BM25 for keyword matching
- **Namespace Isolation**: Separate Pinecone namespaces prevent resume cross-contamination

### Multi-Agent Orchestration
- **Generator Agent**: Creative cover letter writing
- **Critic Agent**: Analytical grading and verification
- **Stateful Workflow**: LangGraph manages conditional routing and retry logic
- **Iterative Refinement**: Up to 3 critique loops for quality

### Real-Time Company Research
- **Tavily AI Integration**: Fetches company mission, values, and recent news
- **Contextual Personalization**: References actual company initiatives in opening paragraphs

### Anti-Hallucination Safeguards
- **Claim Verification Node**: Cross-references every statement against resume chunks
- **Skill Classification**: Separates skills with project evidence from skills only listed
- **Post-Generation Detection**: Auto-removes sentences with unverified claims
- **Transfer Learning Prompts**: Addresses unverified skills through demonstrated adaptability

---

## System Architecture

### High-Level Flow

```
[HITL: User pastes Job Description + Company Name]
                    ↓
Job Description → Validation → Company Research → Vector Retrieval
                                                           ↓
                                        Resume Chunks → LLM Grading
                                                           ↓
                                        Grade ≥60% → Generate Summary
                                                           ↓
                                        Claim Verification → Cover Letter
                                                           ↓
                                        Critique → Refine (max 3x) 
                                                           ↓
                            [HITL: Review Generated Cover Letter]
```

### Component Breakdown

**1. Ingestion Pipeline** (`database.py`)
```
Resume → Format Detection → Text Extraction → PII Redaction → 
Section Splitting → Chunking → Nomic + BM25 Embeddings → Pinecone
```

**2. Retrieval System**
- **Hybrid Search**: α=0.4 weighting (60% dense, 40% sparse)
- **Cross-Namespace**: Queries all resumes simultaneously
- **Top-K Selection**: Retrieves 10 best chunks across all resumes

**3. Verification Layer**
- **Skills with Evidence**: Project-backed claims (specific metrics allowed)
- **Skills without Evidence**: Listed skills (generic mention only)
- **Unverified Skills**: Job requirements not in resume (transfer learning approach)

---

## Safety & Privacy

### PII Protection
**Presidio Configuration**:
- NLP Engine: spaCy's `en_core_web_trf` (transformer-based)
- Custom patterns: LinkedIn, GitHub, phone numbers
- Burmese name dictionary: 50+ common name components

**Redaction Examples**:
```
Original: "Htet Khaing Win, +1-555-123-4567, linkedin.com/in/htetkhaingwin"
Redacted: "[CANDIDATE_NAME], [PHONE_NUMBER], [LINKEDIN_URL]"
```

### Hallucination Prevention
**Multi-Layer Defense**:
1. **Prompt Engineering**: Explicit allow/deny skill lists
2. **Post-Generation Check**: Regex detection of unverified skills
3. **Auto-Fix**: Sentence removal with fallback to error state
4. **Minimum Content**: Rejects letters <200 words after cleanup

**Example Detection**:
```
Detected: "I have extensive Terraform experience"
Resume: Terraform not mentioned anywhere
Action: Remove sentence or trigger fallback
```

### Error Handling
**Fallback Triggers**:
- Invalid job description format
- Empty resume database
- Retrieval score <50% after 2 rewrite attempts
- Verification failure (insufficient verifiable skills)
- Hallucination detected in final output

---

## Installation

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running
- Pinecone account (free tier)
- Tavily API key (free tier: 1000 requests/month)

### Step 1: Clone Repository
```bash
git clone https://github.com/htet-khaing-win/job-application-rag-agent.git
cd job-application-rag-agent
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

### Step 3: Download Ollama Models or Get your own API keys
```bash
ollama pull mistral:7b-instruct
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### Step 4: Configure Environment
Create `.env` file:
```env
PINECONE_API_KEY=your_pinecone_api_key
TAVILY_API_KEY=your_tavily_api_key
```

**Get API Keys**:
- Pinecone: https://app.pinecone.io/ → API Keys → Create Key
- Tavily: https://app.tavily.com/ → Sign up → Copy Key

### Step 5: Fit BM25 Model
**IMPORTANT**: Run this before ingesting resumes to enable hybrid search.

```bash
python utilities/Fitting.py
```

This creates `utilities/fitted_bm25.json` for sparse vector encoding. Without this, the system falls back to default (cold) BM25 encoder with reduced accuracy.

### Step 6: Ingest Resumes
```bash
python database.py /path/to/resume.pdf
```

**Supported formats**: PDF, DOCX, TXT  
**Max file size**: 5MB  
**Auto-applies**: PII redaction, section detection, chunking

---

## Usage

### Generate Cover Letter

```bash
python main.py
```

**Interactive Flow**:
```
Paste your Job Description below.
When finished, type 'DONE' on a new line and press Enter.

[Paste job description here]
DONE

Please enter the company name: PeePeePooPooAI
 Researching company information...
 Retrieved 10 relevant chunks. Top Match Score: 87.3%.
 
Your Cover Letter is Ready
---------------------------------

With 3 years of experience building production ML systems, including a 
recommendation engine that processes 2M+ daily requests with 40ms latency, 
I'm excited to apply for the Senior ML Engineer position at PeePeePooPooAI...

[Full letter]

---------------------------------
 Retrieval Score: 87/100
 Refinement Iterations: 2
```

### Low Match Example

**Input**: Job requiring skills not in resume  
**Output**:
```
Quality Check Failed - Unverified Claims Detected

Cover letter contained unverified claims about: Terraform, Kubernetes. 
Auto-fix resulted in insufficient content (156 words).

Solutions:
1. Upload a resume matching this role better
2. Try a job posting aligned with your background
3. Ensure resume includes specific project descriptions
```

### Manage Resumes

**List stored resumes**:
```python
from database import list_stored_resumes
print(list_stored_resumes())
```

**Delete resume**:
```python
from database import delete_resume_from_pinecone
delete_resume_from_pinecone("resume.pdf")
```

---

## Project Structure

```
job-application-rag-agent/
│
├── main.py                 # Entry point and CLI
├── graph.py                # LangGraph workflow definition
├── node.py                 # Agent nodes (generate, verify, critique)
├── state.py                # Pydantic state schema
├── database.py             # Pinecone operations and ingestion
├── privacy.py              # Presidio PII configuration
│
├── utilities/
│   ├── Fitting.py          # BM25 model fitting (run first)
│   └── fitted_bm25.json    # Generated sparse encoder
│
├── requirements.txt        # Dependencies
├── .env                    # API keys (gitignored)
└── README.md               # This file
```

### Key Files

**`database.py`**:
- `ingest_resume_to_pinecone()`: PII redaction + vectorization
- `retrieve_resumes_node()`: Hybrid search across all resumes
- `split_by_sections()`: Resume section detection

**`node.py`**:
- `verify_claims_node()`: Cross-references summary against resume
- `write_cover_letter_node()`: Generation with hallucination check
- `critique_letter_node()`: Quality evaluation

**`graph.py`**:
- Conditional routing (rewrite, refine, fallback)
- Dual-LLM orchestration (generator vs critic)

---

## Development Setup
```bash
pip install -r requirements.txt
python utilities/Fitting.py
python database.py test_resume.pdf
pytest tests/
```


## License

MIT License - see [LICENSE](LICENSE) file for details.

---
