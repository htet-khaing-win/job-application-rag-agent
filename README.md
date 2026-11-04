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

### Step 4: Add API Keys (no `.env` required)
Keys live in an **encrypted local vault** (Fernet; master key in your OS keychain). Nothing is hard-coded or stored in `.env`.

```bash
streamlit run app.py            # GUI: API Keys tab
# or, headless:
python main.py add-key          # prompted, input hidden
python main.py list-keys        # masked
```

**Get API Keys**: Pinecone (https://app.pinecone.io/), Tavily (https://app.tavily.com/), plus any chat provider you want (OpenAI, Gemini, Anthropic; Ollama needs none).

For CI/headless only, set `JOBAGENT_ALLOW_ENV_KEYS=1` to let the vault fall back to `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, `PINECONE_API_KEY`.

### Per-agent models (customisation)
Each sub-agent has its own provider / model / key alias (**Agents & Models** tab, stored in `~/.job_application_agent/profile.json`, no secrets):

| Agent | Role |
|---|---|
| JD Analyst | validates + distils the job description |
| Researcher | live company research (Tavily) |
| Retriever | hybrid search over resumes (embeddings + Pinecone) |
| Copywriter | drafts the letter |
| Fact Checker | blocks/repairs claims about skills with no resume evidence |
| Hiring Manager | critiques the draft |
| Editor | applies edits |

The second model in each chat role is its automatic failover.

### Self-healing behaviour
Retry with backoff → provider failover → circuit breaker → JSON output repair → graceful degradation (e.g. critic outage ships the verified draft; writer outage routes to a clear fallback). Ungrounded skill claims are repaired by the Fact Checker or deterministically removed. Every recovery action appears in the GUI "Self-healing log".

### GUI features
`streamlit run app.py` (binds to `127.0.0.1` only):

- **Generate**: live view of the writer's draft as it streams in, per-agent progress, a pre-run cost/token estimate, spend-this-month vs per-key caps (the run is blocked if a cap would be exceeded), a run timeout, an editable draft with **Re-check grounding** (flags sentences that claim skills your resume doesn't support), a skills-gap table (evidence chunk per verified skill), `.txt` / `.docx` / `.pdf` export, and recent-letter history with a diff against the previous letter for the same company. Streamlit's own Stop button cancels a run.
- **Agents & Models**: presets, per-agent provider/model/key, a **Test connection** button per agent (and **Test all**) with 🟢/🔴/⚪ status lights, and a side-by-side profile comparison that lists only the differences.
- **API Keys / Security**: key age, last use and rotation reminders, per-key monthly spend caps, an optional **vault passphrase** (lock/unlock; while locked the app shows only the unlock form), and editable retrieval settings (`alpha`, max resumes searched, top-k) persisted in your profile.
- **Runs**: run ids, durations, cost and p50/p95 per agent from the local run log. The log stores timings, token counts and cost only, never letter or resume text.
- **Chaos mode** (sidebar): a demo that injects random provider faults so you can watch the self-healing log recover. Leave it off for real use.

Tables are rendered as escaped markdown, not `st.dataframe`, so resume text can't inject markup and the GUI doesn't depend on pandas.

### Evaluation
`python -m evals <command>` (see `--help`). Commands: `retrieval` (recall@k / precision@k / MRR / nDCG with bootstrap CIs and paired tests), `letters` (claim-level faithfulness), `chaos` (fault injection against fake models), `ablate` and `sweep` (**live**: need your keys), `label` (gold-set tooling), `perf` (latency/token/cost percentiles) and `regress` (CI regression gate against a stored baseline). Reports are written to `evals/reports/` with the git SHA.

**Building a gold set from real listings.** The repository can generate a gold set from public AI/ML engineering listings on MyCareersFuture (Singapore's government job portal) and your own resumes:

```bash
python -m evals.mcf fetch --target 50                     # public API, rate-limited; -> evals/gold_src/listings.json
python -m evals.goldgen build --resumes resumes           # -> evals/gold_set.json
python -m evals.goldgen audit -n 10                       # spot-check labels by eye
python -m evals retrieval --gold evals/gold_set.json --collect baseline --out baseline.json   # live
python -m evals retrieval --gold evals/gold_set.json --collect hybrid   --out hybrid.json     # live
python -m evals retrieval --gold evals/gold_set.json --runs-a baseline.json --runs-b hybrid.json
```

How it is labelled (a **silver standard**, not human-adjudicated): required skills come from the listing text plus MyCareersFuture's own structured skill tags, mapped through a skill taxonomy (`evals/taxonomy.py`); a skill is *supported* if the resume text matches the same taxonomy; the *relevant spans* are the resume lines that evidence the supported skills. Retrieval is scored by matching retrieved chunk text against those spans, so the labels survive re-chunking and don't depend on Pinecone chunk ids. Skills merely *listed* on a resume count as supported, and taxonomy gaps cause missed or spurious skills, so audit a sample and treat results as indicative until a human has reviewed them. Listing text belongs to the employers: `evals/gold_src/` and `evals/gold_set.json` are gitignored, so keep them local. The hand-labelled workflow (`python -m evals label ...`) remains available for a human-reviewed gold set.

### Developer tooling
- **Config**: `pyproject.toml` configures ruff (correctness and security rules), mypy and pytest. Pytest markers: `heavy` (needs presidio + the spaCy model) and `live` (calls real providers; never run in CI).
- **Make**: `make test | test-fast | lint | type | audit | eval | chaos | perf | check-staged | lock | run | docker`.
- **Dependencies**: `requirements.txt` (runtime) and `requirements-dev.txt` (test/lint/audit). `scripts/lock.sh` generates `requirements.lock` with sha256 hashes via pip-tools; the Dockerfile installs from it with `--require-hashes` when it exists. The lock file is **not** checked in until you run the script (it needs network access).
- **CI** (`.github/workflows/test.yml`): fast tests, a separate `heavy` job (allowed to fail while the spaCy download proves stable), and a security job (`pip-audit`, ruff, mypy, gitleaks). Read-only permissions, no provider keys or repository secrets. Dependabot covers pip, Actions and Docker.
- **Pre-commit**: `.pre-commit-config.yaml` (ruff, gitleaks, private-key detection, large-file check). Run `pre-commit install` once.
- **Docker**: non-root image; `make docker` publishes the GUI on `127.0.0.1` only.

### Security notes
Secrets are redacted from logs/errors; JD and web research are fenced as untrusted data in prompts; uploads are validated by extension, magic bytes and size; filenames are sanitised; the GUI binds to `127.0.0.1` with XSRF protection. Do not expose the GUI to a network without an authenticating reverse proxy.

Uploads: `.docx` files must be structurally valid (contain `word/document.xml`) and are rejected when they expand beyond 50MB, exceed 2000 parts, contain unsafe paths, or have a compression ratio above 100:1. The redactor scrubs well-known key shapes, `key=value` pairs and opaque `Bearer` tokens. Passphrase protection is crash-safe: the vault and its metadata are committed together through a journal, and start-up recovery completes or rolls back an interrupted change.

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
├── app.py                  # Streamlit GUI
├── export.py               # .docx / .pdf export
├── evals/                  # evaluation harness (python -m evals)
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Test / lint / audit tooling
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
pip install -r requirements-dev.txt
python utilities/Fitting.py
python database.py test_resume.pdf
pytest tests/
```


## License

MIT License - see [LICENSE](LICENSE) file for details.

---
