# Job Application RAG Agent

A self-healing, multi-agent RAG system that writes tailored cover letters by matching your resumes to a job description. It uses hybrid vector search, per-agent model selection, a Streamlit GUI, and an encrypted local key vault. Letters are fact-checked against your resume before you see them.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Customising Agents and Keys](#customising-agents-and-keys)
- [Self-Healing](#self-healing)
- [Security and Privacy](#security-and-privacy)
- [Evaluation](#evaluation)
- [Developer Tooling](#developer-tooling)
- [Known Limitations](#known-limitations)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

I hate doing repetitive things, especially writing cover letters when there's no guarantee a human will ever read them. I'm also wary of sharing my resume with LLMs: leaked personal information, invented qualifications and generic, AI-sounding output are all real risks. So I automated the workflow and built in the safeguards.

The result is a structured, verifiable pipeline. Your resume is PII-redacted before it is stored. Every letter is checked so it can't claim skills your resume doesn't evidence. The pipeline recovers from provider outages instead of failing.

**Good luck out there!**

---

## Key Features

**Privacy first**
- Presidio PII redaction (names, emails, phones, locations, LinkedIn/GitHub/certificate URLs) runs before anything is embedded or stored, with extra recognition for Burmese name components.
- Separate Pinecone namespace per resume prevents cross-contamination.
- Fully local operation is possible with Ollama models (see [presets](#customising-agents-and-keys)); embeddings must then match your index dimension.

**Hybrid retrieval**
- Dense embeddings plus BM25 sparse vectors, section-aware chunking (Experience, Education, Skills, ...), PDF/DOCX/TXT ingestion.

**Specialised sub-agents**
- JD Analyst, Researcher, Retriever, Analyst, Copywriter, Fact Checker, Hiring Manager and Editor, orchestrated as a LangGraph workflow.
- Each model-backed agent can use its own provider, model and API key.

**Anti-hallucination**
- Skills are checked against retrieved resume evidence. The Copywriter may only mention verified skills, or unverified ones framed as "prepared to learn".
- The Fact Checker finds sentences that claim an unverified skill and has a model rewrite them. If the rewrite doesn't fix it, the sentences are removed, so an ungrounded claim cannot ship.

**Live company research**
- Tavily search for company mission and news, treated as untrusted data in prompts.

**GUI and CLI**
- Streamlit app: generate letters, manage resumes, assign models/keys per agent, test connections, view per-agent timing and the self-healing log, export `.txt`/`.docx`.

---

## How It Works

```
Job description + company
        |
   JD Analyst (validate + distil)
        |
   +----+------------------+
   |                       |
Retriever               Researcher
(hybrid search)         (Tavily)
   |                       |
 Analyst                   |
(skill evidence)           |
   +----------+------------+
              |
        Context join
              |
         Copywriter  --> Fact Checker --> Hiring Manager (critique)
                                              |
                                        Editor (one refinement pass)
                                              |
                                          Cover letter
```

- **Retrieval gate:** if retrieval yields no usable score, or there are no resumes, the run ends in a clear fallback message instead of generating a letter.
- **Refinement:** the critique can trigger one editor pass.
- **Graceful degradation:** a failed Researcher, Critic or Editor does not fail the run; a failed Copywriter routes to a fallback explanation.

---

## Installation

### Prerequisites
- Python 3.11
- A Pinecone account (free tier works)
- At least one chat/embedding provider key (OpenAI, Gemini or Anthropic), or [Ollama](https://ollama.ai/) for local models
- A Tavily key for company research (free tier available)

### 1. Clone and install
```bash
git clone https://github.com/htet-khaing-win/job-application-rag-agent.git
cd job-application-rag-agent
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

### 2. Add API keys (no `.env` needed)
Keys are stored in an **encrypted local vault** (Fernet). The master key lives in your OS keychain, with a `0600` key file as fallback. Nothing is hard-coded or kept in `.env`.

```bash
streamlit run app.py          # GUI: "API Keys" tab
# or headless:
python main.py add-key        # prompted, input hidden
python main.py list-keys      # masked output
python main.py remove-key NAME
```

For CI or headless use only, set `JOBAGENT_ALLOW_ENV_KEYS=1` to let the vault fall back to `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `TAVILY_API_KEY` and `PINECONE_API_KEY`. Local data lives in `~/.job_application_agent/` (override with `JOBAGENT_HOME`).

### 3. Optional: local models
```bash
ollama pull qwen2.5:7b
ollama pull mistral:7b-instruct
ollama pull nomic-embed-text
```
Note: the default Pinecone index is 1536-dimensional (OpenAI `text-embedding-3-small`). `nomic-embed-text` is 768-dimensional, so switching embeddings requires a new index and re-ingesting resumes.

### 4. Fit the BM25 encoder (recommended)
```bash
python utilities/Fitting.py
```
This writes `utilities/fitted_bm25.json`. Without it, a generic (cold) BM25 encoder is used with reduced keyword accuracy. Run it after ingesting resumes.

### 5. Ingest resumes
In the GUI's **Resumes** tab, or:
```bash
python database.py /path/to/resume.pdf
```
Supported: PDF, DOCX, TXT; max 5 MB. PII redaction, section detection and chunking are applied automatically.

---

## Usage

### GUI
```bash
streamlit run app.py
```
Tabs: **Generate**, **Resumes**, **Agents & Models**, **API Keys**, **Security**. The Generate tab shows the letter, retrieval score, refinement count, verified and unverified skills, per-agent timing, and a log of every recovery action.

### CLI
```bash
python main.py
```
Paste the job description, type `DONE` on its own line, then enter the company name. The letter is printed, followed by any self-healing events.

### Managing resumes from Python
```python
from runtime import Runtime, Profile, activate_runtime
from vault import KeyVault
from database import list_stored_resumes, delete_resume_from_pinecone

activate_runtime(Runtime(KeyVault(), Profile.load()))
print(list_stored_resumes())
delete_resume_from_pinecone("resume.pdf")
```

---

## Customising Agents and Keys

Each agent has a provider, model and key alias, set in the **Agents & Models** tab and saved in `~/.job_application_agent/profile.json` (no secrets). The second model in each chat role is its automatic failover.

| Agent | Role |
|---|---|
| JD Analyst | validates and distils the job description |
| Researcher | live company research (Tavily) |
| Retriever | hybrid search over your resumes (embeddings + Pinecone) |
| Copywriter | drafts the letter |
| Fact Checker | repairs letters that claim unsupported skills |
| Hiring Manager | critiques the draft |
| Editor | applies the critique's edits |

Presets: **Balanced** (OpenAI + Gemini failover), **Cheap** (Gemini Flash), **Best quality** (Claude + GPT), **Fully local** (Ollama). Each agent has a **Test connection** button that probes its primary model and reports a redacted error if it fails.

---

## Self-Healing

Retry with backoff, then provider failover, behind a circuit breaker. Malformed JSON from a model is sent back for repair. When a model is unavailable the pipeline degrades gracefully: a critic outage ships the verified draft, and a writer outage ends in a clear fallback message. Every recovery action appears in the GUI log and in `result["healing_log"]`.

---

## Security and Privacy

- **Keys:** encrypted vault; never in `.env`, logs, the profile JSON or error messages; masked in the UI.
- **Redaction:** known key shapes (OpenAI/Anthropic, Google, Tavily, Pinecone) and `key=value` pairs are scrubbed from logs and UI errors.
- **Prompt injection:** the job description and scraped web content are fenced as untrusted data, with closing-tag spoofing neutralised, control and bidi characters stripped, and length capped. Suspected injection is flagged in the log.
- **Uploads:** extension, magic bytes, size and encoding checks; filenames sanitised against path traversal.
- **Network:** the GUI binds to `127.0.0.1` with XSRF protection on and telemetry off. Do not expose it to a network without an authenticating reverse proxy.
- **PII:** redacted before embedding and storage.

These protections are covered by unit tests using fake models. Behaviour against real provider APIs has not been independently verified.

---

## Evaluation

`python -m evals` computes recall@k, precision@k, MRR and nDCG with bootstrap 95% confidence intervals and a paired permutation test, plus a deterministic letter-faithfulness score.

```bash
# 1. Copy evals/gold_set.example.json to evals/gold_set.json and label 30-50 queries
# 2. Collect ranked lists (live; needs vault keys)
python -m evals --gold evals/gold_set.json --collect baseline --out baseline.json
python -m evals --gold evals/gold_set.json --collect hybrid   --out hybrid.json
# 3. Compare
python -m evals --gold evals/gold_set.json --runs-a baseline.json --runs-b hybrid.json
# Faithfulness of generated letters
python -m evals --letters letters.json
```

Reports are written to `evals/reports/` with the git SHA. Labelling the gold set is manual work and the harness is only as good as those labels; with fewer than 30 queries, expect wide confidence intervals.

---

## Developer Tooling

- `make test | lint | type | eval | check-staged | run | docker`
- `pyproject.toml` configures ruff, mypy and pytest.
- `.pre-commit-config.yaml` runs ruff, gitleaks and private-key detection.
- A non-root `Dockerfile` for the GUI. It has not been build-tested in this repository's CI.
- `make check-staged` refuses to stage resumes, vault files, profiles or `.env` files.

Tests use fake models and a temporary vault, so no provider keys are needed:
```bash
pip install -r requirements.txt
pytest
```
`tests/pii_test.py` needs Presidio and the spaCy transformer model; it is skipped when they are not installed.

---

## Known Limitations

- **Retrieval covers only the first two stored resumes.** Namespaces beyond the second are never queried.
- **Hybrid weighting is not applied as intended.** The retrieval call passes an `alpha` argument that Pinecone's `query()` does not define; the dense/sparse mix should instead be applied by scaling the query vectors.
- **One refinement pass.** The critique/edit loop runs at most once.
- **No CI workflow is committed to this branch.**
- **Live provider behaviour is untested.** The pipeline is exercised with fake models; real-provider runs are your responsibility to verify.

The branch `feature/hardening-evals-ci` addresses several of these (all-resume retrieval, vector-scaled hybrid weighting, a CI workflow, an expanded evaluation CLI, vault passphrase protection and per-key spend caps). It is awaiting review and is not part of `main`.

---

## Project Structure

```
job-application-rag-agent/
├── main.py            # CLI entry point (+ add-key / list-keys / remove-key)
├── app.py             # Streamlit GUI
├── pipeline.py        # validated entry point used by CLI and GUI
├── graph.py           # LangGraph workflow and routing
├── subagents.py       # sub-agents and the self-healing Supervisor
├── node.py            # node logic (JD ingest, research, write, critique, refine)
├── database.py        # Pinecone ingestion and hybrid retrieval
├── runtime.py         # per-agent profile, presets, connection tests
├── providers.py       # provider registry and lazy client factories
├── resilience.py      # retry, failover, circuit breaker, JSON repair
├── vault.py           # encrypted API-key vault
├── security.py        # redaction, sanitising, upload validation
├── privacy.py         # Presidio PII configuration
├── export.py          # .docx export
├── state.py           # Pydantic state schema
├── evals/             # retrieval metrics, faithfulness, eval runner
├── utilities/         # BM25 fitting + fitted encoder
├── tests/             # unit and fault-injection tests (fake models)
├── Makefile  Dockerfile  pyproject.toml  .pre-commit-config.yaml
└── requirements.txt
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
