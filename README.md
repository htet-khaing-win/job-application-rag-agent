# Job Application RAG Agent

A self-healing, multi-agent RAG system that writes tailored cover letters by matching your resumes to a job description. It uses hybrid vector search, per-agent model selection, a Streamlit GUI, and an encrypted local key vault. Letters are fact-checked against your resume before you see them.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [GUI](#gui)
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
- Dense embeddings plus BM25 sparse vectors, with the mix applied by scaling the query vectors (`alpha` is configurable). Section-aware chunking (Experience, Education, Skills, ...), PDF/DOCX/TXT ingestion. Every stored resume is searched, up to a configurable cap.

**Specialised sub-agents**
- JD Analyst, Researcher, Retriever, Analyst, Copywriter, Fact Checker, Hiring Manager and Editor, orchestrated as a LangGraph workflow. Each model-backed agent can use its own provider, model and API key.

**Anti-hallucination**
- Skills are checked against retrieved resume evidence. The Copywriter may only mention verified skills, or unverified ones framed as "prepared to learn".
- The Fact Checker finds sentences that claim an unverified skill and has a model rewrite them. If the rewrite doesn't fix it, the sentences are removed, so an ungrounded claim cannot ship.

**Cost and key control**
- Pre-run cost estimate, per-run cost and token tracking, per-key monthly spend caps, key age and rotation reminders, and an optional vault passphrase with a lock button.

**Evaluation built in**
- Retrieval metrics with confidence intervals, claim-level faithfulness, fault-injection and ablation harnesses, and a gold-set builder that uses real public job listings.

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
(hybrid search)         (Tavily, cached)
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
- **Caching:** query embeddings and company research (24h) are cached by content hash. Research results are never cached when empty or failed.

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
Keys are stored in an **encrypted local vault** (Fernet). The master key lives in your OS keychain, with a `0600` key file as fallback; you can instead protect the vault with a passphrase. Nothing is hard-coded or kept in `.env`.

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

## GUI

```bash
streamlit run app.py
```
It binds to `127.0.0.1` only.

- **Generate:** live view of the writer's draft as it streams in, per-agent progress, a pre-run cost/token estimate, spend-this-month vs per-key caps (the run is blocked if a cap would be exceeded), a run timeout, an editable draft with **Re-check grounding** (flags sentences that claim skills your resume doesn't support), a skills-gap table with the evidence line for each verified skill, `.txt` / `.docx` / `.pdf` export, and recent-letter history with a diff against the previous letter for the same company. Streamlit's own Stop button cancels a run.
- **Resumes:** upload, list and delete resumes.
- **Agents & Models:** presets, per-agent provider/model/key, a **Test connection** button per agent (and **Test all**) with status lights, and a side-by-side profile comparison that lists only the differences.
- **API Keys / Security:** key age, last use and rotation reminders, per-key monthly spend caps, an optional **vault passphrase** (lock/unlock; while locked the app shows only the unlock form), and editable retrieval settings (`alpha`, max resumes searched, top-k) saved in your profile.
- **Runs:** run ids, durations, cost and p50/p95 per agent from the local run log. The log stores timings, token counts and cost only, never letter or resume text.
- **Chaos mode** (sidebar): a demo that injects random provider faults so you can watch the self-healing log recover. Leave it off for real use.

Tables are rendered as escaped markdown, not `st.dataframe`, so resume text can't inject markup and the GUI doesn't depend on pandas.

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
| Judge | pairwise letter judge used only by evaluations (use a different model from the Copywriter) |

Presets: **Balanced** (OpenAI + Gemini failover), **Cheap** (Gemini Flash), **Best quality** (Claude + GPT), **Fully local** (Ollama).

---

## Self-Healing

Retry with backoff, then provider failover, behind a circuit breaker. Malformed JSON from a model is sent back for repair. When a model is unavailable the pipeline degrades gracefully: a critic outage ships the verified draft, and a writer outage ends in a clear fallback message. A key over its monthly cap fails over to the next model instead of failing. Every recovery action appears in the GUI log and in `result["healing_log"]`.

---

## Security and Privacy

- **Keys:** encrypted vault; never in `.env`, logs, the profile JSON or error messages; masked in the UI. Passphrase mode derives the key with scrypt, and enabling or disabling it is crash-safe: the vault and its metadata are committed together through a journal, and start-up recovery completes or rolls back an interrupted change.
- **Redaction:** known key shapes (OpenAI/Anthropic, Google, Tavily, Pinecone), `key=value` pairs and opaque `Bearer` tokens are scrubbed from logs and UI errors.
- **Prompt injection:** the job description and scraped web content are fenced as untrusted data, with closing-tag spoofing neutralised, control and bidi characters stripped, and length capped. Suspected injection is flagged in the log.
- **Uploads:** extension, magic bytes, size and encoding checks; filenames sanitised against path traversal. `.docx` files must be structurally valid and are rejected when they expand beyond 50 MB, exceed 2000 parts, contain unsafe paths, or have a compression ratio above 100:1.
- **Network:** the GUI binds to `127.0.0.1` with XSRF protection on and telemetry off. Do not expose it to a network without an authenticating reverse proxy.
- **PII:** redacted before embedding and storage.

These protections are covered by unit tests, a 55-payload injection corpus and fuzzing, all using fake models. Behaviour against real provider APIs has not been independently verified.

---

## Evaluation

`python -m evals <command>` (see `--help`):

| Command | What it does |
|---|---|
| `retrieval` | recall@k / precision@k / MRR / nDCG with bootstrap CIs, paired permutation and Wilcoxon tests, Holm correction and effect sizes |
| `letters` | claim-level faithfulness of generated letters (lexical, LLM or model entailment) |
| `chaos` | fault-injection run of the self-healing graph against fake models (no keys needed) |
| `ablate`, `sweep` | switch pipeline components off one at a time; hybrid `alpha` sweep with cross-validated selection (**live**: need your keys) |
| `label` | hand-labelling sheets and gold-set building |
| `perf` | p50/p95 latency, tokens and cost per agent from the run log |
| `regress` | exits non-zero when tracked metrics regress against a stored baseline |

Reports are written to `evals/reports/` with the git SHA, model ids, seeds and cost.

### Building a gold set from real listings
The repository can build a gold set from public AI/ML engineering listings on MyCareersFuture (Singapore's government job portal) and your own resumes:

```bash
# 1. Read each listing page (browser or saved page text), then ingest it through the safety layer:
python -m evals.mcf_pages ingest --url <listing url> --file page.txt
python -m evals.mcf_pages collect                       # -> evals/gold_src/listings.json
python -m evals.mcf_pages audit                         # safety report
# (fallback) rate-limited public API:  python -m evals.mcf fetch --target 50

# 2. Build and inspect the gold set
python -m evals.goldgen build --resumes resumes         # -> evals/gold_set.json
python -m evals.goldgen audit -n 10                     # spot-check labels by eye

# 3. Compare retrieval methods (live: needs your keys and an ingested index)
python -m evals retrieval --gold evals/gold_set.json --collect baseline --out baseline.json
python -m evals retrieval --gold evals/gold_set.json --collect hybrid   --out hybrid.json
python -m evals retrieval --gold evals/gold_set.json --runs-a baseline.json --runs-b hybrid.json
```

How it is labelled (a **silver standard**, not human-adjudicated): required skills come from the listing text, matched against a skill taxonomy (`evals/taxonomy.py`); the site's auto-suggested skill tags only corroborate and never add a requirement. A skill is *supported* if the resume text matches the same taxonomy, and the *relevant spans* are the resume lines that evidence the supported skills. Retrieval is scored by matching retrieved chunk text against those spans, so labels survive re-chunking and don't depend on Pinecone chunk ids. Skills merely *listed* on a resume count as supported, and taxonomy gaps cause missed or spurious skills, so audit a sample and treat results as indicative until a human has reviewed them.

Listing text is untrusted third-party data. `evals/listing_safety.py` only allows genuine `mycareersfuture.gov.sg/job/...` pages, strips links, emails, phone numbers, agency licence ids and contact lines, removes active HTML, and flags shorteners, raw-IP or executable links and text addressed to an AI. Listing text and the gold set are gitignored: keep them local and don't redistribute them.

---

## Developer Tooling

- **Config:** `pyproject.toml` configures ruff (correctness and security rules), mypy and pytest. Pytest markers: `heavy` (needs Presidio and the spaCy model) and `live` (calls real providers; never run in CI).
- **Make:** `make test | test-fast | lint | type | audit | eval | chaos | perf | check-staged | lock | run | docker`. `make check-staged` refuses to stage resumes, vault files, profiles or `.env` files.
- **Dependencies:** `requirements.txt` (runtime) and `requirements-dev.txt` (test, lint, audit). `scripts/lock.sh` generates `requirements.lock` with sha256 hashes via pip-tools; the Dockerfile installs from it with `--require-hashes` when it exists. The lock file is not checked in until you run the script (it needs network access).
- **CI** (`.github/workflows/test.yml`): fast tests, a separate `heavy` job (allowed to fail while the spaCy download proves stable), and a security job (`pip-audit`, ruff, mypy, gitleaks). Read-only permissions, no provider keys or repository secrets. Dependabot covers pip, Actions and Docker.
- **Pre-commit:** `.pre-commit-config.yaml` (ruff, gitleaks, private-key detection, large-file check). Run `pre-commit install` once.
- **Docker:** non-root image; `make docker` publishes the GUI on `127.0.0.1` only.

Tests use fake models and a temporary vault, so no provider keys are needed:
```bash
pip install -r requirements.txt
pytest
```
`tests/pii_test.py` needs Presidio and the spaCy transformer model; it is marked `heavy` and skipped when they are not installed.

---

## Known Limitations

- **Live provider behaviour is unverified.** The pipeline is exercised with fake models; run it with your own keys before relying on it. The Docker image and the CI workflow's first run are also unverified.
- **The gold set is a silver standard** (rule-based labels from a skill taxonomy), and your resumes share most of their content, so there are few distinct evidence spans. Treat retrieval comparisons as indicative.
- **Retrieval scores changed.** Applying `alpha` by scaling the query vectors changes the score magnitudes relative to earlier runs, so old scores and thresholds are not directly comparable.
- **One refinement pass.** The critique/edit loop runs at most once.
- **Timeouts cancel at the next await.** A synchronous provider call blocked inside an SDK is bounded by that SDK's own timeout.

---

## Project Structure

```
job-application-rag-agent/
├── main.py            # CLI entry point (+ add-key / list-keys / remove-key)
├── app.py             # Streamlit GUI
├── pipeline.py        # validated entry point: timeout, budget check, run log
├── graph.py           # LangGraph workflow and routing
├── subagents.py       # sub-agents and the self-healing Supervisor
├── node.py            # node logic (JD ingest, research, write, critique, refine)
├── database.py        # Pinecone ingestion and hybrid retrieval
├── runtime.py         # per-agent profile, presets, settings, connection tests
├── providers.py       # provider registry and lazy client factories
├── resilience.py      # retry, failover, circuit breaker, JSON repair, streaming sink
├── vault.py           # encrypted API-key vault (+ passphrase, crash-safe)
├── cost.py            # price estimates, run-cost estimate, spend ledger
├── cache.py  runlog.py# content-hash caches; privacy-preserving run log
├── security.py        # redaction, sanitising, upload validation
├── privacy.py         # Presidio PII configuration
├── export.py          # .docx and .pdf export
├── state.py           # Pydantic state schema
├── evals/             # metrics, faithfulness, judge, chaos, ablation, sweep,
│                      #   gold-set tooling (mcf_pages, listing_safety, goldgen, spans)
├── utilities/         # BM25 fitting + fitted encoder
├── scripts/           # lock.sh
├── tests/             # unit, fault-injection, security-corpus and eval tests
├── .github/           # CI workflow, Dependabot
├── Makefile  Dockerfile  pyproject.toml  .pre-commit-config.yaml
└── requirements.txt  requirements-dev.txt
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
