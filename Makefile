.PHONY: test test-fast lint type audit eval chaos perf check-staged lock run docker

test:
	python -m pytest

# everything except tests that need presidio + the spaCy model
test-fast:
	python -m pytest -m "not heavy and not live"

lint:
	ruff check .

type:
	mypy

# known-vulnerability scan of the pinned dependency set
audit:
	pip-audit -r requirements.txt

# needs evals/gold_set.json (see evals/gold_set.example.json) and your keys in the vault
eval:
	python -m evals retrieval --gold evals/gold_set.json --runs-a baseline.json --runs-b hybrid.json

# fault-injection harness (fake models) + prompt-injection / upload / redaction suites; no keys needed
chaos:
	python -m evals chaos
	python -m pytest tests/chaos_security_test.py tests/security_corpus_test.py

# p50/p95 latency, tokens and cost from the local run log
perf:
	python -m evals perf

# refuse to commit personal data or secrets
check-staged:
	@! git diff --cached --name-only | grep -E '(^resumes/|vault\.enc|master\.key|profile\.json|(^|/)\.env)' 	  && echo "staged files OK" || (echo "BLOCKED: staged file looks sensitive" && exit 1)

# regenerate requirements.lock with hashes (needs network)
lock:
	bash scripts/lock.sh

run:
	streamlit run app.py

docker:
	docker build -t job-agent .
	docker run --rm -p 127.0.0.1:8501:8501 -v job-agent-data:/home/app/.job_application_agent job-agent
