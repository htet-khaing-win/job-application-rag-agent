.PHONY: test lint type eval check-staged run docker
test:
	python -m pytest
lint:
	ruff check .
type:
	mypy
eval:
	python -m evals --gold evals/gold_set.json --runs-a baseline.json --runs-b hybrid.json
# refuse to commit personal data or secrets
check-staged:
	@! git diff --cached --name-only | grep -E '(^resumes/|vault\.enc|master\.key|profile\.json|(^|/)\.env)' \
	  && echo "staged files OK" || (echo "BLOCKED: staged file looks sensitive" && exit 1)
run:
	streamlit run app.py
docker:
	docker build -t job-agent .
	docker run --rm -p 127.0.0.1:8501:8501 -v job-agent-data:/home/app/.job_application_agent job-agent
