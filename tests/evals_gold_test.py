import pytest

from evals import goldgen, mcf
from evals.spans import evaluate_run_spans, query_metrics, span_in_chunk
from evals.taxonomy import TECH, extract_skills, normalise_pdf_text


# ---------------------------------------------------------------- taxonomy precision
@pytest.mark.parametrize("text,absent", [
    ("Built scalable services", "scala"),
    ("Built a ReAct-based autonomous agent", "react"),
    ("Integrated the OpenAI API for summaries", "rest api"),
    ("Excellent communicator", "excel"),
    ("We use Mistral internally", "java"),
    ("Experience with JavaScript", "java"),
])
def test_known_false_positives_stay_fixed(text, absent):
    assert absent not in extract_skills(text)


@pytest.mark.parametrize("text,present", [
    ("Worked with Gemini and GPT-4", "llm"),
    ("PyTorch, TensorFlow and Keras", "pytorch"),
    ("RAG pipelines on Pinecone", "rag"),
    ("Pinecone vector store", "vector database"),
    ("Deployed with Docker and Kubernetes", "kubernetes"),
    ("Spark and Databricks jobs", "spark"),
    ("PostgreSQL and MySQL", "sql"),
    ("CUDA-accelerated training", "gpu"),
    ("OpenCV based image processing", "computer vision"),
])
def test_core_skills_are_detected(text, present):
    assert present in extract_skills(text)


def test_soft_skills_are_not_technical():
    assert "leadership" not in TECH and "python" in TECH


def test_letter_spaced_pdf_text_is_repaired_but_normal_text_untouched():
    spaced = "M L O p s \t & \t D e p l o y m e n t :  D o c k e r ,  F a s t A P I"
    fixed = normalise_pdf_text(spaced)
    assert "Docker" in fixed and "FastAPI" in fixed and "MLOps" in fixed
    normal = "Built a RAG system with LangChain and Pinecone."
    assert normalise_pdf_text(normal) == normal


# ---------------------------------------------------------------- spans
def test_span_matching_survives_whitespace_and_redaction():
    span = "Engineered an end-to-end ML pipeline leveraging MobileNetV2 and VGG16 architectures"
    chunk = "[CANDIDATE_NAME]   Engineered an end-to-end\nML pipeline  leveraging MobileNetV2 and VGG16 architectures. Deployed app"
    assert span_in_chunk(span, chunk)
    assert not span_in_chunk(span, "Managed stakeholders through UAT and sprint demos in Jira")
    assert not span_in_chunk("too short", "too short indeed")      # < 3 tokens never matches


def test_query_metrics_known_values():
    spans = ["alpha beta gamma delta epsilon", "one two three four five six"]
    chunks = [{"text": "unrelated words only here"}, {"text": "alpha beta gamma delta epsilon zeta"}]
    m = query_metrics(chunks, spans, 2)
    assert m["recall@2"] == 0.5 and m["precision@2"] == 0.5 and m["mrr"] == 0.5
    assert 0 < m["ndcg@2"] < 1
    assert query_metrics(chunks, spans, 1)["recall@1"] == 0.0


def test_evaluate_run_spans_skips_negatives_and_separates_good_from_bad_runs():
    queries = [
        {"id": "a", "relevant_spans": ["alpha beta gamma delta epsilon"], "bucket": "high"},
        {"id": "b", "relevant_spans": ["one two three four five six"], "bucket": "low"},
        {"id": "neg", "relevant_spans": []},
    ]
    good = {"a": [{"id": "1", "text": "alpha beta gamma delta epsilon"}],
            "b": [{"id": "2", "text": "one two three four five six seven"}]}
    bad = {"a": [{"id": "9", "text": "nothing here"}], "b": [{"id": "8", "text": "nothing"}]}
    g, b = evaluate_run_spans(good, queries, 3), evaluate_run_spans(bad, queries, 3)
    assert len(g["mrr"]) == 2                                   # the negative is not scored
    assert g["mrr"] == [1.0, 1.0] and b["mrr"] == [0.0, 0.0]
    assert evaluate_run_spans({}, queries, 3)["recall@3"] == [0.0, 0.0]   # missing run = zero, not a crash


# ---------------------------------------------------------------- gold builder
RESUME_ML = ("Skills: Python, PyTorch, TensorFlow, Docker, FastAPI, SQL (PostgreSQL)\n"
             "Built a CNN image classifier with Keras and OpenCV deployed with Docker\n"
             "Contact: someone@example.com | +65 8123 4567 | linkedin.com/in/someone\n"
             "Led a team of three to run Jira sprints and stakeholder demos\n")
RESUME_PM = "Managed Agile sprints with Jira and prepared Excel reports for stakeholders every week\n"
LISTING = {"uuid": "u1", "title": "Machine Learning Engineer", "company": "Acme", "url": "https://example/jobs/u1",
           "description": "Requirements: Python, PyTorch, Kubernetes and Spark experience. Build RAG with LangChain. "
                          "Deploy with Docker on AWS.", "listed_skills": ["Python", "Kubernetes"], "key_skills": ["Python"]}
THIN = {"uuid": "u2", "title": "AI Engineer", "company": "Thin Co", "url": "https://example/jobs/u2",
        "description": "Exciting role, great culture, Python only.", "listed_skills": [], "key_skills": []}


def _profiles():
    return {"ml.txt": goldgen.resume_profile(RESUME_ML), "pm.txt": goldgen.resume_profile(RESUME_PM)}


def test_resume_profile_excludes_pii_lines_from_anchors():
    prof = goldgen.resume_profile(RESUME_ML)
    assert "pytorch" in prof["skills"] and "docker" in prof["skills"]
    assert all("@" not in a and "linkedin" not in a.lower() for a in prof["anchors"].values())


def test_gold_labels_supported_unsupported_and_negatives():
    gold = goldgen.build([LISTING, THIN], _profiles())
    qs = {q["namespace"]: q for q in gold["queries"]}
    assert len(gold["queries"]) == 2                               # thin listing skipped, 1 listing x 2 resumes
    assert gold["meta"]["skipped_listings"][0]["uuid"] == "u2"
    ml, pm = qs["ml.txt"], qs["pm.txt"]
    assert {"python", "pytorch", "docker"} <= set(ml["supported_skills"])
    assert {"kubernetes", "spark", "rag", "langchain"} <= set(ml["unsupported_skills"])
    assert ml["relevant_spans"] and not ml["expect_fallback"]
    assert pm["expect_fallback"] and pm["relevant_spans"] == [] and pm["bucket"] == "low"
    assert gold["meta"]["negatives"] == 1 and gold["meta"]["scorable_queries"] == 1
    assert "SILVER" in gold["meta"]["labeling"]
    assert set(ml["supported_skills"]).isdisjoint(ml["unsupported_skills"])


def test_skill_on_a_short_line_still_counts_as_supported():
    resume = "Skills: Python, PyTorch, TensorFlow, Docker, FastAPI and SQL for model work\nNER patterns\n"
    prof = goldgen.resume_profile(resume)
    assert "nlp" in prof["skills"] and "nlp" not in prof["anchors"]      # supported, but no quotable line
    job = dict(LISTING, description="Requirements: Python, NLP, PyTorch and Docker. Build RAG with LangChain.")
    q = goldgen.build([job], {"r.txt": prof})["queries"][0]
    assert "nlp" in q["supported_skills"] and q["relevant_spans"]


def test_suggested_tags_never_add_requirements_on_their_own():
    job = dict(LISTING, listed_skills=["Python", "Kubernetes", "Terraform", "Toothbrushing"])
    req = goldgen.required_skills(job)
    assert "terraform" not in req["skills"]                      # tag only, absent from the posting text
    assert "terraform" in req["tag_only_ignored"] and "kubernetes" in req["from_tags"]


def test_non_ai_listings_are_skipped_even_with_many_tech_skills():
    fullstack = {"uuid": "u3", "title": "AI Engineer", "company": "Web Co", "url": "https://example/jobs/u3",
                 "description": "Build services in Java, Docker, PostgreSQL, Kubernetes and React on AWS with CI/CD.",
                 "listed_skills": [], "key_skills": []}
    gold = goldgen.build([fullstack], _profiles())
    assert gold["queries"] == [] and "not clearly AI/ML" in gold["meta"]["skipped_listings"][0]["reason"]


def test_gold_spans_actually_match_the_resume_text_they_came_from():
    gold = goldgen.build([LISTING], _profiles())
    q = next(q for q in gold["queries"] if q["namespace"] == "ml.txt")
    chunk = {"text": RESUME_ML}
    assert query_metrics([chunk], q["relevant_spans"], 1)["recall@1"] == 1.0


def test_audit_report_renders():
    gold = goldgen.build([LISTING], _profiles())
    out = goldgen.audit(gold, 2)
    assert "required" in out and "supported" in out


# ---------------------------------------------------------------- MCF parsing (no network)
def test_html_to_text_and_normalise():
    raw = {"uuid": "x", "title": "ML Engineer", "metadata": {"jobPostId": "MCF-1", "jobDetailsUrl": "https://m/x"},
           "postedCompany": {"name": "Acme"}, "minimumYearsExperience": 3,
           "description": "<p>Build <b>models</b></p><ul><li>Python</li><li>PyTorch</li></ul>&amp; more",
           "skills": [{"skill": "Python", "isKeySkill": True}, {"skill": "SQL", "isKeySkill": False}],
           "positionLevels": [{"position": "Senior"}]}
    rec = mcf.normalise(raw)
    assert "Build models" in rec["description"] and "- Python" in rec["description"] and "& more" in rec["description"]
    assert rec["listed_skills"] == ["Python", "SQL"] and rec["key_skills"] == ["Python"]
    assert rec["company"] == "Acme" and rec["url"] == "https://m/x"


@pytest.mark.parametrize("title,ok", [
    ("Machine Learning Engineer", True), ("Senior AI Engineer (LLM)", True), ("Data Scientist", True),
    ("MLOps Engineer", True), ("AI Intern", False), ("Sales Engineer", False),
    ("Marketing Manager", False), ("Warehouse Operator", False),
])
def test_title_filter(title, ok):
    rec = {"title": title, "description": "x" * 500}
    assert mcf.wanted(rec) is ok


def test_short_descriptions_are_rejected():
    assert not mcf.wanted({"title": "Machine Learning Engineer", "description": "short"})
