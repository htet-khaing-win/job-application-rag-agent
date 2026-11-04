"""Canonical skill taxonomy for AI/ML engineering listings and resumes.

Each canonical skill has regex aliases. Matching is word-bounded; short or ambiguous tokens
(R, Go, C, ML, AI) are case-sensitive. The same taxonomy labels both sides of a pair, so a skill
is "supported" exactly when the resume text matches one of its aliases.
"""
from __future__ import annotations

import re
from typing import Dict, FrozenSet, List, Pattern, Sequence, Tuple

# canonical -> (case_sensitive, [alias regexes])
_RAW: Dict[str, Tuple[bool, Sequence[str]]] = {
    # languages
    "python": (False, [r"python"]),
    "java": (True, [r"Java(?!\s?[Ss]cript)"]),
    "javascript": (False, [r"java\s?script", r"node\.?js", r"typescript"]),
    "c++": (False, [r"c\+\+"]),
    "c#": (False, [r"c#", r"\.net"]),
    "scala": (False, [r"\bscala\b"]),
    "go": (True, [r"Golang", r"\bGo\b(?= (?:language|programming|services|backend))"]),
    "r": (True, [r"\bR\b(?= (?:programming|language|and Python|studio))", r"RStudio"]),
    "sql": (False, [r"\bsql\b", r"postgres(?:ql)?", r"mysql", r"sqlite", r"nosql", r"mongodb", r"oracle db"]),
    "bash": (False, [r"\bbash\b", r"shell script\w*"]),
    # ml / dl core
    "machine learning": (True, [r"[Mm]achine [Ll]earning", r"\bML\b"]),
    "deep learning": (False, [r"deep learning", r"neural networks?", r"\bcnns?\b", r"\brnns?\b", r"\blstms?\b"]),
    "pytorch": (False, [r"pytorch"]),
    "tensorflow": (False, [r"tensorflow"]),
    "keras": (False, [r"keras"]),
    "scikit-learn": (False, [r"scikit[- ]?learn", r"sklearn"]),
    "xgboost": (False, [r"xgboost", r"lightgbm", r"catboost", r"gradient boost\w*"]),
    "transformers": (False, [r"transformers?\b(?! to)", r"hugging ?face", r"\bbert\b", r"attention mechanism"]),
    "transfer learning": (False, [r"transfer learning", r"fine[- ]?tun\w+"]),
    "hyperparameter tuning": (False, [r"hyper-?parameter\w*", r"optuna", r"grid search"]),
    "reinforcement learning": (False, [r"reinforcement learning", r"\brlhf\b"]),
    "time series": (False, [r"time[- ]series", r"forecasting"]),
    "recommender systems": (False, [r"recommend(?:er|ation)s?(?: systems?| engines?)?"]),
    "statistics": (False, [r"statistic\w*", r"hypothesis test\w*", r"a/b test\w*", r"bayesian"]),
    "feature engineering": (False, [r"feature engineering", r"feature stores?"]),
    "model evaluation": (False, [r"model evaluation", r"model validation", r"validation accuracy", r"evaluation metrics"]),
    "optimization": (False, [r"model optimi[sz]ation", r"quantization", r"onnx", r"tensorrt", r"distillation", r"pruning"]),
    # genai
    "llm": (True, [r"\bLLMs?\b", r"(?i:large language models?)", r"(?i:\bgpt-?\d\w*\b)", r"(?i:\bllama ?\d?\b)", r"(?i:\bclaude\b)", r"(?i:\bgemini\b)", r"(?i:\bmistral\b)"]),
    "generative ai": (False, [r"generative ai", r"\bgenai\b", r"gen ai", r"diffusion models?", r"stable diffusion"]),
    "rag": (True, [r"\bRAG\b", r"(?i:retrieval[- ]augmented)", r"(?i:vector (?:search|database|store)s?)"]),
    "prompt engineering": (False, [r"prompt engineering", r"prompt design", r"prompt\w* (?:templates?|optimi[sz]ation)"]),
    "langchain": (False, [r"langchain", r"langgraph", r"llamaindex", r"llama[- ]index"]),
    "agents": (False, [r"ai agents?", r"agentic", r"multi-?agent", r"react agents?", r"autonomous agents?", r"tool[- ]use", r"function calling"]),
    "vector database": (False, [r"pinecone", r"faiss", r"weaviate", r"milvus", r"qdrant", r"chroma(?:db)?", r"pgvector", r"vector (?:database|db)s?"]),
    "openai api": (False, [r"openai", r"azure openai", r"gemini api", r"anthropic", r"bedrock", r"vertex ai"]),
    "embeddings": (False, [r"embeddings?", r"sentence[- ]transformers?"]),
    "nlp": (True, [r"\bNLP\b", r"(?i:natural language processing)", r"(?i:text classification)", r"(?i:named entity)", r"\bNER\b", r"(?i:sentiment analysis)", r"(?i:text mining)"]),
    "computer vision": (False, [r"computer vision", r"image (?:recognition|classification|processing|segmentation)", r"object detection", r"opencv", r"\byolo\w*", r"facial (?:emotion )?recognition", r"image-based"]),
    "speech": (False, [r"speech recognition", r"\basr\b", r"text[- ]to[- ]speech", r"whisper"]),
    # data / mlops
    "pandas": (False, [r"pandas"]),
    "numpy": (False, [r"numpy"]),
    "spark": (False, [r"pyspark", r"apache spark", r"\bspark\b(?! plug)", r"databricks"]),
    "hadoop": (False, [r"hadoop", r"\bhive\b", r"\bhdfs\b"]),
    "kafka": (False, [r"kafka", r"kinesis", r"pub/?sub", r"rabbitmq"]),
    "airflow": (False, [r"airflow", r"prefect", r"dagster", r"luigi"]),
    "etl": (True, [r"\bETL\b", r"\bELT\b", r"(?i:data pipelines?)", r"(?i:data engineering)", r"(?i:data (?:warehouse|lake)s?)"]),
    "mlops": (False, [r"mlops", r"mlflow", r"kubeflow", r"model (?:serving|deployment|monitoring|registry)", r"ml pipelines?", r"sagemaker", r"weights & biases", r"\bwandb\b"]),
    "docker": (False, [r"docker", r"containeri[sz]\w+"]),
    "kubernetes": (False, [r"kubernetes", r"\bk8s\b", r"helm", r"\bgke\b", r"\beks\b", r"openshift"]),
    "ci/cd": (True, [r"CI/CD", r"CI-CD", r"(?i:jenkins)", r"(?i:github actions)", r"(?i:gitlab ci)", r"(?i:continuous (?:integration|delivery|deployment))"]),
    "git": (False, [r"\bgit\b", r"github", r"gitlab", r"bitbucket", r"version control"]),
    "linux": (False, [r"linux", r"unix"]),
    "aws": (False, [r"\baws\b", r"amazon web services", r"\bs3\b", r"\bec2\b", r"lambda functions?", r"sagemaker"]),
    "azure": (False, [r"azure", r"azure ml"]),
    "gcp": (False, [r"\bgcp\b", r"google cloud", r"bigquery", r"vertex ai"]),
    "cloud": (False, [r"cloud (?:platforms?|computing|infrastructure|services)", r"\bcloud\b"]),
    "terraform": (False, [r"terraform", r"infrastructure as code", r"ansible", r"cloudformation"]),
    "gpu": (True, [r"\bGPUs?\b", r"(?i:\bcuda\b)", r"(?i:distributed training)", r"(?i:multi-?gpu)", r"(?i:tensor cores?)"]),
    "parallel processing": (False, [r"multi-?processing", r"parallel (?:processing|computing)", r"multi-?thread\w*", r"concurren\w+"]),
    # apps / apis
    "fastapi": (False, [r"fastapi", r"fast api", r"flask", r"django"]),
    "rest api": (False, [r"rest(?:ful)? ?apis?", r"api (?:development|design|integration|management)s?", r"microservices?", r"grpc", r"graphql"]),
    "streamlit": (False, [r"streamlit", r"gradio", r"plotly dash"]),
    "react": (False, [r"react\.js", r"reactjs", r"react native", r"\bangular\b", r"\bvue(?:\.js)?\b", r"\bnext\.js\b"]),
    "databases": (False, [r"redis", r"elasticsearch", r"cassandra", r"dynamodb", r"neo4j", r"graph database"]),
    "testing": (False, [r"unit test\w*", r"pytest", r"test[- ]driven", r"integration tests?", r"\bqa\b"]),
    # analytics / bi
    "power bi": (False, [r"power ?bi", r"\bdax\b"]),
    "tableau": (False, [r"tableau", r"looker", r"qlik", r"superset"]),
    "excel": (False, [r"\bexcel\b"]),
    "data visualization": (False, [r"data visuali[sz]ation", r"matplotlib", r"seaborn", r"plotly", r"dashboards?"]),
    # practice
    "agile": (False, [r"\bagile\b", r"scrum", r"kanban", r"\bjira\b"]),
    "documentation": (False, [r"technical documentation", r"documentation"]),
    "security": (False, [r"security", r"privacy", r"\bpii\b", r"cyber ?security", r"data protection", r"gdpr"]),
    "research": (False, [r"research", r"publications?", r"\bpapers?\b", r"phd", r"patent\w*"]),
    "leadership": (False, [r"leadership", r"mentor\w*", r"led a \w+", r"team lead", r"stakeholder"]),
    "communication": (False, [r"communication skills?", r"cross-functional", r"collaborat\w+"]),
}

_COMPILED: Dict[str, List[Pattern[str]]] = {
    skill: [re.compile(p, 0 if cs else re.IGNORECASE) for p in pats] for skill, (cs, pats) in _RAW.items()
}

# MCF "skills" tags are free-form; map the common ones onto the taxonomy through the same aliases.
SKILLS: FrozenSet[str] = frozenset(_RAW)
# Soft / generic skills: recorded, but excluded from retrieval relevance (they match almost any resume).
SOFT: FrozenSet[str] = frozenset({"documentation", "research", "leadership", "communication", "agile",
                                  "security", "cloud", "testing", "git", "linux", "excel"})
TECH: FrozenSet[str] = SKILLS - SOFT


def extract_skills(text: str) -> List[str]:
    """Canonical skills whose aliases occur in `text` (sorted, stable)."""
    return sorted(s for s, pats in _COMPILED.items() if any(p.search(text or "") for p in pats))


def skills_in_line(line: str) -> List[str]:
    return extract_skills(line)


def normalise_pdf_text(raw: str) -> str:
    """Repair letter-spaced PDF extraction ('P y T o r c h' -> 'PyTorch'), keeping word gaps.

    Only lines that are clearly letter-spaced (most tokens are single characters) are touched;
    normally extracted text passes through unchanged.
    """
    out = []
    for line in raw.splitlines():
        tokens = line.split()
        if len(tokens) >= 4 and sum(len(t) == 1 for t in tokens) / len(tokens) > 0.5:
            groups = [g.replace(" ", "") for g in re.split(r"\s*	\s*|\s{2,}", line.strip()) if g]
            line = " ".join(groups)
        out.append(line.strip())
    return chr(10).join(ln for ln in out if ln)
