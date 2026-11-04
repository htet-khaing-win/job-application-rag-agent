"""Build a gold set from MyCareersFuture listings x local resumes.

  python -m evals.goldgen build --listings evals/gold_src/listings.json --resumes resumes \
         --out evals/gold_set.json

Labelling method (honest label: this is a SILVER standard, not human-adjudicated):
  * required skills = taxonomy skills found in the listing text (MyCareersFuture's auto-suggested skill tags
    only corroborate; they never add a requirement);
  * resume skills   = taxonomy skills found in the resume text;
  * a skill is "supported" if both sides have it, "unsupported" if only the listing has it;
  * relevant spans  = the resume lines that evidence the supported skills (PII lines excluded).
Retrieval is scored by matching retrieved chunk text against these spans (evals/spans.py), so the
labels do not depend on any particular Pinecone chunk id or chunking.

Outputs contain third-party listing text: evals/gold_set.json is gitignored. Do not commit or share it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from evals.taxonomy import TECH, extract_skills, normalise_pdf_text

_PII = re.compile(r"(@|https?://|www\.|linkedin|github\.com|\+\d[\d\s-]{6,}|\b\d{8,}\b)", re.I)
MIN_REQUIRED = 3
MIN_ML_SKILLS = 2          # listing must name at least this many core AI/ML skills to count as AI/ML engineering
CORE_ML = frozenset({"machine learning", "deep learning", "pytorch", "tensorflow", "keras", "scikit-learn", "llm",
                     "generative ai", "rag", "nlp", "computer vision", "transformers", "mlops", "agents",
                     "embeddings", "vector database", "reinforcement learning", "xgboost", "statistics",
                     "recommender systems", "time series", "speech", "transfer learning", "langchain"})
MAX_SPANS = 12
MAX_JD_CHARS = 6000


def read_resume_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        raw = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
        return normalise_pdf_text(raw)
    return path.read_text(encoding="utf-8")


def resume_profile(text: str) -> Dict[str, Any]:
    """{skills: [...], anchors: {skill: best evidencing line}} using only non-PII lines."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 25 and not _PII.search(ln)]
    anchors: Dict[str, str] = {}
    best: Dict[str, tuple] = {}
    for line in lines:
        line_skills = [k for k in extract_skills(line) if k in TECH]
        # prefer the line that evidences the most skills at once, then the shorter one
        score = (len(line_skills), -len(line))
        for s in line_skills:
            if s not in best or score > best[s]:
                best[s] = score
                anchors[s] = line
    # A skill counts as supported if it appears ANYWHERE in the resume text (including short lines and PII-adjacent
    # lines); spans can only come from lines that are long enough and free of contact details.
    skills = sorted(s for s in extract_skills(text) if s in TECH)
    return {"skills": skills, "anchors": {k: anchors[k] for k in sorted(anchors)}}


def required_skills(listing: Mapping[str, Any]) -> Dict[str, Any]:
    text = f"{listing.get('title', '')}\n{listing.get('description', '')}"
    from_text = {s for s in extract_skills(text) if s in TECH}
    # MyCareersFuture's "add these skills" tags are auto-suggested and are NOT always in the posting, so they
    # never add a requirement on their own; they only corroborate skills the posting text already names.
    from_tags: set = set()
    for tag in listing.get("listed_skills") or []:
        from_tags |= {s for s in extract_skills(tag) if s in TECH}
    key: set = set()
    for tag in listing.get("key_skills") or []:
        key |= {s for s in extract_skills(tag) if s in TECH}
    return {"skills": sorted(from_text), "from_tags": sorted(from_tags & from_text), "key": sorted(key & from_text),
            "tag_only_ignored": sorted(from_tags - from_text)}


def _bucket(cov: float) -> str:
    return "high" if cov >= 0.6 else "medium" if cov >= 0.3 else "low"


def build(listings: Sequence[Mapping[str, Any]], resumes: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    queries: List[Dict[str, Any]] = []
    skipped = []
    for i, job in enumerate(listings):
        req = required_skills(job)
        if len(req["skills"]) < MIN_REQUIRED:
            skipped.append({"uuid": job.get("uuid"), "title": job.get("title"), "reason": "too few technical skills"})
            continue
        if len(CORE_ML & set(req["skills"])) < MIN_ML_SKILLS:
            skipped.append({"uuid": job.get("uuid"), "title": job.get("title"),
                            "reason": f"not clearly AI/ML (fewer than {MIN_ML_SKILLS} core AI/ML skills named)"})
            continue
        jd = f"{job['title']}\n{job['description']}"[:MAX_JD_CHARS]
        for name, prof in resumes.items():
            supported = [s for s in req["skills"] if s in prof["skills"]]
            unsupported = [s for s in req["skills"] if s not in prof["skills"]]
            # spans: anchors for supported skills, best-covering lines first, de-duplicated
            lines: Dict[str, int] = {}
            for s in supported:
                line = prof["anchors"].get(s)
                if line is None:          # supported, but no clean line to quote as evidence
                    continue
                lines[line] = lines.get(line, 0) + 1
            spans = [ln for ln, _ in sorted(lines.items(), key=lambda kv: (-kv[1], len(kv[0])))][:MAX_SPANS]
            cov = len(supported) / len(req["skills"])
            queries.append({
                "id": f"{i:03d}_{Path(name).stem}",
                "jd": jd, "namespace": name,
                "source": {"title": job["title"], "company": job["company"], "url": job["url"],
                           "uuid": job["uuid"], "posted": job.get("posted")},
                "required_skills": req["skills"], "key_skills": req["key"],
                "supported_skills": supported, "unsupported_skills": unsupported,
                "coverage": round(cov, 3), "bucket": _bucket(cov),
                "relevant_spans": spans,
                "expect_fallback": not supported,
            })
    positives = [q for q in queries if q["relevant_spans"]]
    meta = {
        "schema_version": 2,
        "source": "MyCareersFuture (mycareersfuture.gov.sg) public listings x local resumes",
        "labeling": "automatic taxonomy match; SILVER standard, not human-adjudicated",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "listings_used": len({q["source"]["uuid"] for q in queries}), "queries": len(queries),
        "scorable_queries": len(positives), "negatives": len(queries) - len(positives),
        "buckets": {b: sum(q["bucket"] == b for q in queries) for b in ("high", "medium", "low")},
        "skipped_listings": skipped,
        "caveats": ["Skills only LISTED on a resume count as supported (no depth/recency judgement).",
                    "Taxonomy gaps cause missed or spurious skills; spot-check with `evals.goldgen audit`.",
                    "Listing text belongs to employers; keep this file local."],
    }
    return {"meta": meta, "resume_profiles": {n: {"skills": p["skills"]} for n, p in resumes.items()},
            "queries": queries}


def audit(gold: Mapping[str, Any], n: int = 8, seed: int = 0) -> str:
    import random
    rng = random.Random(seed)
    sample = rng.sample(gold["queries"], min(n, len(gold["queries"])))
    out = []
    for q in sample:
        out.append(f"{q['id']}  {q['source']['title']} @ {q['source']['company']}  [{q['bucket']} {q['coverage']}]")
        out.append(f"   required   : {', '.join(q['required_skills'])}")
        out.append(f"   supported  : {', '.join(q['supported_skills']) or '-'}")
        out.append(f"   unsupported: {', '.join(q['unsupported_skills']) or '-'}")
        for s in q["relevant_spans"][:2]:
            out.append(f"   span       : {s[:110]}")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="evals.goldgen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--listings", default="evals/gold_src/listings.json")
    b.add_argument("--resumes", default="resumes")
    b.add_argument("--out", default="evals/gold_set.json")
    a = sub.add_parser("audit")
    a.add_argument("--gold", default="evals/gold_set.json")
    a.add_argument("-n", type=int, default=8)
    args = ap.parse_args(argv)
    if args.cmd == "audit":
        print(audit(json.loads(Path(args.gold).read_text(encoding="utf-8")), args.n))
        return 0
    listings = json.loads(Path(args.listings).read_text(encoding="utf-8"))["listings"]
    resumes = {p.name: resume_profile(read_resume_text(p))
               for p in sorted(Path(args.resumes).iterdir()) if p.suffix.lower() in {".pdf", ".txt"}}
    gold = build(listings, resumes)
    Path(args.out).write_text(json.dumps(gold, indent=1, ensure_ascii=False), encoding="utf-8")
    m = gold["meta"]
    print(f"{m['queries']} queries from {m['listings_used']} listings x {len(resumes)} resumes; "
          f"{m['scorable_queries']} scorable, {m['negatives']} negatives; buckets {m['buckets']} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
