"""Fetch AI/ML engineering listings from MyCareersFuture (Singapore government job portal).

  python -m evals.mcf fetch --out evals/gold_src/listings.json --target 50

Uses the portal's public JSON API that the website itself calls. Be polite: this sends a clear
User-Agent, waits between requests and stops at --target listings. Listing text belongs to the
posting employers, so the output (evals/gold_src/) is gitignored: use it for local evaluation,
do not redistribute it. Each record keeps the source URL for attribution.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional

from evals.listing_safety import audit_text, clean_listing_text, safe_listing_url

API = "https://api.mycareersfuture.gov.sg/v2/jobs"
UA = "job-application-agent-gold-set/1.0 (local research; contact: repo owner)"
SEARCH_TERMS = ["machine learning engineer", "AI engineer", "MLOps engineer", "deep learning engineer",
                "LLM engineer", "generative AI engineer", "computer vision engineer", "NLP engineer",
                "AI/ML engineer", "applied scientist"]
TITLE_OK = re.compile(r"(?i)\b(machine learning|ml|mlops|ai|a\.i\.|artificial intelligence|deep learning|llm|"
                      r"generative|genai|nlp|computer vision|data scien\w+|applied scien\w+)\b")
ROLE_OK = re.compile(r"(?i)(engineer|developer|scientist|architect|specialist|researcher|lead)")
TITLE_BAD = re.compile(r"(?i)\b(intern|internship|trainee|attachment|sales|recruit\w*|marketing)\b")


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in {"br", "p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n" + ("- " if tag == "li" else ""))

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(raw: str) -> str:
    p = _Text()
    p.feed(raw or "")
    text = html.unescape("".join(p.parts)).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def _get(url: str, retries: int = 3, timeout: int = 90) -> Dict[str, Any]:
    if not url.startswith(API + "?"):      # only ever talk to the one fixed https endpoint
        raise ValueError("refusing to fetch a URL outside the MyCareersFuture API")
    last: Optional[BaseException] = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})  # noqa: S310
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https endpoint
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"MCF request failed: {last}")


def normalise(job: Dict[str, Any]) -> Dict[str, Any]:
    meta = job.get("metadata") or {}
    company = (job.get("postedCompany") or {}).get("name") or ""
    skills = [s.get("skill") for s in job.get("skills") or [] if s.get("skill")]
    key_skills = [s.get("skill") for s in job.get("skills") or [] if s.get("isKeySkill") and s.get("skill")]
    url = meta.get("jobDetailsUrl") or f"https://www.mycareersfuture.gov.sg/job/{job.get('uuid')}"
    raw_text = html_to_text(job.get("description") or "")
    cleaned = clean_listing_text(raw_text)          # strips links/emails/phones/active HTML
    audit = audit_text(str(job.get("title", "")) + chr(10) + raw_text)
    return {
        "uuid": job.get("uuid"),
        "post_id": meta.get("jobPostId"),
        "title": (job.get("title") or "").strip(),
        "company": company,
        "url": url,
        "url_ok": safe_listing_url(url),
        "posted": meta.get("newPostingDate") or meta.get("originalPostingDate"),
        "min_years_experience": job.get("minimumYearsExperience"),
        "position_levels": [p.get("position") for p in job.get("positionLevels") or []],
        "listed_skills": skills,
        "key_skills": key_skills,
        "description": cleaned["text"],
        "safety": {"removed": cleaned["removed"], "active_html": cleaned["active_html"], "flags": audit["flags"]},
        "source": "MyCareersFuture (mycareersfuture.gov.sg)",
    }


def wanted(rec: Dict[str, Any]) -> bool:
    t = rec["title"]
    return bool(TITLE_OK.search(t) and ROLE_OK.search(t) and not TITLE_BAD.search(t)
                and len(rec["description"]) >= 400 and rec.get("url_ok", True))


def fetch(target: int = 50, pages: int = 2, per_page: int = 20, delay: float = 2.0,
          terms: Optional[List[str]] = None, log=print) -> List[Dict[str, Any]]:
    seen_uuid, seen_pair, out = set(), set(), []
    for term in terms or SEARCH_TERMS:
        for page in range(pages):
            if len(out) >= target:
                return out
            q = urllib.parse.urlencode({"search": term, "limit": per_page, "page": page})
            try:
                data = _get(f"{API}?{q}")
            except RuntimeError as e:
                log(f"skip {term!r} p{page}: {e}")
                break
            results = data.get("results") or []
            kept = 0
            for job in results:
                rec = normalise(job)
                pair = (rec["company"].lower(), rec["title"].lower())
                if rec["uuid"] in seen_uuid or pair in seen_pair or not wanted(rec):
                    continue
                seen_uuid.add(rec["uuid"]); seen_pair.add(pair); out.append(rec); kept += 1
            log(f"{term!r} page {page}: {len(results)} results, kept {kept}, total {len(out)}")
            if len(results) < per_page:
                break
            time.sleep(delay)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="evals.mcf")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", default="evals/gold_src/listings.json")
    f.add_argument("--target", type=int, default=50)
    f.add_argument("--pages", type=int, default=2)
    a = ap.parse_args(argv)
    recs = fetch(a.target, a.pages)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                       "listings": recs}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(recs)} listings -> {a.out}")
    return 0 if recs else 1


if __name__ == "__main__":
    sys.exit(main())
