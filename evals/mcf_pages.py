"""Ingest MyCareersFuture listings that were READ FROM THE LIVE WEBSITE (page text), not from the API.

  python -m evals.mcf_pages ingest --url <listing url> --file page.txt     # one page -> evals/gold_src/pages/<id>.json
  python -m evals.mcf_pages collect                                        # all pages -> evals/gold_src/listings.json
  python -m evals.mcf_pages audit                                          # safety report over everything collected

`page.txt` is the visible text of a listing page (what a person would read). Nothing here touches the
network. Every record passes through evals.listing_safety: the URL must be a genuine listing page, links /
emails / phone numbers / active HTML are removed from the body, and anything suspicious is flagged.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from evals.listing_safety import audit_listings, audit_text, clean_listing_text, safe_listing_url

PAGES_DIR = Path("evals/gold_src/pages")
_POST_ID = re.compile(r"^MCF-\d{4}-\d{5,9}$")
_UUID_TAIL = re.compile(r"-([0-9a-f]{32})$")
_EXP = re.compile(r"^(\d{1,2}) years? exp$", re.I)
_BODY_START = "Roles & Responsibilities"
_BODY_STOPS = ("See how you compare with other applicants", "Tell employers what skills you have",
               "About the Company", "Log in to Apply", "People who viewed this job also viewed")
_SKILL_START = "Add these skills if you have them"
_SKILL_STOPS = ("Grow your job opportunities", "Grow your skills, advance your career", "About the Company",
                "Explore some relevant courses", "Tell employers what skills you have")
_BADGE = re.compile(r"^TYPICALLY REPLIES IN \d+ DAYS\s*")      # the only badge seen glued to titles
_NOISE = re.compile(r"(?i)^(report issue|beta|you can view this job in|english \(original\)|machine translation)")


def _lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines()]


def _uuid_from_url(url: str) -> str:
    m = _UUID_TAIL.search(url.rstrip("/").rsplit("/", 1)[-1])
    return m.group(1) if m else ""


def _slice(lines: List[str], start: str, stops: tuple, include_start: bool = False) -> List[str]:
    try:
        i = next(k for k, ln in enumerate(lines) if ln == start)
    except StopIteration:
        return []
    out: List[str] = []
    for ln in lines[i if include_start else i + 1:]:
        if ln in stops:
            break
        out.append(ln)
    return out


def parse_page(url: str, page_text: str) -> Dict[str, Any]:
    """Page text -> listing record. Raises ValueError for anything that is not a genuine listing page."""
    if not safe_listing_url(url):
        raise ValueError("not a MyCareersFuture listing URL")
    uuid = _uuid_from_url(url)
    if not uuid:
        raise ValueError("listing URL has no 32-hex id")
    lines = _lines(page_text)
    if "Getting job details..." in lines or _BODY_START not in lines:
        raise ValueError("page text is not a fully loaded listing (no 'Roles & Responsibilities')")

    pid_idx = next((k for k, ln in enumerate(lines) if _POST_ID.match(ln)), None)
    if pid_idx is None or pid_idx < 2:
        raise ValueError("could not find the job post id")
    title, company = html.unescape(_BADGE.sub("", lines[pid_idx - 1])).strip(), ""
    for k in range(pid_idx - 2, -1, -1):          # company is the nearest non-empty line above the title
        if lines[k] and not _NOISE.match(lines[k]):
            company = html.unescape(_BADGE.sub("", lines[k])).strip()
            break

    header = [ln for ln in lines[pid_idx + 1:lines.index(_BODY_START)] if ln]
    exp = next((int(m.group(1)) for ln in header if (m := _EXP.match(ln))), None)
    levels = [ln for ln in header if ln in {
        "Senior Management", "Middle Management", "Manager", "Professional", "Senior Executive", "Executive",
        "Junior Executive", "Non-executive", "Fresh/entry level", "Fresh/Entry Level"}]

    body_raw = "\n".join(ln for ln in _slice(lines, _BODY_START, _BODY_STOPS))
    skills = [html.unescape(ln) for ln in _slice(lines, _SKILL_START, _SKILL_STOPS) if ln and len(ln) <= 60]
    cleaned = clean_listing_text(body_raw)
    audit = audit_text(f"{title}\n{body_raw}")
    return {
        "uuid": uuid, "post_id": lines[pid_idx], "title": title, "company": company, "url": url, "url_ok": True,
        "min_years_experience": exp, "position_levels": levels[:2],
        "listed_skills": skills, "key_skills": [],
        "description": cleaned["text"],
        "safety": {"removed": cleaned["removed"], "active_html": cleaned["active_html"], "flags": audit["flags"]},
        "source": "MyCareersFuture (mycareersfuture.gov.sg), read from the live listing page",
        "read_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def ingest(url: str, page_text: str, pages_dir: Path = PAGES_DIR) -> Dict[str, Any]:
    rec = parse_page(url, page_text)
    if len(rec["description"]) < 300:
        raise ValueError("description too short to be a real listing body")
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / f"{rec['uuid']}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    return rec


def collect(pages_dir: Path = PAGES_DIR, out: Path = Path("evals/gold_src/listings.json")) -> List[Dict[str, Any]]:
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(pages_dir.glob("*.json"))]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "method": "read from live listing pages", "listings": recs},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    return recs


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="evals.mcf_pages")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("--url", required=True)
    i.add_argument("--file", required=True)
    sub.add_parser("collect")
    sub.add_parser("audit")
    a = ap.parse_args(argv)
    if a.cmd == "ingest":
        try:
            rec = ingest(a.url, Path(a.file).read_text(encoding="utf-8"))
        except ValueError as e:
            print(f"REJECTED: {e}")
            return 1
        s = rec["safety"]
        print(f"OK {rec['uuid']} | {rec['title']} | {rec['company']} | {len(rec['description'])} chars | "
              f"skills {len(rec['listed_skills'])} | removed {s['removed']} | flags {s['flags'] or 'none'}")
        return 0
    if a.cmd == "collect":
        recs = collect()
        print(f"Collected {len(recs)} listings -> evals/gold_src/listings.json")
        return 0 if recs else 1
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(PAGES_DIR.glob("*.json"))]
    rep = audit_listings(recs)
    print(json.dumps({k: v for k, v in rep.items() if k != "details"}, indent=1))
    for d in rep["details"]:
        print(f"FLAGGED {d['uuid']} {d['title']}: url_ok={d['url_ok']} flags={d['flags']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
