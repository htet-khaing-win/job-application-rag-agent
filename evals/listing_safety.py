"""Safety checks for third-party job-listing text and URLs.

Listing text is untrusted data written by employers. This module makes sure the gold-set pipeline
(a) only ever *navigates* to genuine MyCareersFuture listing pages, (b) never stores clickable links,
emails or phone numbers from the body, and (c) reports anything suspicious for a human to look at.
Nothing here performs network I/O or downloads files.
"""
from __future__ import annotations

import ipaddress
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence
from urllib.parse import urlsplit

from security import sanitize_text

ALLOWED_HOSTS = {"www.mycareersfuture.gov.sg", "mycareersfuture.gov.sg"}
_LISTING_PATH = re.compile(r"^/job/[A-Za-z0-9\-_/]{3,200}$")

_URL = re.compile(r"(?i)\b(?:https?://|ftp://|www\.)[^\s<>\"')\]]+|\b[a-z0-9.-]+\.(?:com|net|org|io|sg|co|gov|edu|xyz|top|ru|cn|tk|ml|ga|cf|zip|mov)(?:/[^\s<>\"')\]]*)?\b(?=[\s,.;:)]|$)")
_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){7,12}\d(?!\d)")
_SCHEME_RISK = re.compile(r"(?i)\b(?:javascript|data|vbscript|file|blob):[^\s]*")
_EXEC_EXT = re.compile(r"(?i)\.(?:exe|msi|bat|cmd|ps1|vbs|js|jar|apk|dmg|scr|lnk|iso|zip|rar|7z|docm|xlsm|pptm)\b")
_SHORTENER = re.compile(r"(?i)\b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|is\.gd|buff\.ly|rb\.gy|cutt\.ly|shorturl\.at)/")
_HTML_ACTIVE = re.compile(r"(?i)<\s*(?:script|iframe|object|embed|form|meta|link|style)\b|on\w+\s*=\s*[\"']")
_INJECTION = re.compile(r"(?i)(ignore (?:all |any )?(?:previous|prior|above) (?:instructions|prompts)|"
                        r"disregard (?:the )?(?:system|previous)|you are now |system prompt|"
                        r"reveal your|api[_ ]?key|run the following|execute (?:this|the following)|"
                        r"download (?:and )?(?:install|run))")


def safe_listing_url(url: str) -> bool:
    """True only for https://www.mycareersfuture.gov.sg/job/<slug> pages (no creds, port, query or fragment)."""
    try:
        u = urlsplit(url or "")
    except ValueError:
        return False
    if u.scheme != "https" or u.hostname not in ALLOWED_HOSTS or u.username or u.password:
        return False
    if u.port not in (None, 443) or u.query or u.fragment:
        return False
    return bool(_LISTING_PATH.match(u.path))


def strip_links(text: str) -> Dict[str, Any]:
    """Remove URLs, emails and phone numbers; return cleaned text and what was removed."""
    emails = _EMAIL.findall(text)
    no_mail = _EMAIL.sub(" ", text)
    removed = {"urls": _URL.findall(no_mail) + _SCHEME_RISK.findall(no_mail), "emails": emails,
               "phones": _PHONE.findall(no_mail)}
    cleaned = _EMAIL.sub("[email removed]", text)
    cleaned = _SCHEME_RISK.sub("[link removed]", cleaned)
    cleaned = _URL.sub("[link removed]", cleaned)
    cleaned = _PHONE.sub("[phone removed]", cleaned)
    return {"text": cleaned, "removed": removed}


_CONTACT_LINE = re.compile(
    r"(?i)(ea\s*licen[cs]e|licen[cs]e\s*(?:no|number)|\bea\s*(?:no|reg)|personnel\s*(?:reg|no)|registration\s*(?:no|number)|"
    r"\breg(?:istration)?\.?\s*no\b|\bcontact(?:\s*(?:person|us|details))?\s*[:\-]|\brecruiter\s*[:\-]|"
    r"\bconsultant\s*[:\-]|\battn\.?\s*[:\-]|\battention\s*[:\-]|\bhr\s*(?:manager|contact)\s*[:\-]|"
    r"\bwhatsapp\b|\bwechat\b|\btelegram\b|\b\d{2}[A-Z]\d{4}\b|\bR\d{7}\b)")


def drop_contact_lines(text: str) -> Dict[str, Any]:
    """Remove lines that name a person to contact or carry agency licence numbers (personal data)."""
    kept, dropped = [], 0
    for line in text.splitlines():
        if _CONTACT_LINE.search(line):
            dropped += 1
        else:
            kept.append(line)
    return {"text": "\n".join(kept), "dropped": dropped}


_DASHES = {ord(c): "-" for c in"\u2010\u2011\u2012\u2013\u2014\u2015\u2212"}


def clean_listing_text(text: str, max_chars: int = 20_000) -> Dict[str, Any]:
    """Normalise unicode/control chars, drop active HTML remnants, contact lines and links. Returns text + counts."""
    base = sanitize_text(text, max_chars).translate(_DASHES)   # non-breaking hyphens etc. defeat skill matching
    base = re.sub(r"(?<=\w)\ufffd(?=\w)", "'", base).replace("\ufffd", "")   # mangled apostrophes ("we?re")
    active = len(_HTML_ACTIVE.findall(base))
    base = _HTML_ACTIVE.sub(" ", base)
    contacts = drop_contact_lines(base)
    res = strip_links(contacts["text"])
    removed = {k: len(v) for k, v in res["removed"].items()}
    removed["contact_lines"] = contacts["dropped"]
    return {"text": res["text"], "removed": removed, "active_html": active}


def _domain(raw: str) -> str:
    raw = raw if "://" in raw else "http://" + raw
    try:
        return (urlsplit(raw).hostname or "").lower()
    except ValueError:
        return ""


def _path(raw: str) -> str:
    raw = raw if "://" in raw or ":" in raw.split("/")[0] else "http://" + raw
    try:
        return urlsplit(raw).path or ""
    except ValueError:
        return ""


def audit_text(text: str) -> Dict[str, Any]:
    """Findings for ONE piece of raw (pre-cleaning) listing text. `flags` are the risky ones."""
    no_mail = _EMAIL.sub(" ", text)
    urls = _URL.findall(no_mail) + _SCHEME_RISK.findall(no_mail)
    flags: List[str] = []
    if _SCHEME_RISK.search(text):
        flags.append("script/data/file scheme link")
    if any(_EXEC_EXT.search(_path(u)) for u in urls):
        flags.append("link to executable/archive/macro file type")
    if _SHORTENER.search(text):
        flags.append("URL shortener")
    for u in urls:
        host = _domain(u)
        try:
            ipaddress.ip_address(host)
            flags.append("link to a raw IP address")
        except ValueError:
            pass
        if host.startswith("xn--") or ".xn--" in host:
            flags.append("punycode (possible look-alike) domain")
    if _HTML_ACTIVE.search(text):
        flags.append("active HTML (script/iframe/form/event handler)")
    if _INJECTION.search(text):
        flags.append("text addressed to an AI/assistant (treated as data, never followed)")
    return {"urls": len(urls), "emails": len(_EMAIL.findall(text)), "domains": sorted({_domain(u) for u in urls if _domain(u)}),
            "flags": sorted(set(flags))}


def audit_listings(listings: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Audit raw listing records (description text + source URL). Pure function, no network."""
    per: List[Dict[str, Any]] = []
    domains: Counter = Counter()
    for rec in listings:
        a = audit_text(f"{rec.get('title', '')}\n{rec.get('description', '')}")
        a["url_ok"] = safe_listing_url(str(rec.get("url", "")))
        a["uuid"], a["title"] = rec.get("uuid"), rec.get("title")
        domains.update(a["domains"])
        per.append(a)
    flagged = [p for p in per if p["flags"] or not p["url_ok"]]
    return {"listings": len(per), "flagged": len(flagged), "bad_source_urls": sum(not p["url_ok"] for p in per),
            "links_in_bodies": sum(p["urls"] for p in per), "emails_in_bodies": sum(p["emails"] for p in per),
            "top_linked_domains": domains.most_common(10), "details": flagged}
