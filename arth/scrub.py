"""PII scrubber. Runs on every input before storage, logging or any teacher call.

Replaces personal names, account and card numbers, phone numbers, emails, PAN,
Aadhaar, personal VPAs and long reference numbers with placeholders. Merchant VPAs
and business names are kept because they carry the category signal.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from dataclasses import dataclass, field

from .extract import Extracted, extract, looks_like_person, name_lexicon, vpa_is_personal
from .gazetteer import Gazetteer

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
AADHAAR_RE = re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b")
CARD_RE = re.compile(r"\b\d{4,6}[Xx*]{4,10}\d{4}\b|\b(?:\d{4}[ -]?){3}\d{4}\b")
MASKED_ACCT_RE = re.compile(r"(?<![A-Za-z])(?:[Xx*]{1,}\d{3,6})\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[ -]?)?[6-9]\d{9}(?!\d)")
IFSC_RE = re.compile(r"\b([A-Z]{4})0[A-Z0-9]{6}\b")
LONG_NUM_RE = re.compile(r"(?<![\d,.])\d{6,}(?![\d,]|\.\d)")
WORD_RE = re.compile(r"(?<![A-Za-z<])[A-Za-z]+(?![A-Za-z>])")
SEGMENT_SPLIT = re.compile(r"[/:|*()\-]|\s{2,}")
VPA_RE = re.compile(r"(?<![A-Za-z0-9.])[a-zA-Z0-9][a-zA-Z0-9._]{1,63}@[a-zA-Z][a-zA-Z0-9]{1,30}\b")


@dataclass
class ScrubResult:
    text: str
    replaced: dict[str, int] = field(default_factory=dict)


def scrub(text: str, gaz: Gazetteer | None = None, ex: Extracted | None = None) -> ScrubResult:
    gaz = gaz or Gazetteer.default()
    ex = ex or extract(text, gaz)
    counts: dict[str, int] = {}

    def sub(pat: re.Pattern, repl, s: str, key: str) -> str:
        out, n = pat.subn(repl, s)
        if n:
            counts[key] = counts.get(key, 0) + n
        return out

    s = text
    s = sub(EMAIL_RE, "<EMAIL>", s, "email")
    # VPAs next: personal handles go, merchant handles stay.
    s = sub(VPA_RE, lambda m: "<VPA>" if vpa_is_personal(m.group(0), gaz) else m.group(0), s, "vpa")
    counts.pop("vpa", None)
    n_vpa = s.count("<VPA>") - text.count("<VPA>")
    if n_vpa:
        counts["vpa"] = n_vpa
    s = sub(PAN_RE, "<PAN>", s, "pan")
    s = sub(AADHAAR_RE, "<AADHAAR>", s, "aadhaar")
    s = sub(CARD_RE, "<CARD>", s, "card")
    s = sub(PHONE_RE, "<PHONE>", s, "phone")
    s = sub(IFSC_RE, lambda m: f"{m.group(1)}<IFSC>", s, "ifsc")
    s = sub(MASKED_ACCT_RE, "<ACCT>", s, "account")
    s = sub(LONG_NUM_RE, "<NUM>", s, "number")

    names = set()
    if ex.counterparty and ex.counterparty_type in {"person", "unknown", "self"} and looks_like_person(ex.counterparty):
        names.add(ex.counterparty)
    # Structured narrations: any separator-delimited field that reads as a person's name.
    for seg in SEGMENT_SPLIT.split(s):
        seg = seg.strip()
        if seg and "<" not in seg and looks_like_person(seg) and not gaz.match_text(seg):
            names.add(seg)
    first, last, _ = name_lexicon()
    # Free text: "<first name> <surname|initial> [<surname>]" anywhere, unless it is a known merchant.
    toks = list(WORD_RE.finditer(s))
    for i, m in enumerate(toks[:-1]):
        a, b = m.group(0).lower(), toks[i + 1].group(0).lower()
        if a in first and (b in last or len(b) == 1) and s[m.end():toks[i + 1].start()].strip() == "":
            end = toks[i + 1].end()
            if i + 2 < len(toks) and toks[i + 2].group(0).lower() in last \
                    and s[end:toks[i + 2].start()].strip() == "":
                end = toks[i + 2].end()
            span = s[m.start():end]
            if not gaz.match_text(span):
                names.add(span)
    for name in sorted(names, key=len, reverse=True):
        toks = [t for t in re.split(r"[^A-Za-z]+", name) if t]
        if not any(t.lower() in first or t.lower() in last for t in toks):
            continue  # kinship words ("MUMMY", "papa") identify no one and carry signal
        s = sub(re.compile(re.escape(name), re.I), "<NAME>", s, "name")
        # Also drop the name's own tokens elsewhere ("RAHUL SHARMA ... for rahul").
        for tok in toks:
            if len(tok) >= 3 and (tok.lower() in first or tok.lower() in last):
                s = sub(re.compile(rf"(?<![A-Za-z<]){re.escape(tok)}(?![A-Za-z>])", re.I), "<NAME>", s, "name")
    return ScrubResult(s, counts)


def _salt() -> bytes:
    return os.environ.get("ARTH_HASH_SALT", "arth-dev-salt").encode()


def hash_id(*parts: str) -> str:
    """Keyed hash used wherever an identifier must appear in logs or the review store."""
    h = hmac.new(_salt(), "\x1f".join(parts).encode(), hashlib.sha256)
    return h.hexdigest()[:20]
