"""Deterministic field extraction from Indian transaction text (FR-2).

Amount, balance, date, account suffix and channel come from regular expressions,
never from a model. Counterparty parsing is also rule-based; its output feeds the
gazetteer, the PII scrubber and the student's input text.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from functools import lru_cache
from importlib import resources

from .gazetteer import Gazetteer, MerchantHit

_NUM = r"(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
_CUR = r"(?:rs\.?|inr|₹)"
AMOUNT_RE = re.compile(rf"{_CUR}\s*{_NUM}", re.I)
AMOUNT_AFTER_VERB_RE = re.compile(
    rf"(?:debited|credited|spent|paid|sent|received|withdrawn|deposited)\s+(?:by|for|with|of)?\s*{_NUM}(?![\d/-])",
    re.I,
)
BALANCE_RE = re.compile(
    rf"(?:avl\.?\s*bal(?:ance)?|available\s+bal(?:ance)?|a/c\s+bal(?:ance)?|clear\s+bal|\bbal(?:ance)?)\s*(?:is|:|-)?\s*(?:{_CUR})?\s*:?\s*{_NUM}",
    re.I,
)
ACCOUNT_RE = re.compile(
    r"(?:a/c|acct|account|ac|a/c\s*no\.?|card(?:\s+no\.?)?|card\s+ending(?:\s+with)?)\s*(?:no\.?)?\s*[:\-]?\s*"
    r"(?:[x*]+|\d{4}[x*]+[x*\d]*?)(\d{3,6})\b",
    re.I,
)
BARE_MASK_RE = re.compile(r"\b[xX*]{2,}(\d{3,6})\b")
VPA_RE = re.compile(r"(?<![A-Za-z0-9.])([a-zA-Z0-9][a-zA-Z0-9._]{1,63}@[a-zA-Z][a-zA-Z0-9]{1,30})\b")
REF_RE = re.compile(r"(?:ref(?:erence)?\.?\s*(?:no\.?)?|utr|rrn)\s*[:\-]?\s*([A-Za-z0-9]{6,22})", re.I)

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "ymd"),
    (re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})\b"), "dmy"),
    (re.compile(r"\b(\d{1,2})[-\s]?([A-Za-z]{3})[a-z]*[-\s,]*(\d{4}|\d{2})\b"), "dMy"),
]

CHANNEL_RULES: list[tuple[str, re.Pattern]] = [
    ("NACH", re.compile(r"\b(?:NACH|ACH|ECS|E-?MANDATE|AUTOPAY|SI\s+DR|STANDING\s+INSTRUCTION)\b", re.I)),
    ("UPI", re.compile(r"\bUPI\b|\bVPA\b|@(?:ok\w+|ybl|ibl|axl|paytm|upi|apl|icici|sbi|hdfcbank|axisbank|kotak|yesbank)\b|UPI[/-]", re.I)),
    ("IMPS", re.compile(r"\bIMPS\b|\bMMT\b", re.I)),
    ("NEFT", re.compile(r"\bNEFT\b", re.I)),
    ("RTGS", re.compile(r"\bRTGS\b", re.I)),
    ("ATM", re.compile(r"\b(?:ATW|NWD|ATM|CASH\s+WDL|CASH\s+WITHDRAWAL|EAW)\b", re.I)),
    ("CARD", re.compile(r"\b(?:POS|ECOM|PCD|VPS|IPS|debit\s+card|credit\s+card|card\s+(?:no\.?|ending|xx|\*)|spent\s+(?:on|using|via)?\s*(?:your\s+)?(?:\w+\s+)?card)\b|\bspent\b.*\bcard\b", re.I)),
    ("BILLPAY", re.compile(r"\b(?:BBPS|BIL/|BPAY|BILLPAY|BILL\s*DESK|BILLDESK)\b|BIL/", re.I)),
    ("CHEQUE", re.compile(r"\b(?:CHQ|CHEQUE|CLG|CLEARING|CTS)\b", re.I)),
    ("CASH", re.compile(r"\b(?:CASH\s+DEP(?:OSIT)?|CDM|BY\s+CASH|CASH-)\b", re.I)),
    ("INTEREST", re.compile(r"\b(?:INT\.?\s*PD|INT\.?\s*CR|INTEREST|CREDIT\s+INTEREST)\b", re.I)),
    ("CHARGES", re.compile(r"\b(?:CHGS?|CHARGES?|FEE|PENALTY|AMC|GST\s+ON)\b", re.I)),
    ("NETBANKING", re.compile(r"\b(?:NETBANKING|NET\s+BANKING|IB|MB|INB|TPT|INFT|INF|FT|TRF|TRANSFER)\b", re.I)),
]

_DEBIT_RE = re.compile(
    r"\b(?:debited|debit|dr|spent|sent|paid|withdrawn|purchase|txn\s+of|payment\s+of)\b|/DR/|\bD-|UPI/P2M|UPI/P2A/DR", re.I)
_CREDIT_RE = re.compile(
    r"\b(?:credited|credit|cr|received|deposited|refund(?:ed)?|reversal|cashback|int\.?\s*pd)\b|/CR/|\bC-", re.I)

BUSINESS_WORDS = re.compile(
    r"\b(?:PVT|PRIVATE|LTD|LIMITED|LLP|INC|CORP|CO|COMPANY|ENTERPRISES?|TRADERS?|TRADING|STORES?|MART|"
    r"AGENC(?:Y|IES)|SERVICES?|SOLUTIONS?|TECHNOLOGIES|TECH|INDUSTRIES|SUPERMARKET|RESTAURANT|HOTEL|CAFE|"
    r"DHABA|BAKERY|SWEETS|PHARMA(?:CY)?|MEDICALS?|CHEMISTS?|HOSPITAL|CLINIC|DIAGNOSTICS?|LABS?|MOTORS|"
    r"AUTOMOBILES|PETROL|FUELS?|FILLING|ELECTRICALS?|ELECTRONICS|HARDWARE|TEXTILES|GARMENTS|FOODS?|"
    r"KITCHEN|BAZAAR|EMPORIUM|CENTRE|CENTER|ASSOCIATES|CONSULTANTS?|INTERNATIONAL|GLOBAL|VENTURES|"
    r"FINANCE|FINSERV|CAPITAL|BANK|INSURANCE|SCHOOL|COLLEGE|ACADEMY|UNIVERSITY|INSTITUTE|TRUST|"
    r"SOCIETY|GYM|FITNESS|SALON|STUDIO|PAYMENTS?|SETTLEMENT|BHANDAR|KIRANA|GENERAL|SONS)\b", re.I)
MERCHANT_VPA_HINT = re.compile(
    r"(?:paytmqr|bharatpe|q\d{5,}|merchant|store|shop|mart|enterprise|traders|\.rzp|razorpay|billdesk|"
    r"payu|cashfree|gpay-\d|bqr|mab\.|ezetap|pinelabs|instamojo|\.pos)", re.I)
PERSONAL_VPA_HANDLES = {
    "okhdfcbank", "okaxis", "oksbi", "okicici", "ybl", "ibl", "axl", "paytm", "upi", "apl",
    "naviaxis", "kotak", "icici", "sbi", "hdfcbank", "axisbank", "yesbank", "fbl", "boi", "pnb",
    "barodampay", "cnrb", "ptyes", "ptsbi", "pthdfc", "ptaxis", "jupiteraxis", "freecharge", "airtel",
}
SELF_RE = re.compile(r"\b(?:SELF|OWN\s+A/?C|OWN\s+ACCOUNT|TO\s+SELF|BY\s+SELF|SWEEP)\b", re.I)
INSTITUTION_RE = re.compile(
    r"\b(?:INT\.?\s*PD|INTEREST|CHGS?|CHARGES?|AMC|ATW|NWD|ATM|CASH\s+DEP|CDM|DIVIDEND|DIV|"
    r"TAX|GST|TDS|CBDT|RETURN|RTN|BOUNCE)\b", re.I)

_STOPWORDS_NAME = {
    "UPI", "IMPS", "NEFT", "RTGS", "P2A", "P2M", "DR", "CR", "PAY", "PAYMENT", "SENT", "TO", "FROM", "BY",
    "MB", "IB", "TRF", "TRANSFER", "FT", "INFT", "NA", "NULL", "OTHERS", "OTH", "NO", "REMARKS",
}
_NAME_TOKEN = re.compile(r"^[A-Za-z][A-Za-z.']{0,20}$")
_SCRUBBED_PERSON = re.compile(r"<NAME>|<VPA>|<PHONE>")
_IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$", re.I)


@dataclass
class Extracted:
    amount: float | None = None
    balance: float | None = None
    date: str | None = None
    account_suffix: str | None = None
    channel: str = "OTHER"
    direction: str | None = None
    vpas: list[str] = field(default_factory=list)
    reference: str | None = None
    counterparty: str | None = None  # raw counterparty string; may be a person's name (PII)
    counterparty_type: str = "unknown"  # merchant | person | institution | self | unknown
    merchant: str | None = None
    merchant_concept: str | None = None
    remark: str | None = None

    def public(self) -> dict:
        """Fields safe to return to the caller and to log (no raw counterparty)."""
        d = asdict(self)
        d.pop("counterparty")
        d.pop("vpas")
        d.pop("remark")
        d.pop("reference")
        return d


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def parse_date(text: str) -> str | None:
    for pat, kind in DATE_PATTERNS:
        for m in pat.finditer(text):
            try:
                if kind == "ymd":
                    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                elif kind == "dmy":
                    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                else:
                    mon = m.group(2).lower()[:3]
                    if mon not in _MONTHS:
                        continue
                    d, mo, y = int(m.group(1)), _MONTHS[mon], int(m.group(3))
                if y < 100:
                    y += 2000
                return date(y, mo, d).isoformat()
            except ValueError:
                continue
    return None


def parse_amount(text: str) -> float | None:
    for m in AMOUNT_RE.finditer(text):
        before = text[max(0, m.start() - 18): m.start()].lower()
        if re.search(r"bal|balance|limit|avl", before):
            continue
        return _to_float(m.group(1))
    m = AMOUNT_AFTER_VERB_RE.search(text)
    if m:
        return _to_float(m.group(1))
    return None


def detect_channel(text: str) -> str:
    for name, pat in CHANNEL_RULES:
        if pat.search(text):
            return name
    return "OTHER"


def detect_direction(text: str) -> str | None:
    d = _DEBIT_RE.search(text)
    c = _CREDIT_RE.search(text)
    if d and c:
        return "debit" if d.start() < c.start() else "credit"
    if d:
        return "debit"
    if c:
        return "credit"
    return None


_LEAD_NOISE = re.compile(r"^(?:(?:upi|imps|neft|rtgs|vpa|from|by|to|via|mr\.?|mrs\.?|ms\.?)\s+)+", re.I)


def _clean_name(s: str) -> str | None:
    s = re.sub(r"(?:\s+[\dXx*]{4,})+\s*$", "", s)  # trailing reference numbers
    s = re.sub(r"\s+", " ", s.strip(" .:-/*"))
    s = _LEAD_NOISE.sub("", s).strip(" .:-/*")
    if re.search(r"\b(?:a/c|ac|acct|account|bank\s+card|card)\b", s, re.I):
        return None
    if not s or s.upper() in _STOPWORDS_NAME or s.isdigit() or _IFSC_RE.match(s):
        return None
    if re.fullmatch(r"[\dXx*]+", s):
        return None
    return s


def _split_structured(text: str) -> tuple[str | None, str | None]:
    """Counterparty and remark from structured statement narrations."""
    t = text.strip()
    up = t.upper()
    if re.match(r"^UPI[/:]", up):
        parts = [p.strip() for p in t.split(t[3])]
        # UPI/DR/ref/NAME/BANK/vpa/note   or   UPI/P2A/ref/NAME/BANK/note
        cands = [p for p in parts[1:] if p and not re.fullmatch(r"(?:DR|CR|P2A|P2M|\d+)", p, re.I)]
        name = _clean_name(cands[0]) if cands else None
        remark = cands[-1] if len(cands) >= 3 else None
        return name, remark
    if re.match(r"^UPI-", up):
        parts = [p.strip() for p in t.split("-")]
        cands = [p for p in parts[1:] if p and not re.fullmatch(r"(?:DR|CR|P2A|P2M|\d+)", p, re.I)]
        name = _clean_name(cands[0]) if cands else None
        remark = parts[-1] if len(parts) > 4 else None
        return name, remark
    m = re.match(r"^(?:IMPS|NEFT|RTGS|MMT)[/\- ]+(.*)$", t, re.I)
    if m:
        rest = m.group(1)
        parts = [p.strip() for p in re.split(r"[/\-]", rest) if p.strip()]
        parts = [p for p in parts if not re.fullmatch(r"(?:P2A|P2M|CR|DR|INWARD|OUTWARD|[A-Z]{4}0[A-Z0-9]{6}|[A-Z]{4}[A-Z0-9]*\d{6,}|\d+)", p, re.I)]
        name = _clean_name(parts[0]) if parts else None
        remark = parts[-1] if len(parts) >= 2 else None
        return name, remark
    return None, None


_SMS_CP_RE = [
    re.compile(r"\b(?:to|towards|trf\s+to|transfer\s+to|paid\s+to|sent\s+to)\s+(?:vpa\s+)?([A-Za-z][A-Za-z0-9 .&'@\-]{1,48}?)(?=\s+(?:on|via|ref|upi|for|dt|at|from|avl|info|\.|,)|[.,;(]|$)", re.I),
    re.compile(r"\b(?:from|by|frm)\s+(?:vpa\s+)?([A-Za-z][A-Za-z0-9 .&'@\-]{1,48}?)(?=\s+(?:on|via|ref|upi|for|dt|at|to|avl|info|\.|,)|[.,;(]|$)", re.I),
    re.compile(r"\b(?:at|@)\s+([A-Za-z][A-Za-z0-9 .&'\-]{1,48}?)(?=\s+(?:on|via|ref|for|dt|avl|info)|[.,;(]|$)", re.I),
    re.compile(r"\bInfo:?\s*([A-Za-z][A-Za-z0-9 .&'/\-]{1,48}?)(?=[.,;]|\s+avl|$)", re.I),
]


def _sms_counterparty(text: str) -> str | None:
    for pat in _SMS_CP_RE:
        for m in pat.finditer(text):
            cand = m.group(1).strip()
            if re.match(r"^(?:a/c|ac|acct|account|your|card|xx|\*|rs|inr)\b", cand, re.I) \
                    or re.match(r"^(?:rs|inr)\.?\s*\d", cand, re.I):
                continue
            c = _clean_name(cand)
            if c:
                return c
    return None


@lru_cache(maxsize=1)
def name_lexicon() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    raw = json.loads(resources.files("arth.data").joinpath("names.json").read_text())
    return frozenset(raw["first"]), frozenset(raw["last"]), frozenset(raw["kin"])


def _name_tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z]+", name.lower()) if t]


def looks_like_person(name: str | None) -> bool:
    """A counterparty is a person when it is 1-4 name-like tokens, at least one of which
    is a known Indian first name, surname or kinship term, and none is a business word."""
    if not name:
        return False
    if BUSINESS_WORDS.search(name) or INSTITUTION_RE.search(name):
        return False
    toks = name.replace(".", " ").replace("(", " ").replace(")", " ").split()
    if not 1 <= len(toks) <= 4 or not all(_NAME_TOKEN.match(t) for t in toks):
        return False
    first, last, kin = name_lexicon()
    return any(t in first or t in last or t in kin for t in _name_tokens(name))


def vpa_is_personal(vpa: str, gaz: Gazetteer) -> bool:
    if gaz.is_known_vpa(vpa):
        return False
    local = vpa.partition("@")[0].lower()
    if MERCHANT_VPA_HINT.search(vpa):
        return False
    if re.search(r"\d{10}", local):
        return True
    first, last, kin = name_lexicon()
    chunks = [c for c in re.split(r"[^a-z]+", local) if c]
    if any(c in first or c in last or c in kin for c in chunks):
        return True
    # glued names: "rahulsharma", "priyanair12"
    return any(c.startswith(f) and len(f) >= 4 for c in chunks for f in first)


def extract(text: str, gaz: Gazetteer | None = None, direction: str | None = None) -> Extracted:
    gaz = gaz or Gazetteer.default()
    ex = Extracted()
    ex.amount = parse_amount(text)
    bm = BALANCE_RE.search(text)
    if bm:
        ex.balance = _to_float(bm.group(1))
    ex.date = parse_date(text)
    am = ACCOUNT_RE.search(text) or BARE_MASK_RE.search(text)
    if am:
        ex.account_suffix = am.group(1)
    ex.channel = detect_channel(text)
    ex.direction = direction or detect_direction(text)
    ex.vpas = VPA_RE.findall(text)
    rm = REF_RE.search(text)
    if rm:
        ex.reference = rm.group(1)

    cp, remark = _split_structured(text)
    if cp is None:
        cp = _sms_counterparty(text)
    if cp and "@" in cp:
        cp = None  # a VPA, not a name; the VPA list carries it
    ex.counterparty, ex.remark = cp, remark

    hit: MerchantHit | None = gaz.lookup(text, ex.vpas)
    if hit:
        ex.merchant, ex.merchant_concept = hit.name, hit.concept

    if SELF_RE.search(text):
        ex.counterparty_type = "self"
    elif hit:
        ex.counterparty_type = "merchant"
    elif cp and BUSINESS_WORDS.search(cp):
        ex.counterparty_type = "merchant"
    elif ex.channel in {"INTEREST", "CHARGES", "ATM", "CASH"} or (not cp and INSTITUTION_RE.search(text)):
        ex.counterparty_type = "institution"
    elif any(MERCHANT_VPA_HINT.search(v) for v in ex.vpas):
        ex.counterparty_type = "merchant"
    elif ex.channel == "CARD":
        ex.counterparty_type = "merchant"
    elif looks_like_person(cp) or any(vpa_is_personal(v, gaz) for v in ex.vpas) or _SCRUBBED_PERSON.search(text):
        ex.counterparty_type = "person"  # also true for already-scrubbed text
    return ex


def parse_iso_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
