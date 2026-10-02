"""Baselines every student must beat (PRD quality gate "Accuracy").

- ``regex``: a hand-written rule parser of the kind customers run today: gazetteer
  exact match plus keyword rules, with a direction-based fallback.
- ``embedding_zero_shot``: cosine similarity between the line and each option's
  description, no training. Stands in for "zero-shot" when no LLM is configured.
- ``llm_zero_shot``: the teacher LLM with no calibration, when HF_TOKEN is set.
"""

from __future__ import annotations

import re

import numpy as np
import scipy.sparse as sp

from ..prepare import Prepared

KEYWORD_RULES: list[tuple[str, re.Pattern]] = [
    ("bounce", re.compile(r"\b(?:RTN|RETURN(?:ED)?|BOUNCE|DISHONOU?R|INSUFFICIENT)\b", re.I)),
    ("bank_charges", re.compile(r"\b(?:CHGS?|CHARGES?|ANNUAL\s+FEE|AMC|MIN\s+BAL|SMS\s+ALERT)\b", re.I)),
    ("interest", re.compile(r"\b(?:INT\.?\s*PD|INTEREST|INT\s+CR)\b", re.I)),
    ("dividend", re.compile(r"\b(?:DIVIDEND|DIV)\b", re.I)),
    ("refund", re.compile(r"\b(?:REFUND|REVERSAL|REV|CASHBACK)\b", re.I)),
    ("salary", re.compile(r"\b(?:SALARY|SAL|PAYROLL|STIPEND)\b", re.I)),
    ("atm_cash", re.compile(r"\b(?:ATW|NWD|ATM|CASH\s+WDL|EAW)\b", re.I)),
    ("cash_deposit", re.compile(r"\b(?:CASH\s+DEP|CDM|BY\s+CASH)\b", re.I)),
    ("tax", re.compile(r"\b(?:INCOME\s+TAX|GST|TDS|CBDT|CHALLAN|ADVANCE\s+TAX|PROPERTY\s+TAX)\b", re.I)),
    ("credit_card_bill", re.compile(r"\b(?:CREDIT\s+CARD|CC\s+PAYMENT|CARD\s+PAYMENT|CC\s+BILL)\b", re.I)),
    ("emi", re.compile(r"\b(?:EMI|LOAN|NACH|ACH\s+D|ECS)\b", re.I)),
    ("investment", re.compile(r"\b(?:SIP|MUTUAL\s+FUND|MF|ZERODHA|GROWW|NPS|PPF)\b", re.I)),
    ("insurance", re.compile(r"\b(?:INSURANCE|PREMIUM|LIC)\b", re.I)),
    ("rent", re.compile(r"\b(?:RENT|KIRAYA)\b", re.I)),
    ("self_transfer", re.compile(r"\b(?:SELF|OWN\s+A/?C|SWEEP)\b", re.I)),
    ("utilities", re.compile(r"\b(?:ELECTRICITY|EB\s+BILL|POWER|WATER|GAS)\b", re.I)),
    ("telecom", re.compile(r"\b(?:RECHARGE|BROADBAND|POSTPAID|PREPAID|DTH)\b", re.I)),
    ("fuel", re.compile(r"\b(?:PETROL|FUEL|DIESEL|FILLING)\b", re.I)),
]


def regex_concept(text: str, prep: Prepared) -> str:
    if prep.extracted.merchant_concept:
        return prep.extracted.merchant_concept
    for concept, pat in KEYWORD_RULES:
        if pat.search(text):
            return concept
    if prep.extracted.counterparty_type == "person":
        return "p2p_in" if prep.direction == "credit" else "p2p_out"
    return "p2p_in" if prep.direction == "credit" else "shopping"


def embedding_zero_shot(X, option_embeddings) -> np.ndarray:
    S = X @ option_embeddings.T
    return S.toarray() if sp.issparse(S) else np.asarray(S)
