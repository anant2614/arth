"""Student input rendering and code-computed history features.

History features (recurrence, amount stability, cadence) are computed in code and
written into the input text as tokens, so a single-pass student can read patterns
like "fixed amount, monthly, same person" instead of having to infer them.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

from .extract import Extracted, parse_iso_date


@dataclass
class History:
    same_counterparty_count: int | None = None
    amount_cv: float | None = None
    median_gap_days: float | None = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "History | None":
        if not d:
            return None
        return cls(d.get("same_counterparty_count"), d.get("amount_cv"), d.get("median_gap_days"))


def amount_bucket(amount: float | None) -> str:
    if amount is None:
        return "amt_unknown"
    for edge, name in ((100, "lt100"), (500, "100_500"), (2000, "500_2k"), (10000, "2k_10k"),
                       (50000, "10k_50k"), (200000, "50k_2l")):
        if amount < edge:
            return f"amt_{name}"
    return "amt_gt2l"


def history_tokens(h: History | None) -> list[str]:
    if h is None:
        return ["hist_none"]
    toks = []
    c = h.same_counterparty_count
    if c is not None:
        toks.append("hist_first" if c <= 1 else "hist_few" if c <= 3 else "hist_repeat" if c <= 11 else "hist_many")
    cv = h.amount_cv
    if cv is not None:
        toks.append("amt_fixed" if cv < 0.03 else "amt_stable" if cv < 0.2 else "amt_variable")
    g = h.median_gap_days
    if g is not None:
        toks.append(
            "gap_daily" if g < 3 else "gap_weekly" if g < 10 else "gap_fortnight" if g < 20
            else "gap_monthly" if g <= 35 else "gap_quarterly" if g <= 100 else "gap_rare"
        )
    if c is not None and c >= 3 and cv is not None and cv < 0.1 and g is not None and 25 <= g <= 35:
        toks.append("recurring_monthly_fixed")
    return toks or ["hist_none"]


def render(scrubbed_text: str, ex: Extracted, amount: float | None = None,
           direction: str | None = None, history: History | None = None) -> str:
    """The text the students read: structured hints, then the scrubbed narration."""
    amount = amount if amount is not None else ex.amount
    direction = direction or ex.direction or "unknown"
    # Plain words where possible, so hints share vocabulary with option descriptions.
    hints = [direction, ex.channel.lower(), ex.counterparty_type, amount_bucket(amount)]
    if amount is not None and amount >= 1000 and amount % 500 == 0:
        hints.append("amt_round")
    if ex.merchant:
        hints.append(f"merchant {ex.merchant.lower()}")
        hints.append(f"known {ex.merchant_concept.replace('_', ' ')}")
    hints.extend(history_tokens(history))
    return " ".join(hints) + " | " + scrubbed_text.lower()


def _cp_key(ex: Extracted) -> str | None:
    if ex.merchant:
        return f"m:{ex.merchant}"
    if ex.vpas:
        return f"v:{ex.vpas[0].lower()}"
    if ex.counterparty:
        return f"c:{ex.counterparty.lower()}"
    return None


def compute_history(lines: list[dict], extracted: list[Extracted]) -> list[History | None]:
    """FR-13: history features from one account's lines (each line has amount, date)."""
    groups: dict[str, list[int]] = defaultdict(list)
    keys = []
    for i, (line, ex) in enumerate(zip(lines, extracted)):
        k = _cp_key(ex)
        if k is not None:
            k = f"{line.get('direction') or ex.direction}:{k}"
            groups[k].append(i)
        keys.append(k)
    out: list[History | None] = []
    for i, k in enumerate(keys):
        if k is None:
            out.append(None)
            continue
        idx = groups[k]
        amounts = [a for a in ((lines[j].get("amount") or extracted[j].amount) for j in idx) if a]
        cv = None
        if len(amounts) >= 2 and statistics.mean(amounts) > 0:
            cv = round(statistics.pstdev(amounts) / statistics.mean(amounts), 4)
        dates = sorted(d for d in (parse_iso_date(lines[j].get("date") or extracted[j].date) for j in idx) if d)
        gap = None
        if len(dates) >= 2:
            gap = float(statistics.median((b - a).days for a, b in zip(dates, dates[1:])))
        out.append(History(len(idx), cv, gap))
    return out
