"""Gold set loading and splitting.

The gold set is human-labelled and never enters student training. It is split into a
calibration split (temperature scaling, certified thresholds) and an eval split (quality
gates). The split is by bank so evaluation measures generalisation across bank formats
rather than memorisation of one bank's template.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

DEFAULT_GOLD = Path(__file__).resolve().parent.parent / "data" / "gold" / "proxy_gold.jsonl"


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def split_gold(rows: list[dict], cal_frac: float = 0.4, key: str = "bank", salt: str = "arth-gold-v1"
               ) -> tuple[list[dict], list[dict]]:
    """Deterministic group split: whole groups go to calibration until ``cal_frac`` is reached."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(r.get(key)), []).append(r)
    order = sorted(groups, key=lambda g: zlib.crc32(f"{salt}:{g}".encode()))
    cal, ev = [], []
    for g in order:
        (cal if len(cal) < cal_frac * len(rows) else ev).extend(groups[g])
    return cal, ev
