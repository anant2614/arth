"""Merchant gazetteer: known merchants, billers and VPA handles mapped to concepts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

_SPLIT_VPA = re.compile(r"[.\-_\d]+")


@dataclass(frozen=True)
class MerchantHit:
    name: str
    concept: str
    matched: str  # the alias or VPA that matched
    via: str  # "vpa" or "alias"


class Gazetteer:
    def __init__(self, merchants: list[dict], vpa_handles: dict[str, str]):
        self.by_name = {m["name"]: m for m in merchants}
        self.vpa_handles = {k.lower(): v for k, v in vpa_handles.items()}
        # Longest alias first so "SWIGGYINSTAMART" beats "SWIGGY".
        pairs = sorted(
            ((a.upper(), m["name"]) for m in merchants for a in m["aliases"]),
            key=lambda p: -len(p[0]),
        )
        self._alias_patterns = []
        for alias, name in pairs:
            esc = re.escape(alias)
            # Short aliases (KFC, MGL, OYO) need word boundaries; long ones may be glued
            # to other tokens in narrations ("ZOMATOONLINE", "BAJAJFINANCE-12").
            pat = esc if len(alias) >= 6 else rf"(?<![A-Z0-9]){esc}(?![A-Z0-9])"
            self._alias_patterns.append((re.compile(pat), alias, name))

    @classmethod
    def default(cls) -> "Gazetteer":
        return _default_gazetteer()

    def concept_of(self, name: str) -> str:
        return self.by_name[name]["concept"]

    def match_vpa(self, vpa: str) -> MerchantHit | None:
        local = vpa.split("@", 1)[0].lower()
        name = self.vpa_handles.get(local)
        if name is None:
            head = _SPLIT_VPA.split(local)[0]
            name = self.vpa_handles.get(head) if len(head) >= 3 else None
        if name is None:
            return None
        return MerchantHit(name, self.concept_of(name), vpa, "vpa")

    def match_text(self, text: str) -> MerchantHit | None:
        upper = text.upper()
        for pat, alias, name in self._alias_patterns:
            if pat.search(upper):
                return MerchantHit(name, self.concept_of(name), alias, "alias")
        return None

    def lookup(self, text: str, vpas: list[str] | None = None) -> MerchantHit | None:
        for vpa in vpas or []:
            hit = self.match_vpa(vpa)
            if hit:
                return hit
        return self.match_text(text)

    def is_known_vpa(self, vpa: str) -> bool:
        return self.match_vpa(vpa) is not None


@lru_cache(maxsize=1)
def _default_gazetteer() -> Gazetteer:
    raw = json.loads(resources.files("arth.data").joinpath("gazetteer.json").read_text())
    return Gazetteer(raw["merchants"], raw["vpa_handles"])
