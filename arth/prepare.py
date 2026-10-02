"""Per-line preprocessing shared by training, evaluation and serving.

Order matters for privacy: extract on the raw text in memory, scrub, and from then on
only the scrubbed text and the rendered student input are kept or passed anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

from .extract import Extracted, extract
from .features import History, render
from .gazetteer import Gazetteer
from .scrub import scrub


@dataclass
class Prepared:
    extracted: Extracted
    scrubbed: str
    rendered: str
    amount: float | None
    direction: str | None
    history: History | None


def prepare(text: str, amount: float | None = None, direction: str | None = None,
            history: dict | History | None = None, gaz: Gazetteer | None = None) -> Prepared:
    gaz = gaz or Gazetteer.default()
    ex = extract(text, gaz, direction=direction)
    scrubbed = scrub(text, gaz, ex).text
    h = history if isinstance(history, History) or history is None else History.from_dict(history)
    amt = amount if amount is not None else ex.amount
    rendered = render(scrubbed, ex, amt, direction or ex.direction, h)
    return Prepared(ex, scrubbed, rendered, amt, direction or ex.direction, h)


def prepare_rows(rows: list[dict], gaz: Gazetteer | None = None) -> list[Prepared]:
    gaz = gaz or Gazetteer.default()
    return [prepare(r["text"], r.get("amount"), r.get("direction"), r.get("history"), gaz) for r in rows]
