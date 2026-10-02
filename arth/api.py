"""HTTP API (FastAPI).

    ARTH_MODELS_DIR=models ARTH_DB=arth.sqlite uvicorn arth.api:app

Endpoints (all JSON):

- ``POST /v1/categorize``            single or batch (<= 1,000 lines) against questions or a taxonomy
- ``POST /v1/taxonomies``            save a taxonomy and certify its thresholds on a labelled sample
- ``GET  /v1/taxonomies[/{id}]``     saved taxonomies and their certificates
- ``POST /v1/taxonomies/{id}/recertify``  retrain the fast path with corrections and re-certify
- ``GET  /v1/review``                review queue: non-auto answers with their top 3 options
- ``POST /v1/corrections``           reviewer label for a queued item or a transaction
- ``GET  /v1/corrections/export``    scrubbed corrections for the next training cycle
- ``GET  /v1/models``, ``POST /v1/models/rollback``, ``POST /v1/models/promote``
- ``GET  /v1/health``
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
import datetime as dt
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from .engine import (AuditMonitor, Engine, StudentBundle, TaxonomyModel, builtin_taxonomy, correction_key,
                     fit_saved_taxonomy)
from .prepare import prepare
from .questions import MAX_OPTIONS, Question
from .scrub import hash_id
from .store import Store

log = logging.getLogger("arth.api")

MAX_BATCH = 1000
MAX_QUESTIONS = 20
NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"


# ---------------------------------------------------------------------------
# Schemas


class HistoryIn(BaseModel):
    same_counterparty_count: int | None = Field(None, ge=0)
    amount_cv: float | None = Field(None, ge=0)
    median_gap_days: float | None = Field(None, ge=0)


class TransactionIn(BaseModel):
    id: str | None = Field(None, max_length=128)
    text: str = Field(min_length=1, max_length=1000)
    amount: float | None = Field(None, ge=0)
    direction: Literal["debit", "credit"] | None = None
    date: dt.date | None = None
    history: HistoryIn | None = None

    def as_dict(self) -> dict:
        d = self.model_dump()
        d["date"] = self.date.isoformat() if self.date else None
        d["history"] = self.history.model_dump() if self.history else None
        return d


class QuestionIn(BaseModel):
    name: str = Field(pattern=NAME_PATTERN)
    type: Literal["choice", "bool"]
    options: dict[str, str] | None = None
    description: str | None = Field(None, max_length=500)

    @model_validator(mode="after")
    def _check(self) -> "QuestionIn":
        if self.type == "choice":
            if not self.options or len(self.options) < 2:
                raise ValueError(f"choice question '{self.name}' needs at least 2 options")
            if len(self.options) > MAX_OPTIONS:
                raise ValueError(f"choice question '{self.name}' has {len(self.options)} options; max is {MAX_OPTIONS}")
            for k, v in self.options.items():
                if not k or len(k) > 64:
                    raise ValueError(f"option names must be 1-64 characters: {k!r}")
                if len(v) > 300:
                    raise ValueError(f"option description too long for {k!r} (max 300)")
        elif self.options:
            raise ValueError(f"bool question '{self.name}' takes no options; use 'description'")
        return self

    def to_question(self) -> Question:
        if self.type == "choice":
            return Question.choice(self.name, self.options)
        return Question.boolean(self.name, self.description)


def _unique_names(qs: list[QuestionIn] | None) -> None:
    if qs:
        names = [q.name for q in qs]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"duplicate question names: {sorted(dup)}")


class CategorizeRequest(BaseModel):
    transactions: list[TransactionIn] = Field(min_length=1, max_length=MAX_BATCH)
    questions: list[QuestionIn] | None = Field(None, max_length=MAX_QUESTIONS)
    taxonomy_id: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "CategorizeRequest":
        if not self.questions and not self.taxonomy_id:
            raise ValueError("send 'questions', a 'taxonomy_id', or both")
        _unique_names(self.questions)
        return self


class LabeledLine(TransactionIn):
    labels: dict[str, str | bool]


class TaxonomyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    questions: list[QuestionIn] = Field(min_length=1, max_length=MAX_QUESTIONS)
    labeled_sample: list[LabeledLine] = Field(default_factory=list, max_length=5000)
    opt_in_training: bool = False

    @model_validator(mode="after")
    def _check(self) -> "TaxonomyCreate":
        _unique_names(self.questions)
        qs = {q.name: q for q in self.questions}
        for i, line in enumerate(self.labeled_sample):
            for name, value in line.labels.items():
                q = qs.get(name)
                if q is None:
                    raise ValueError(f"labeled_sample[{i}] labels unknown question '{name}'")
                if q.type == "choice" and value not in q.options:
                    raise ValueError(f"labeled_sample[{i}].{name}: {value!r} is not an option")
                if q.type == "bool" and not isinstance(value, bool):
                    raise ValueError(f"labeled_sample[{i}].{name}: bool questions take true/false")
        return self


class CorrectionIn(BaseModel):
    review_id: str | None = None
    transaction: TransactionIn | None = None
    taxonomy_id: str | None = None
    question: str | None = None
    value: str | bool

    @model_validator(mode="after")
    def _check(self) -> "CorrectionIn":
        if not self.review_id and not (self.transaction and self.taxonomy_id and self.question):
            raise ValueError("send 'review_id', or 'transaction' + 'taxonomy_id' + 'question'")
        return self


class PromoteIn(BaseModel):
    version: str


# ---------------------------------------------------------------------------
# Service


class Service:
    def __init__(self, store: Store, registry=None, bundle: StudentBundle | None = None, teacher=None,
                 audit: AuditMonitor | None = None):
        self.store = store
        self.registry = registry
        self.teacher = teacher
        self.audit = audit or AuditMonitor()
        self._lock = threading.Lock()
        self.engine: Engine | None = None
        if bundle is None and registry is not None and registry.current():
            bundle = registry.load()
        if bundle is not None:
            self.set_bundle(bundle)

    def set_bundle(self, bundle: StudentBundle) -> None:
        with self._lock:
            self.engine = Engine(bundle, self.teacher, self.audit)

    @property
    def model_version(self) -> str:
        return f"student:{self.engine.bundle.version}" if self.engine else "none"

    def require_engine(self) -> Engine:
        if self.engine is None:
            raise HTTPException(503, "no model loaded; train one with `python -m arth.train`")
        return self.engine

    # taxonomies -----------------------------------------------------------
    def resolve_taxonomy(self, tax_id: str) -> tuple[TaxonomyModel, list[Question]]:
        eng = self.require_engine()
        if tax_id.startswith("builtin:"):
            tm = builtin_taxonomy(eng.bundle, tax_id.split(":", 1)[1])
            if tm is None:
                raise HTTPException(404, f"unknown built-in taxonomy '{tax_id}'")
            return tm, [qm.question for qm in tm.questions.values()]
        rec = self.store.get_taxonomy(tax_id)
        if rec is None:
            raise HTTPException(404, f"unknown taxonomy '{tax_id}'")
        tm: TaxonomyModel = rec["model"]
        if tm is None or tm.model_version != eng.bundle.version:
            tm = self.fit_and_store(tax_id, rec["name"], rec["questions"], rec["sample"], rec["opt_in_training"])
        return tm, [qm.question for qm in tm.questions.values()]

    def fit_and_store(self, tax_id: str | None, name: str, questions: list[dict], sample: list[dict],
                      opt_in: bool) -> TaxonomyModel:
        eng = self.require_engine()
        qs = [QuestionIn(**q).to_question() for q in questions]
        labels = {q.name: [s["labels"].get(q.name) for s in sample] for q in qs
                  if sample and all(q.name in s["labels"] for s in sample)}
        corrections = self.store.corrections_for(tax_id) if tax_id else None
        tid = tax_id or f"tax_{uuid.uuid4().hex[:12]}"
        tm = fit_saved_taxonomy(eng.bundle, tid, qs, sample, labels, corrections=corrections)
        self.store.save_taxonomy(tid, name, questions, sample, tm, eng.bundle.version, opt_in)
        return tm


def taxonomy_view(tm: TaxonomyModel) -> dict:
    return {
        "id": tm.id,
        "model_version": f"student:{tm.model_version}",
        "questions": {
            name: {"type": qm.question.type, "source": qm.source,
                   "certificate": qm.certificate.to_dict() if qm.certificate else None}
            for name, qm in tm.questions.items()
        },
    }


def _scrub_line(t: dict) -> dict:
    """What we keep of a labelled line: scrubbed text plus the numeric context."""
    p = prepare(t["text"], t.get("amount"), t.get("direction"), t.get("history"))
    return {"text": p.scrubbed, "amount": t.get("amount"), "direction": t.get("direction"),
            "history": t.get("history"), "labels": t.get("labels", {})}


def create_app(service: Service | None = None) -> FastAPI:
    if service is None:
        from .registry import Registry

        teacher = None
        if os.environ.get("HF_TOKEN") and os.environ.get("ARTH_TEACHER", "hf") == "hf":
            from .teacher import HFTeacher

            teacher = HFTeacher()
        service = Service(Store(os.environ.get("ARTH_DB", "arth.sqlite")),
                          Registry(os.environ.get("ARTH_MODELS_DIR", "models")), teacher=teacher)

    app = FastAPI(title="Arth: Indian transaction categorization", version="0.1.0")
    app.state.service = service
    api_keys = {k.strip() for k in os.environ.get("ARTH_API_KEYS", "").split(",") if k.strip()}

    def auth(x_api_key: str | None = Header(None)) -> None:
        if api_keys and x_api_key not in api_keys:
            raise HTTPException(401, "missing or invalid X-API-Key")

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        t = time.perf_counter()
        resp = await call_next(request)
        # Method, path, status and latency only: never bodies or narrations.
        log.info("%s %s %s %.1fms", request.method, request.url.path, resp.status_code,
                 (time.perf_counter() - t) * 1000)
        return resp

    svc = service

    @app.get("/v1/health")
    def health():
        return {"status": "ok" if svc.engine else "no_model", "model_version": svc.model_version,
                "teacher": getattr(svc.teacher, "version", None), "audit": svc.audit.status()}

    @app.post("/v1/categorize", dependencies=[Depends(auth)])
    def categorize(req: CategorizeRequest):
        eng = svc.require_engine()
        taxonomy, questions = None, []
        if req.taxonomy_id:
            taxonomy, questions = svc.resolve_taxonomy(req.taxonomy_id)
        if req.questions:
            # Named questions only; a name the taxonomy knows uses its certified model.
            questions = [taxonomy.questions[q.name].question if taxonomy and q.name in taxonomy.questions
                         else q.to_question() for q in req.questions]
        txs = [t.as_dict() for t in req.transactions]
        results = eng.categorize(txs, questions, taxonomy, correction_lookup=svc.store.lookup_corrections)
        tax_key = taxonomy.id if taxonomy else None
        queue, where = [], []
        for ri, r in enumerate(results):
            for qname, a in r["_answers"].items():
                if a.auto:
                    continue
                p = r["_prep"]
                queue.append({
                    "line_hash": hash_id(r["id"] or "", p.scrubbed), "taxonomy_id": req.taxonomy_id,
                    "question": qname, "question_spec": {"type": a.question.type, "options": a.question.option_names,
                                                         "description": a.question.description},
                    "scrubbed_text": p.scrubbed, "rendered_text": p.rendered, "top3": a.top3(), "value": a.to_dict()["value"], "route": a.route,
                    "correction_key": correction_key(tax_key, a.question, p), "model_version": svc.model_version,
                })
                where.append((ri, qname))
        ids = svc.store.enqueue(queue) if queue else []
        out = []
        for r in results:
            r.pop("_answers")
            r.pop("_prep")
            r["model_version"] = svc.model_version
            r["review"] = {}
            out.append(r)
        for (ri, qname), rid in zip(where, ids):
            out[ri]["review"][qname] = rid
        return {"results": out, "model_version": svc.model_version, "taxonomy_id": req.taxonomy_id}

    @app.post("/v1/taxonomies", dependencies=[Depends(auth)], status_code=201)
    def create_taxonomy(req: TaxonomyCreate):
        svc.require_engine()
        sample = [_scrub_line({**l.as_dict(), "labels": l.labels}) for l in req.labeled_sample]
        tm = svc.fit_and_store(None, req.name, [q.model_dump() for q in req.questions], sample, req.opt_in_training)
        return {**taxonomy_view(tm), "name": req.name, "n_labeled": len(sample)}

    @app.get("/v1/taxonomies", dependencies=[Depends(auth)])
    def list_taxonomies():
        eng = svc.require_engine()
        builtins = [{"id": f"builtin:{n}", "type": qm.question.type, "certificate": qm.certificate.to_dict()
                     if qm.certificate else None} for n, qm in eng.bundle.builtin.items()]
        return {"saved": svc.store.list_taxonomies(), "builtin": builtins}

    @app.get("/v1/taxonomies/{tax_id}", dependencies=[Depends(auth)])
    def get_taxonomy(tax_id: str):
        tm, _ = svc.resolve_taxonomy(tax_id)
        return taxonomy_view(tm)

    @app.post("/v1/taxonomies/{tax_id}/recertify", dependencies=[Depends(auth)])
    def recertify(tax_id: str):
        rec = svc.store.get_taxonomy(tax_id)
        if rec is None:
            raise HTTPException(404, f"unknown taxonomy '{tax_id}'")
        tm = svc.fit_and_store(tax_id, rec["name"], rec["questions"], rec["sample"], rec["opt_in_training"])
        return taxonomy_view(tm)

    @app.get("/v1/review", dependencies=[Depends(auth)])
    def review(taxonomy_id: str | None = None, status: Literal["open", "resolved"] = "open", limit: int = 100):
        items = svc.store.review_queue(taxonomy_id, status, max(1, min(limit, 1000)))
        for it in items:
            it.pop("correction_key", None)
            it.pop("rendered_text", None)
        return {"items": items}

    @app.post("/v1/corrections", dependencies=[Depends(auth)], status_code=201)
    def correct(req: CorrectionIn):
        if req.review_id:
            item = svc.store.get_review(req.review_id)
            if item is None:
                raise HTTPException(404, f"unknown review item '{req.review_id}'")
            spec = item["question_spec"]
            _validate_value(spec["type"], spec.get("options"), req.value)
            cid = svc.store.add_correction(item["question"], req.value, item["taxonomy_id"], item["correction_key"],
                                           item["scrubbed_text"], review_id=req.review_id,
                                           rendered_text=item["rendered_text"])
            return {"id": cid, "review_id": req.review_id, "status": "resolved"}
        tm, qs = svc.resolve_taxonomy(req.taxonomy_id)
        q = next((q for q in qs if q.name == req.question), None)
        if q is None:
            raise HTTPException(404, f"taxonomy '{req.taxonomy_id}' has no question '{req.question}'")
        _validate_value(q.type, q.option_names, req.value)
        t = req.transaction.as_dict()
        p = prepare(t["text"], t.get("amount"), t.get("direction"), t.get("history"))
        cid = svc.store.add_correction(q.name, req.value, req.taxonomy_id, correction_key(tm.id, q, p), p.scrubbed,
                                       rendered_text=p.rendered)
        return {"id": cid, "status": "recorded"}

    @app.get("/v1/corrections/export", dependencies=[Depends(auth)])
    def export_corrections():
        return {"corrections": svc.store.export_corrections()}

    @app.get("/v1/models", dependencies=[Depends(auth)])
    def models():
        if svc.registry is None:
            return {"current": svc.model_version, "versions": {}}
        return svc.registry.list()

    @app.post("/v1/models/rollback", dependencies=[Depends(auth)])
    def rollback():
        if svc.registry is None:
            raise HTTPException(409, "no registry configured")
        try:
            v = svc.registry.rollback()
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        svc.set_bundle(svc.registry.load(v))
        return {"current": f"student:{v}"}

    @app.post("/v1/models/promote", dependencies=[Depends(auth)])
    def promote(req: PromoteIn):
        if svc.registry is None:
            raise HTTPException(409, "no registry configured")
        try:
            v = svc.registry.promote(req.version)
        except KeyError:
            raise HTTPException(404, f"unknown model version '{req.version}'")
        svc.set_bundle(svc.registry.load(v))
        return {"current": f"student:{v}"}

    return app


def _validate_value(qtype: str, options: list[str] | None, value) -> None:
    if qtype == "bool" and not isinstance(value, bool):
        raise HTTPException(422, "bool questions take true/false")
    if qtype == "choice" and value not in (options or []):
        raise HTTPException(422, f"{value!r} is not an option")


def __getattr__(name: str):
    # ``uvicorn arth.api:app`` builds the app lazily from the environment.
    if name == "app":
        return create_app()
    raise AttributeError(name)
