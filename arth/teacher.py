"""Tier 2: the open-licensed LLM teacher (fallback, audit, and offline labelling).

``HFTeacher`` calls an open-weight model (default Qwen3-235B-A22B-Instruct, Apache-2.0)
through Hugging Face Inference Providers' OpenAI-compatible router with constrained
JSON output. Probabilities come from token log-probabilities when the provider
returns them (options are coded A, B, C... so one token decides the answer), else from
3-5 sampled votes. Every input is scrubbed again here as defence in depth: the teacher
is the only component that sends text outside our process.
"""

from __future__ import annotations

import json
import math
import os
import string
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np

from .calibration import fit_temperature, softmax
from .questions import Question
from .scrub import scrub

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
CODES = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
CODE_INDEX = {c: i for i, c in enumerate(CODES)}
LOGPROB_MAX_OPTIONS = 26  # one-token codes


@dataclass
class TeacherInput:
    scrubbed_text: str
    direction: str | None = None
    amount: float | None = None
    channel: str | None = None
    merchant: str | None = None
    history: dict | None = None


class Teacher(Protocol):
    version: str

    def answer(self, items: list[TeacherInput], questions: list[Question]) -> list[dict[str, np.ndarray]]:
        """Per item, per question: probability vector (choice: option order; bool: [no, yes])."""


SYSTEM_PROMPT = (
    "You categorise Indian bank transactions: bank SMS alerts, statement narrations, UPI strings "
    "and Account Aggregator descriptions. Personal data has been replaced with placeholders such as "
    "<NAME>, <VPA>, <ACCT>, <NUM>. Use the narration, direction, amount, channel, known merchant and "
    "the account history (how often this counterparty recurs, how stable the amount is, the median gap "
    "in days). Answer every question with exactly one option code. Do not give financial advice."
)


def _question_block(q: Question) -> str:
    if q.type == "bool":
        return f'"{q.name}": yes/no question - {q.bool_text()}? Answer "yes" or "no".'
    lines = [f'"{q.name}": choose one option code:']
    for code, (opt, desc) in zip(CODES, q.options):
        lines.append(f"  {code}) {opt}" + (f" - {desc}" if desc else ""))
    return "\n".join(lines)


def _user_prompt(item: TeacherInput, questions: list[Question]) -> str:
    ctx = {
        "narration": item.scrubbed_text,
        "direction": item.direction,
        "amount_inr": item.amount,
        "channel": item.channel,
        "known_merchant": item.merchant,
        "history": item.history,
    }
    qs = "\n".join(_question_block(q) for q in questions)
    return f"Transaction:\n{json.dumps(ctx, ensure_ascii=False)}\n\nQuestions:\n{qs}\n\nReply with a JSON object mapping each question name to its answer code."


def _schema(questions: list[Question]) -> dict:
    props = {}
    for q in questions:
        enum = ["yes", "no"] if q.type == "bool" else CODES[: len(q.options)]
        props[q.name] = {"type": "string", "enum": enum}
    return {"type": "object", "properties": props, "required": [q.name for q in questions],
            "additionalProperties": False}


def _code_index(q: Question, code: str) -> int | None:
    code = code.strip().strip('"').strip()
    if q.type == "bool":
        c = code.lower()
        return 1 if c.startswith("y") else 0 if c.startswith("n") else None
    i = CODE_INDEX.get(code.upper())
    return i if i is not None and i < len(q.options) else None


class HFTeacher:
    def __init__(self, model: str | None = None, token: str | None = None, samples: int = 5,
                 use_logprobs: bool = True, timeout: float = 60.0, max_workers: int = 4,
                 client: httpx.Client | None = None, temperature_calibration: float = 1.0):
        self.model = model or os.environ.get("ARTH_TEACHER_MODEL", DEFAULT_TEACHER_MODEL)
        self.token = token or os.environ.get("HF_TOKEN")
        if not self.token:
            raise RuntimeError("HF_TOKEN is not set; the teacher needs Hugging Face Inference Providers access")
        self.samples = samples
        self.use_logprobs = use_logprobs
        self.client = client or httpx.Client(timeout=timeout)
        self.max_workers = max_workers
        self.temperature_calibration = temperature_calibration
        self.version = f"teacher:{self.model}"

    def _post(self, body: dict) -> dict:
        r = self.client.post(HF_ROUTER_URL, json=body, headers={"Authorization": f"Bearer {self.token}"})
        r.raise_for_status()
        return r.json()

    def _body(self, item: TeacherInput, questions: list[Question], temperature: float, logprobs: bool) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": _user_prompt(item, questions)}],
            "temperature": temperature,
            "max_tokens": 40 + 12 * len(questions),
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "answers", "schema": _schema(questions), "strict": True}},
        }
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 10
        return body

    @staticmethod
    def _from_logprobs(resp: dict, questions: list[Question]) -> dict[str, np.ndarray] | None:
        choice = resp["choices"][0]
        content = (choice.get("logprobs") or {}).get("content")
        if not content:
            return None
        text, offsets = "", []
        for tok in content:
            offsets.append(len(text))
            text += tok["token"]
        out = {}
        for q in questions:
            key = f'"{q.name}"'
            k = text.find(key)
            if k < 0:
                return None
            v = text.find('"', text.find(":", k + len(key)) + 1)
            if v < 0:
                return None
            # first token that contains the first character of the value
            pos = next((i for i, o in enumerate(offsets) if o + len(content[i]["token"]) > v + 1), None)
            if pos is None:
                return None
            n = 2 if q.type == "bool" else len(q.options)
            probs = np.zeros(n)
            for alt in content[pos].get("top_logprobs", []):
                idx = _code_index(q, alt["token"])
                if idx is not None:
                    probs[idx] += math.exp(alt["logprob"])
            if probs.sum() < 0.5:
                return None
            out[q.name] = probs / probs.sum()
        return out

    def _votes(self, item: TeacherInput, questions: list[Question]) -> dict[str, np.ndarray]:
        counts = {q.name: Counter() for q in questions}
        for _ in range(self.samples):
            resp = self._post(self._body(item, questions, temperature=0.8, logprobs=False))
            try:
                ans = json.loads(resp["choices"][0]["message"]["content"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            for q in questions:
                idx = _code_index(q, str(ans.get(q.name, "")))
                if idx is not None:
                    counts[q.name][idx] += 1
        out = {}
        for q in questions:
            n = 2 if q.type == "bool" else len(q.options)
            p = np.full(n, 0.25)  # light Laplace smoothing
            for idx, c in counts[q.name].items():
                p[idx] += c
            out[q.name] = p / p.sum()
        return out

    def _one(self, item: TeacherInput, questions: list[Question]) -> dict[str, np.ndarray]:
        item = TeacherInput(**{**item.__dict__, "scrubbed_text": scrub(item.scrubbed_text).text})
        res = None
        if self.use_logprobs and all(q.type == "bool" or len(q.options) <= LOGPROB_MAX_OPTIONS for q in questions):
            try:
                res = self._from_logprobs(self._post(self._body(item, questions, 0.0, True)), questions)
            except httpx.HTTPStatusError:
                res = None
        if res is None:
            res = self._votes(item, questions)
        if self.temperature_calibration != 1.0:
            res = {k: softmax(np.log(np.clip(v, 1e-9, 1)), self.temperature_calibration) for k, v in res.items()}
        return res

    def answer(self, items: list[TeacherInput], questions: list[Question]) -> list[dict[str, np.ndarray]]:
        with ThreadPoolExecutor(self.max_workers) as ex:
            return list(ex.map(lambda it: self._one(it, questions), items))


def builtin_label_index(q: Question, concept: str) -> int | None:
    """Label index of ``concept`` for a question that matches a built-in taxonomy or bool."""
    from .concepts import BUILTIN_BOOLS, BUILTIN_TAXONOMIES

    if q.type == "bool":
        for b in BUILTIN_BOOLS.values():
            if b.name == q.name or b.description == q.bool_text():
                return int(b.label(concept))
        return None
    for t in BUILTIN_TAXONOMIES.values():
        if set(q.option_names) == set(t.options):
            return q.option_names.index(t.label(concept))
    return None


class OracleTeacher:
    """Simulated teacher for tests and offline drills: knows each line's concept.

    Lines are keyed by scrubbed text. ``accuracy`` < 1 makes it wrong on a deterministic
    subset of lines, so audit agreement monitoring can be exercised without network access.
    """

    def __init__(self, concept_of: dict[str, str], accuracy: float = 1.0, confidence: float = 0.97,
                 label_fn=builtin_label_index):
        self.concept_of = concept_of
        self.accuracy = accuracy
        self.confidence = confidence
        self.label_fn = label_fn
        self.version = f"teacher:oracle@{accuracy}"
        self.calls = 0

    def answer(self, items: list[TeacherInput], questions: list[Question]) -> list[dict[str, np.ndarray]]:
        import zlib

        self.calls += len(items)
        out = []
        for it in items:
            concept = self.concept_of.get(it.scrubbed_text)
            wrong = (zlib.crc32(it.scrubbed_text.encode()) % 1000) / 1000 >= self.accuracy
            res = {}
            for q in questions:
                n = 2 if q.type == "bool" else len(q.options)
                target = self.label_fn(q, concept) if concept else None
                if target is None:
                    res[q.name] = np.full(n, 1.0 / n)
                    continue
                p = np.full(n, (1 - self.confidence) / max(1, n - 1))
                p[(target + 1) % n if wrong else target] = self.confidence
                res[q.name] = p
            out.append(res)
        return out


def fit_teacher_temperature(probs: list[np.ndarray], labels: list[int]) -> float:
    """Teacher calibration head: one temperature over log-probabilities (PRD step 5)."""
    if not probs:
        return 1.0
    k = max(len(p) for p in probs)
    L = np.full((len(probs), k), -30.0)
    for i, p in enumerate(probs):
        L[i, : len(p)] = np.log(np.clip(p, 1e-9, 1.0))
    return fit_temperature(L, np.asarray(labels))
