"""SQLite persistence: saved taxonomies, review queue (FR-7), corrections (FR-8).

Only scrubbed text is ever written. Identifiers derived from raw text (line ids,
correction keys) are keyed hashes. Scrubbed text in the review queue and in
corrections is deleted after the retention window unless the taxonomy opted in
to training.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import threading
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS taxonomies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    questions TEXT NOT NULL,
    sample TEXT NOT NULL,
    opt_in_training INTEGER NOT NULL DEFAULT 0,
    model BLOB,
    model_version TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS review_items (
    id TEXT PRIMARY KEY,
    line_hash TEXT NOT NULL,
    taxonomy_id TEXT,
    question TEXT NOT NULL,
    question_spec TEXT NOT NULL,
    scrubbed_text TEXT,
    rendered_text TEXT,
    top3 TEXT NOT NULL,
    value TEXT NOT NULL,
    route TEXT NOT NULL,
    correction_key TEXT,
    model_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    corrected_value TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS review_status ON review_items(status, taxonomy_id, created_at);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id TEXT,
    taxonomy_id TEXT,
    question TEXT NOT NULL,
    correction_key TEXT,
    value TEXT NOT NULL,
    scrubbed_text TEXT,
    rendered_text TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS corrections_key ON corrections(correction_key);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    def _exec(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, args)
            rows = cur.fetchall()
            self._conn.commit()
            return rows

    # ---- taxonomies -----------------------------------------------------
    def save_taxonomy(self, tax_id: str | None, name: str, questions: list[dict], sample: list[dict],
                      model, model_version: str, opt_in_training: bool = False) -> str:
        tax_id = tax_id or f"tax_{uuid.uuid4().hex[:12]}"
        now = _now()
        blob = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        self._exec(
            "INSERT INTO taxonomies(id, name, questions, sample, opt_in_training, model, model_version, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name, questions=excluded.questions, "
            "sample=excluded.sample, opt_in_training=excluded.opt_in_training, model=excluded.model, "
            "model_version=excluded.model_version, updated_at=excluded.updated_at",
            (tax_id, name, json.dumps(questions), json.dumps(sample), int(opt_in_training), blob, model_version, now, now),
        )
        return tax_id

    def get_taxonomy(self, tax_id: str) -> dict | None:
        rows = self._exec("SELECT * FROM taxonomies WHERE id=?", (tax_id,))
        if not rows:
            return None
        r = rows[0]
        return {"id": r["id"], "name": r["name"], "questions": json.loads(r["questions"]),
                "sample": json.loads(r["sample"]), "opt_in_training": bool(r["opt_in_training"]),
                "model": pickle.loads(r["model"]) if r["model"] else None, "model_version": r["model_version"],
                "created_at": r["created_at"], "updated_at": r["updated_at"]}

    def list_taxonomies(self) -> list[dict]:
        rows = self._exec("SELECT id, name, model_version, created_at, updated_at FROM taxonomies ORDER BY created_at")
        return [dict(r) for r in rows]

    # ---- review queue ---------------------------------------------------
    def enqueue(self, items: list[dict]) -> list[str]:
        ids = []
        now = _now()
        with self._lock:
            for it in items:
                rid = f"rev_{uuid.uuid4().hex[:16]}"
                self._conn.execute(
                    "INSERT INTO review_items(id, line_hash, taxonomy_id, question, question_spec, scrubbed_text, "
                    "rendered_text, top3, value, route, correction_key, model_version, created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, it["line_hash"], it.get("taxonomy_id"), it["question"], json.dumps(it["question_spec"]),
                     it["scrubbed_text"], it.get("rendered_text"), json.dumps(it["top3"]), json.dumps(it["value"]), it["route"],
                     it.get("correction_key"), it["model_version"], now),
                )
                ids.append(rid)
            self._conn.commit()
        return ids

    def review_queue(self, taxonomy_id: str | None = None, status: str = "open", limit: int = 100) -> list[dict]:
        sql = "SELECT * FROM review_items WHERE status=?"
        args: list = [status]
        if taxonomy_id is not None:
            sql += " AND taxonomy_id=?"
            args.append(taxonomy_id)
        sql += " ORDER BY created_at LIMIT ?"
        args.append(limit)
        return [self._review_row(r) for r in self._exec(sql, tuple(args))]

    def get_review(self, review_id: str) -> dict | None:
        rows = self._exec("SELECT * FROM review_items WHERE id=?", (review_id,))
        return self._review_row(rows[0]) if rows else None

    @staticmethod
    def _review_row(r: sqlite3.Row) -> dict:
        return {"id": r["id"], "line_hash": r["line_hash"], "taxonomy_id": r["taxonomy_id"], "question": r["question"],
                "question_spec": json.loads(r["question_spec"]), "scrubbed_text": r["scrubbed_text"],
                "rendered_text": r["rendered_text"],
                "top3": json.loads(r["top3"]), "value": json.loads(r["value"]), "route": r["route"],
                "model_version": r["model_version"], "status": r["status"],
                "corrected_value": json.loads(r["corrected_value"]) if r["corrected_value"] else None,
                "correction_key": r["correction_key"], "created_at": r["created_at"]}

    # ---- corrections ----------------------------------------------------
    def add_correction(self, question: str, value, taxonomy_id: str | None, correction_key: str | None,
                       scrubbed_text: str | None, review_id: str | None = None, rendered_text: str | None = None) -> int:
        now = _now()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO corrections(review_id, taxonomy_id, question, correction_key, value, scrubbed_text, "
                "rendered_text, created_at) VALUES(?,?,?,?,?,?,?,?)",
                (review_id, taxonomy_id, question, correction_key, json.dumps(value), scrubbed_text, rendered_text, now),
            )
            if review_id:
                self._conn.execute("UPDATE review_items SET status='resolved', corrected_value=?, resolved_at=? WHERE id=?",
                                   (json.dumps(value), now, review_id))
            self._conn.commit()
            return int(cur.lastrowid)

    def lookup_corrections(self, keys: list[str]) -> dict[str, object]:
        """Majority reviewer label per correction key; ties go to the most recent."""
        if not keys:
            return {}
        out = {}
        qmarks = ",".join("?" * len(set(keys)))
        rows = self._exec(f"SELECT correction_key, value FROM corrections WHERE correction_key IN ({qmarks}) "
                          f"ORDER BY id", tuple(set(keys)))
        votes: dict[str, Counter] = {}
        last: dict[str, str] = {}
        for r in rows:
            votes.setdefault(r["correction_key"], Counter())[r["value"]] += 1
            last[r["correction_key"]] = r["value"]
        for k, c in votes.items():
            top = max(c.values())
            winners = [v for v, n in c.items() if n == top]
            out[k] = json.loads(last[k] if last[k] in winners else winners[0])
        return out

    def corrections_for(self, taxonomy_id: str) -> dict[str, list[tuple[str, object]]]:
        """Per question: (rendered student input, reviewer label) pairs still within retention."""
        rows = self._exec("SELECT question, value, rendered_text FROM corrections WHERE taxonomy_id=? "
                          "AND rendered_text IS NOT NULL", (taxonomy_id,))
        out: dict[str, list] = {}
        for r in rows:
            out.setdefault(r["question"], []).append((r["rendered_text"], json.loads(r["value"])))
        return out

    def export_corrections(self) -> list[dict]:
        rows = self._exec("SELECT c.*, t.opt_in_training FROM corrections c LEFT JOIN taxonomies t ON t.id = c.taxonomy_id "
                          "WHERE c.scrubbed_text IS NOT NULL ORDER BY c.id")
        return [{"taxonomy_id": r["taxonomy_id"], "question": r["question"], "value": json.loads(r["value"]),
                 "scrubbed_text": r["scrubbed_text"], "created_at": r["created_at"],
                 "opt_in_training": bool(r["opt_in_training"])} for r in rows]

    # ---- retention ------------------------------------------------------
    def purge_expired(self, days: int = 30, now: datetime | None = None) -> dict:
        cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=days)).isoformat()
        opted = "SELECT id FROM taxonomies WHERE opt_in_training=1"
        with self._lock:
            a = self._conn.execute(
                f"UPDATE review_items SET scrubbed_text=NULL, rendered_text=NULL WHERE created_at < ? "
                f"AND scrubbed_text IS NOT NULL "
                f"AND (taxonomy_id IS NULL OR taxonomy_id NOT IN ({opted}))", (cutoff,)).rowcount
            b = self._conn.execute(
                f"UPDATE corrections SET scrubbed_text=NULL, rendered_text=NULL WHERE created_at < ? "
                f"AND scrubbed_text IS NOT NULL "
                f"AND (taxonomy_id IS NULL OR taxonomy_id NOT IN ({opted}))", (cutoff,)).rowcount
            self._conn.commit()
        return {"review_items_purged": a, "corrections_purged": b}
