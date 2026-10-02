"""FR-11: versioned student bundles (student:vN) with one-step rollback."""

from __future__ import annotations

import gzip
import json
import pickle
import threading
from datetime import datetime, timezone
from pathlib import Path

from .engine import StudentBundle


class Registry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index = self.root / "registry.json"
        self._lock = threading.Lock()

    def _read(self) -> dict:
        if self._index.exists():
            return json.loads(self._index.read_text())
        return {"current": None, "history": [], "versions": {}}

    def _write(self, idx: dict) -> None:
        tmp = self._index.with_suffix(".tmp")
        tmp.write_text(json.dumps(idx, indent=2))
        tmp.replace(self._index)

    def next_version(self) -> str:
        return f"v{len(self._read()['versions']) + 1}"

    def save(self, bundle: StudentBundle, promote: bool = True) -> str:
        with self._lock:
            idx = self._read()
            d = self.root / bundle.version
            d.mkdir(parents=True, exist_ok=True)
            with gzip.open(d / "bundle.pkl.gz", "wb") as f:
                pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
            manifest = {**bundle.manifest, "version": bundle.version,
                        "saved_at": datetime.now(timezone.utc).isoformat()}
            (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
            idx["versions"][bundle.version] = manifest
            if promote:
                idx["history"] = [v for v in idx["history"] if v != bundle.version] + [bundle.version]
                idx["current"] = bundle.version
            self._write(idx)
            return bundle.version

    def load(self, version: str | None = None) -> StudentBundle:
        version = version or self.current()
        if version is None:
            raise FileNotFoundError(f"no model registered under {self.root}")
        with gzip.open(self.root / version / "bundle.pkl.gz", "rb") as f:
            return pickle.load(f)

    def current(self) -> str | None:
        return self._read()["current"]

    def list(self) -> dict:
        return self._read()

    def promote(self, version: str) -> str:
        with self._lock:
            idx = self._read()
            if version not in idx["versions"]:
                raise KeyError(version)
            idx["history"] = [v for v in idx["history"] if v != version] + [version]
            idx["current"] = version
            self._write(idx)
            return version

    def rollback(self) -> str:
        """Make the previously promoted version current again."""
        with self._lock:
            idx = self._read()
            if len(idx["history"]) < 2:
                raise RuntimeError("nothing to roll back to")
            idx["history"].pop()
            idx["current"] = idx["history"][-1]
            self._write(idx)
            return idx["current"]

    def attach_report(self, version: str, report: dict) -> None:
        d = self.root / version
        (d / "eval_report.json").write_text(json.dumps(report, indent=2, default=str))
        with self._lock:
            idx = self._read()
            idx["versions"][version]["eval_report"] = str(d / "eval_report.json")
            idx["versions"][version]["gates_passed"] = report.get("summary", {}).get("all_gates_passed")
            self._write(idx)
