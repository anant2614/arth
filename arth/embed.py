"""Text embedders.

- ``hash``: deterministic char + word n-gram hashing. No download, no fitting, fast.
  Default for tests and for environments without model weights.
- ``bge-small``: BAAI bge-small-en-v1.5, quantised ONNX on CPU (PRD Tier 1a). Needs the
  ``onnx`` extra and either a local copy or Hugging Face Hub access.
- ``hash+bge-small``: both, concatenated and re-normalised.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Protocol

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

Matrix = np.ndarray | sp.csr_matrix


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> Matrix: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


class HashingEmbedder:
    """Returns sparse CSR rows; everything downstream accepts sparse or dense."""

    def __init__(self, dim: int = 16384):
        half = dim // 2
        self.name = f"hash{dim}"
        self.dim = dim
        self._char = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=half,
                                       alternate_sign=True, norm="l2", lowercase=True)
        self._word = HashingVectorizer(analyzer="word", ngram_range=(1, 2), n_features=half,
                                       alternate_sign=True, norm="l2", lowercase=True,
                                       token_pattern=r"(?u)\b\w[\w<>]*\b|<\w+>")

    def encode(self, texts: list[str]) -> sp.csr_matrix:
        if not texts:
            return sp.csr_matrix((0, self.dim), dtype=np.float32)
        c = self._char.transform(texts)
        w = self._word.transform(texts)
        return normalize(sp.hstack([c, w], format="csr").astype(np.float32))


class OnnxBgeEmbedder:
    """bge-small-en-v1.5 (33M params) via ONNX Runtime, CLS pooling."""

    REPO = "Xenova/bge-small-en-v1.5"

    def __init__(self, model_dir: str | None = None, quantized: bool = True, max_len: int = 96):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = model_dir or os.environ.get("ARTH_BGE_DIR")
        fname = "onnx/model_quantized.onnx" if quantized else "onnx/model.onnx"
        if model_dir and os.path.exists(os.path.join(model_dir, fname)):
            model_path = os.path.join(model_dir, fname)
            tok_path = os.path.join(model_dir, "tokenizer.json")
        else:
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(self.REPO, fname)
            tok_path = hf_hub_download(self.REPO, "tokenizer.json")
        self.tok = Tokenizer.from_file(tok_path)
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ARTH_ORT_THREADS", "1"))
        self.sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self.sess.get_inputs()}
        self.name = "bge-small"
        self.dim = 384

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch(texts[i:i + batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self.sess.run(None, feeds)[0]
            out.append(hidden[:, 0, :])
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2(np.vstack(out))


class ConcatEmbedder:
    def __init__(self, parts: list[Embedder], weights: list[float] | None = None):
        self.parts = parts
        self.weights = weights or [1.0] * len(parts)
        self.name = "+".join(p.name for p in parts)
        self.dim = sum(p.dim for p in parts)

    def encode(self, texts: list[str]) -> sp.csr_matrix:
        mats = [sp.csr_matrix(p.encode(texts)) * w for p, w in zip(self.parts, self.weights)]
        return normalize(sp.hstack(mats, format="csr").astype(np.float32))


@lru_cache(maxsize=4)
def get_embedder(spec: str = "hash") -> Embedder:
    parts: list[Embedder] = []
    for s in spec.split("+"):
        if s == "hash":
            parts.append(HashingEmbedder())
        elif s == "bge-small":
            parts.append(OnnxBgeEmbedder())
        else:
            raise ValueError(f"unknown embedder {s!r}")
    return parts[0] if len(parts) == 1 else ConcatEmbedder(parts)
