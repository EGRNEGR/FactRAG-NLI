"""NLI classifiers for the verifier experiments, behind the verifier's NLIBackend interface.

Every model is wrapped by `CachedNLI`: predictions are memoized in SQLite per model, keyed by
the normalized (premise, hypothesis) pair, so threshold and aggregation sweeps re-run the
decision logic only. Inputs are normalized exactly as `generator_verifier.TransformersNLI`
does (document header removed, whitespace collapsed); an over-long pair returns None, never
a truncated premise (same contract as the production backend).

Two-way models (entailment / not_entailment, e.g. bge-m3-zeroshot-v2.0) report the
not-entailment mass as neutral; their contradiction probability is 0.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from generator_verifier import NLIProbabilities

from research.opt2 import data

_HEADER = re.compile(r"^Документ:[^\n]*\nРаздел:[^\n]*\n")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    max_length: int = 512
    batch_size: int = 16


MODELS_DIR = data.ROOT / "models"
SPECS = {
    "mdeberta": ModelSpec("mdeberta", MODELS_DIR / "mdeberta-nli"),
    "bge-m3-zeroshot": ModelSpec("bge-m3-zeroshot", MODELS_DIR / "nli-bge-m3-zeroshot", 1024, 8),
    "xlmr-large-xnli": ModelSpec("xlmr-large-xnli", MODELS_DIR / "nli-xlmr-large-xnli", 512, 8),
    "xlmr-large-xnli-anli": ModelSpec(
        "xlmr-large-xnli-anli", MODELS_DIR / "nli-xlmr-large-xnli-anli", 512, 8
    ),
}


def normalize_pair(premise: str, hypothesis: str) -> tuple[str, str]:
    return " ".join(_HEADER.sub("", premise).split()), " ".join(hypothesis.split())


class HFClassifier:
    """A local sequence-classification NLI checkpoint in float16 on CUDA."""

    def __init__(self, spec: ModelSpec) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(str(spec.path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(spec.path), local_files_only=True
        )
        labels = {str(v).lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail = next(i for name, i in labels.items() if name.startswith("entail"))
        self.contra = next((i for name, i in labels.items() if name.startswith("contra")), None)
        self.neutral = next((i for name, i in labels.items() if name.startswith("neutral")), None)
        self.not_entail = next((i for name, i in labels.items() if name.startswith("not")), None)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            raise RuntimeError("NLI experiments require CUDA; refusing a silent CPU run")
        self.model.to(device=self.device, dtype=torch.float16).eval()

    def lengths(self, pairs: Sequence[tuple[str, str]]) -> list[int]:
        return [
            len(self.tokenizer(p, h, truncation=False, verbose=False)["input_ids"])
            for p, h in pairs
        ]

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NLIProbabilities | None]:
        import torch

        lengths = self.lengths(pairs)
        order = sorted(
            (i for i, n in enumerate(lengths) if n <= self.spec.max_length),
            key=lambda i: lengths[i],
        )
        result: list[NLIProbabilities | None] = [None] * len(pairs)
        for start in range(0, len(order), self.spec.batch_size):
            batch = order[start : start + self.spec.batch_size]
            encoded = self.tokenizer(
                [pairs[i][0] for i in batch],
                [pairs[i][1] for i in batch],
                truncation=False,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(self.model(**encoded).logits.float(), dim=-1).cpu().tolist()
            for i, row in zip(batch, probs, strict=True):
                entail = row[self.entail]
                contra = row[self.contra] if self.contra is not None else 0.0
                result[i] = NLIProbabilities(
                    p_entailment=entail,
                    p_neutral=max(0.0, 1.0 - entail - contra),
                    p_contradiction=contra,
                )
        return result


class CachedNLI:
    """NLIBackend with a persistent per-model prediction cache (thread-safe)."""

    def __init__(self, name: str, *, load: bool = True) -> None:
        self.spec = SPECS[name]
        self.name = name
        path = data.CACHE / "nli_cache" / f"{name}.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS p (k TEXT PRIMARY KEY, e REAL, n REAL, c REAL, ok INTEGER)"
        )
        self._lock = threading.Lock()
        self._model: HFClassifier | None = None
        self._load = load
        self.last_premise: str | None = None
        self.misses = 0

    @staticmethod
    def key(premise: str, hypothesis: str) -> str:
        return hashlib.sha1(f"{premise}\x00{hypothesis}".encode()).hexdigest()

    def _classifier(self) -> HFClassifier:
        if self._model is None:
            if not self._load:
                raise KeyError(f"{self.name}: pair missing from the cache and loading is off")
            self._model = HFClassifier(self.spec)
        return self._model

    def predict_many(self, pairs: Sequence[tuple[str, str]]) -> list[NLIProbabilities | None]:
        normalized = [normalize_pair(p, h) for p, h in pairs]
        keys = [self.key(p, h) for p, h in normalized]
        found: dict[str, NLIProbabilities | None] = {}
        with self._lock:
            for start in range(0, len(keys), 500):
                chunk = keys[start : start + 500]
                rows = self._db.execute(
                    f"SELECT k, e, n, c, ok FROM p WHERE k IN ({','.join('?' * len(chunk))})", chunk
                ).fetchall()
                for k, e, n, c, ok in rows:
                    found[k] = (
                        NLIProbabilities(p_entailment=e, p_neutral=n, p_contradiction=c)
                        if ok
                        else None
                    )
            missing = list(dict.fromkeys(i for i, k in enumerate(keys) if k not in found))
            unique: dict[str, int] = {}
            for i in missing:
                unique.setdefault(keys[i], i)
            if unique:
                self.misses += len(unique)
                todo = list(unique.values())
                predicted = self._classifier().predict([normalized[i] for i in todo])
                rows = []
                for i, p in zip(todo, predicted, strict=True):
                    found[keys[i]] = p
                    rows.append(
                        (keys[i], p.p_entailment, p.p_neutral, p.p_contradiction, 1)
                        if p
                        else (keys[i], 0.0, 0.0, 0.0, 0)
                    )
                self._db.executemany("INSERT OR REPLACE INTO p VALUES (?, ?, ?, ?, ?)", rows)
                self._db.commit()
        return [found[k] for k in keys]

    def predict(self, premise: str, hypothesis: str) -> NLIProbabilities | None:
        self.last_premise = normalize_pair(premise, hypothesis)[0]
        return self.predict_many([(premise, hypothesis)])[0]

    def close(self) -> None:
        self._db.close()
