"""Claim-level verifiers compared with the decision layer of FactRAG-NLI (remark 1).

Every variant keeps the run's answer assembly: claim segmentation, the citation requirement,
absence statements, and the refusal rule (no accepted claim, or more than half of the fact
claims withheld). Only the decision "does the cited source support this claim" differs:

  nli_single   one mDeBERTa-v3 NLI pass over the whole cited chunk, P(e) > 0.40 and
               P(c) < 0.20 (SummaC-style single filter, the rule of the early verifier):
               no sentence windows, no judge, no anchor rule;
  bge_zs       BGE-M3-zeroshot alone: the best entailment over the chunk and its sentence
               windows (an AlignScore-like max over windows) >= threshold, no anchor rule;
  minicheck    MiniCheck-Flan-T5-Large (Tang et al., 2024), official inference: input
               "predict: <chunk></s><claim>", support = softmax over tokens 3 / 209, max over
               500-word chunks, >= threshold (0.5 by default);
  judge_free   the decision layer of the run without the hard anchor rule;
  judge        the decision layer of the run (ours): judge + anchor guard + co-verification.

A claim citing several sources is supported if one source supports it. The baselines do not
use co-verification (a lexical rule of FactRAG-NLI that relies on its own verified claims).
Scores are cached (research.opt2.nli_models for the NLI models, SQLite for MiniCheck), so a
threshold sweep re-runs the decision only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from generator_verifier import (
    ClaimJudge,
    ClaimStatus,
    NLIFactVerifier,
    SourceContext,
    VerifiedClaim,
    _grade,
)
from grounding import segment_claims

from research.review import CACHE, ROOT  # noqa: I001 - selects the run first
from research.opt2.nli_models import CachedNLI, normalize_pair
from research.opt3 import data
from research.opt3.replay import verifier as run_verifier

MINICHECK_PATH = ROOT / "models" / "minicheck-flan-t5-large"
NAMES = ("nli_single", "bge_zs", "minicheck", "judge_free", "judge")
DEFAULT_THRESHOLDS = {"bge_zs": 0.5, "minicheck": 0.5}


class MiniCheck:
    """MiniCheck-Flan-T5-Large with a persistent score cache (official inference recipe)."""

    CHUNK_WORDS = 500
    MAX_LENGTH = 2048

    def __init__(self, path: Path = MINICHECK_PATH, batch_size: int = 8) -> None:
        self.path = path
        self.batch_size = batch_size
        db = CACHE / "minicheck.sqlite"
        db.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS p (k TEXT PRIMARY KEY, s REAL)")
        self._lock = threading.Lock()
        self._model: Any = None
        self.misses = 0

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if os.environ.get("REVIEW_CACHE_ONLY"):
            raise RuntimeError("REVIEW_CACHE_ONLY: a MiniCheck score is missing from the cache")
        if not torch.cuda.is_available():
            raise RuntimeError("MiniCheck scoring requires CUDA; refusing a silent CPU run")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
        self._model = (
            AutoModelForSeq2SeqLM.from_pretrained(str(self.path), local_files_only=True)
            .to("cuda", dtype=torch.bfloat16)
            .eval()
        )

    @classmethod
    def chunks(cls, doc: str) -> list[str]:
        """Sentences packed into chunks of at most CHUNK_WORDS words (MiniCheck's splitting)."""
        result: list[str] = []
        current: list[str] = []
        count = 0
        for sentence in segment_claims(doc, "v2") or [doc]:
            words = len(sentence.split())
            if current and count + words > cls.CHUNK_WORDS:
                result.append(" ".join(current))
                current, count = [], 0
            current.append(sentence)
            count += words
        if current:
            result.append(" ".join(current))
        return [c for c in result if c.strip()] or [doc]

    @staticmethod
    def key(chunk: str, claim: str) -> str:
        return hashlib.sha1(f"minicheck\x00{chunk}\x00{claim}".encode()).hexdigest()

    def _predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        import torch

        if self._model is None:
            self._load()
        eos = self.tokenizer.eos_token
        result: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = self.tokenizer(
                [f"predict: {chunk}{eos}{claim}" for chunk, claim in batch],
                max_length=self.MAX_LENGTH,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            decoder = torch.zeros((len(batch), 1), dtype=torch.long, device="cuda")
            with torch.inference_mode():
                logits = self._model(**encoded, decoder_input_ids=decoder).logits.squeeze(1)
            probs = torch.softmax(logits[:, [3, 209]].float(), dim=-1)[:, 1]
            result.extend(probs.cpu().tolist())
        return result

    def score_many(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Support probability per (document, claim): max over the document's chunks."""
        expanded = [
            [normalize_pair(chunk, claim) for chunk in self.chunks(doc)] for doc, claim in pairs
        ]
        flat = [pair for group in expanded for pair in group]
        keys = [self.key(c, h) for c, h in flat]
        with self._lock:
            found: dict[str, float] = {}
            for start in range(0, len(keys), 500):
                part = keys[start : start + 500]
                rows = self._db.execute(
                    f"SELECT k, s FROM p WHERE k IN ({','.join('?' * len(part))})", part
                ).fetchall()
                found.update(rows)
            todo = list(dict.fromkeys(i for i, k in enumerate(keys) if k not in found))
            unique: dict[str, int] = {}
            for i in todo:
                unique.setdefault(keys[i], i)
            index = list(unique.values())
            self.misses += len(index)
            # Committed in parts: an interrupted run resumes from the cache.
            for start in range(0, len(index), 256):
                batch = index[start : start + 256]
                scores = self._predict([flat[i] for i in batch])
                rows = [(keys[i], s) for i, s in zip(batch, scores, strict=True)]
                self._db.executemany("INSERT OR REPLACE INTO p VALUES (?, ?)", rows)
                self._db.commit()
                found.update(rows)
        out: list[float] = []
        position = 0
        for group in expanded:
            out.append(max(found[keys[position + j]] for j in range(len(group))))
            position += len(group)
        return out


Decision = Callable[["ClaimFilter", str, SourceContext], tuple[float, bool]]


class ClaimFilter(NLIFactVerifier):
    """The run's answer assembly with another claim-level support decision (a baseline)."""

    def __init__(self, name: str, decide: Decision, *, anchors: bool = False) -> None:
        base = data.settings("judge")
        assert base.nli_judge_path is not None
        spec = json.loads(Path(base.nli_judge_path).read_text("utf-8"))
        super().__init__(
            base.model_copy(update={"nli_co_verification": False}),
            backend=CachedNLI("mdeberta"),
            secondary=CachedNLI("bge-m3-zeroshot"),
            judge=ClaimJudge(spec),
        )
        self.name = name
        self.decide = decide
        self.anchors = anchors

    def _judge_claim(
        self,
        text: str,
        hypothesis: str,
        source_ids: tuple[str, ...],
        contexts: Mapping[str, SourceContext],
        is_inherited: bool,
    ) -> VerifiedClaim:
        results = [self.decide(self, hypothesis, contexts[key]) for key in source_ids]
        anchored = not self.anchors or self._anchored(hypothesis, contexts, source_ids)
        best = max(range(len(results)), key=lambda i: results[i][0])
        ok = anchored and any(supported for _, supported in results)
        status = ClaimStatus.VERIFIED if ok else ClaimStatus.UNVERIFIED
        return VerifiedClaim(
            claim=text,
            hypothesis=hypothesis,
            source_ids=source_ids,
            status=status,
            confidence=min(1.0, max(0.0, results[best][0])),
            reason=None if ok else "cited_support_not_established",
            citation="; ".join(contexts[key].citation for key in source_ids),
            evidence_chunk_id=contexts[source_ids[best]].chunk_id,
            citation_inherited=is_inherited,
            grade=_grade(status),
        )


def _nli_single(v: ClaimFilter, hypothesis: str, context: SourceContext) -> tuple[float, bool]:
    p = v.backend.predict(v._premise(context), hypothesis)
    if p is None:
        return 0.0, False
    return p.p_entailment, p.p_entailment > 0.40 and p.p_contradiction < 0.20


def _bge_zs(threshold: float) -> Decision:
    def decide(v: ClaimFilter, hypothesis: str, context: SourceContext) -> tuple[float, bool]:
        premise, variants = v._variants(context)
        texts = [premise, *(text for text, _ in variants)]
        scores = v._predict_many([(t, hypothesis) for t in texts], v.secondary)
        best = max((p.p_entailment for p in scores if p is not None), default=0.0)
        return best, best >= threshold

    return decide


_MINICHECK: MiniCheck | None = None


def minicheck() -> MiniCheck:
    global _MINICHECK
    if _MINICHECK is None:
        _MINICHECK = MiniCheck()
    return _MINICHECK


def _minicheck(threshold: float) -> Decision:
    def decide(v: ClaimFilter, hypothesis: str, context: SourceContext) -> tuple[float, bool]:
        score = minicheck().score_many([(v._premise(context), hypothesis)])[0]
        return score, score >= threshold

    return decide


def build(name: str, threshold: float | None = None) -> NLIFactVerifier:
    """A verifier by name; `threshold` overrides the default of bge_zs and minicheck."""
    if name == "judge":
        return run_verifier("judge")
    if name == "judge_free":
        return run_verifier("judge", nli_anchor_guard=False)
    if name == "nli_single":
        return ClaimFilter(name, _nli_single)
    t = DEFAULT_THRESHOLDS[name] if threshold is None else threshold
    if name == "bge_zs":
        return ClaimFilter(name, _bge_zs(t))
    if name == "minicheck":
        return ClaimFilter(name, _minicheck(t))
    raise KeyError(name)
