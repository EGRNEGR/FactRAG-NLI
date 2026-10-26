"""Schema of the stratified evaluation dataset v3 (drafts and source-bound cases).

Drafts are authored as JSON with verbatim evidence quotes; ``scripts/build_eval_dataset_v3.py``
binds every quote to production chunk IDs and writes ``data/eval_dataset_v3.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal[
    "single_hop", "multi_hop", "tabular_code", "adversarial", "unanswerable", "lexical_trap"
]
Behavior = Literal["answer", "refusal", "refusal_or_correction"]
NLILabel = Literal["entailment", "neutral", "contradiction"]
CATEGORIES: tuple[str, ...] = (
    "single_hop", "multi_hop", "tabular_code", "adversarial", "unanswerable", "lexical_trap"
)
EXPECTED_LABEL: dict[str, str] = {
    "answer": "entailment",
    "refusal_or_correction": "contradiction",
    "refusal": "neutral",
}


def normalize(text: str) -> str:
    """Whitespace- and case-insensitive form used for quote binding and answer checks."""
    return " ".join(text.replace("ё", "е").replace("Ё", "Е").split()).casefold()


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class NLIProbeDraft(_Strict):
    premise_quote: str = Field(min_length=12)
    hypothesis_en: str = Field(min_length=8)
    hypothesis_ru: str = Field(min_length=8)
    label: NLILabel


class DraftCase(_Strict):
    id: str = Field(pattern=r"^V3-[A-Z0-9]+-\d{3}$")
    category: Category
    subtype: str = Field(min_length=2)
    language: Literal["ru", "en"]
    document: str = Field(min_length=3)
    question: str = Field(min_length=10)
    ground_truth_answer: str = Field(min_length=2)
    expected_behavior: Behavior
    quotes: list[str] = Field(default_factory=list)
    nli_probe: NLIProbeDraft
    answer_must_contain: list[str] = Field(default_factory=list)
    absence_terms: list[str] = Field(default_factory=list)
    distortion: str | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        answerable = self.expected_behavior != "refusal"
        if answerable and not self.quotes:
            raise ValueError(f"{self.id}: answerable or adversarial case needs evidence quotes")
        if not answerable and (self.quotes or not self.absence_terms):
            raise ValueError(f"{self.id}: refusal case needs absence_terms and no quotes")
        if answerable and not self.answer_must_contain:
            raise ValueError(f"{self.id}: answer_must_contain is required for scoring")
        if self.nli_probe.label != EXPECTED_LABEL[self.expected_behavior]:
            raise ValueError(f"{self.id}: probe label does not match expected behavior")
        rules = {
            "unanswerable": {"refusal"},
            "adversarial": {"refusal_or_correction"},
            "lexical_trap": {"answer", "refusal"},
        }
        if self.expected_behavior not in rules.get(self.category, {"answer"}):
            raise ValueError(f"{self.id}: behavior {self.expected_behavior} invalid for category")
        if self.category == "adversarial" and not self.distortion:
            raise ValueError(f"{self.id}: adversarial case must describe the distortion")
        if self.category == "multi_hop" and len(self.quotes) < 2:
            raise ValueError(f"{self.id}: multi_hop needs at least two evidence quotes")
        for term in self.absence_terms:
            re.compile(term)
        for fact in self.answer_must_contain:
            if fact.startswith("re:"):
                re.compile(fact[3:])
        return self


class Evidence(_Strict):
    quote: str
    chunk_ids: list[str] = Field(min_length=1)


class NLIProbe(_Strict):
    premise_chunk_id: str
    premise_quote: str
    hypothesis_en: str
    hypothesis_ru: str
    label: NLILabel


class EvalCase(_Strict):
    id: str
    category: Category
    subtype: str
    language: Literal["ru", "en"]
    document: str
    document_id: str
    question: str
    ground_truth_answer: str
    expected_behavior: Behavior
    expected_nli_verdict: NLILabel
    evidence: list[Evidence]
    gold_chunk_ids: list[str]
    nli_probe: NLIProbe
    answer_must_contain: list[str]
    absence_terms: list[str]
    distortion: str | None = None
    created_by: Literal["ai_pregenerated", "human"] = "ai_pregenerated"
    review_status: Literal["ai_draft", "human_reviewed"] = "ai_draft"

    @property
    def answerable(self) -> bool:
        return self.expected_behavior != "refusal"


class CorpusEntry(_Strict):
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    document_id: str
    chunks: int = Field(ge=1)
    derived_from: str | None = None


class EvalDatasetV3(_Strict):
    version: Literal[3] = 3
    description: str
    corpus_root: str
    chunking: dict[str, int | str]
    documents: list[CorpusEntry] = Field(min_length=1)
    cases: list[EvalCase] = Field(min_length=1)
    stats: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique(self) -> Self:
        ids = [c.id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate case ID")
        known = {d.path for d in self.documents}
        missing = {c.document for c in self.cases} - known
        if missing:
            raise ValueError(f"Cases reference documents outside the manifest: {sorted(missing)}")
        return self


def load_drafts(directory: Path) -> list[DraftCase]:
    cases: list[DraftCase] = []
    for path in sorted(directory.glob("*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{path}: draft file must contain a JSON list")
        cases.extend(DraftCase.model_validate(row) for row in rows)
    return cases


def load_dataset(path: Path) -> EvalDatasetV3:
    return EvalDatasetV3.model_validate_json(path.read_text(encoding="utf-8"))


def contains_fact(answer: str, alternatives: str) -> bool:
    """``alternatives`` is ``"a|b|c"``; any normalized alternative must occur in the answer.

    A single ``re:<pattern>`` entry is matched as a regular expression on the normalized
    answer (needed for Russian negation traps such as "четн" inside "нечетн").
    """
    text = normalize(answer)
    if alternatives.startswith("re:"):
        return re.search(alternatives[3:], text) is not None
    return any(normalize(option) in text for option in alternatives.split("|") if option.strip())
