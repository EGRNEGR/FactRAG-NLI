"""Bind v3 drafts to production chunks and write data/eval_dataset_v3.json.

Checks (all errors are collected and reported in one pass):
* every evidence quote and NLI premise quote occurs verbatim (whitespace/case-insensitive)
  in a chunk produced by the production parser (BGE-M3 tokenizer, 320/384/50);
* multi_hop evidence spans at least two disjoint chunk sets;
* absence_terms of refusal cases match nowhere in the indexed corpus;
* no near-duplicates within v3 or against earlier datasets (token Jaccard >= 0.8).

Usage:  python scripts/build_eval_dataset_v3.py [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from document_parser import DocumentParser  # noqa: E402
from eval_dataset import (  # noqa: E402
    CorpusEntry,
    DraftCase,
    EvalCase,
    EvalDatasetV3,
    Evidence,
    NLIProbe,
    load_drafts,
    normalize,
)
from scripts.download_quality_corpus import TextExtractor  # noqa: E402
from settings import RAGSettings  # noqa: E402

CORPUS = ROOT / "data" / "test_docs"
DERIVED = CORPUS / "derived"
DRAFTS = ROOT / "data" / "eval_v3" / "drafts"
OUTPUT = ROOT / "data" / "eval_dataset_v3.json"
CACHE = ROOT / ".artifacts" / "eval_v3_parse_cache"
CHUNKING = {"chunk_target_tokens": 320, "chunk_max_tokens": 384, "chunk_overlap_tokens": 50}
PRIOR = (
    ("data/golden_dataset.json", "examples"),
    ("data/holdout_dataset.json", "examples"),
    ("data/extended_dataset_v2.json", "cases"),
    ("data/extended_questions.json", None),
    ("data/qa_drafts.json", None),
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def derive_html() -> dict[str, str]:
    """Deterministic HTML -> Markdown conversion (same extractor as the golden corpus)."""
    DERIVED.mkdir(exist_ok=True)
    mapping: dict[str, str] = {}
    for html in sorted(CORPUS.glob("*.html")):
        extractor = TextExtractor()
        extractor.feed(html.read_bytes().decode("utf-8-sig"))
        target = DERIVED / f"{html.stem}.md"
        text = extractor.text()
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8", newline="\n")
        mapping[target.relative_to(CORPUS).as_posix()] = html.name
    return mapping


def corpus_files(mapping: dict[str, str]) -> list[str]:
    native = [
        p.name for p in sorted(CORPUS.iterdir()) if p.suffix.lower() in {".txt", ".md", ".pdf"}
    ]
    return native + sorted(mapping)


def parse_all(files: Sequence[str]) -> dict[str, dict[str, Any]]:
    settings = RAGSettings().model_copy(update=CHUNKING)
    parser: DocumentParser | None = None
    CACHE.mkdir(parents=True, exist_ok=True)
    tokenizer = settings.embedding_model_path.name
    parsed: dict[str, dict[str, Any]] = {}
    for name in files:
        path = CORPUS / name
        digest = sha256(path)
        key = hashlib.sha256(
            json.dumps([digest, CHUNKING, tokenizer], sort_keys=True).encode()
        ).hexdigest()[:20]
        cached = CACHE / f"{key}.json"
        if cached.exists():
            parsed[name] = json.loads(cached.read_text(encoding="utf-8"))
            continue
        if parser is None:
            parser = DocumentParser.from_settings(settings)
        print(f"Parsing {name} ...", file=sys.stderr, flush=True)
        doc = parser.parse(path)
        record = {
            "sha256": digest,
            "document_id": doc.document_id,
            "chunks": [{"chunk_id": c.chunk_id, "text": c.text} for c in doc.chunks],
        }
        cached.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        parsed[name] = record
    return parsed


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", normalize(text)))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def prior_questions() -> list[str]:
    questions: list[str] = []
    for rel, key in PRIOR:
        data = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        rows = data[key] if key else data
        questions.extend(str(row["question"]) for row in rows)
    return questions


def bind(
    drafts: list[DraftCase], parsed: dict[str, dict[str, Any]]
) -> tuple[list[EvalCase], list[str]]:
    errors: list[str] = []
    norm_chunks = {
        name: [(c["chunk_id"], normalize(c["text"])) for c in rec["chunks"]]
        for name, rec in parsed.items()
    }
    corpus_text = "\n".join(text for chunks in norm_chunks.values() for _, text in chunks)

    def find(name: str, quote: str) -> list[str]:
        needle = normalize(quote)
        return [cid for cid, text in norm_chunks[name] if needle in text]

    cases: list[EvalCase] = []
    for draft in drafts:
        if draft.document not in parsed:
            errors.append(f"{draft.id}: unknown document {draft.document}")
            continue
        evidence: list[Evidence] = []
        for quote in draft.quotes:
            ids = find(draft.document, quote)
            if ids:
                evidence.append(Evidence(quote=quote, chunk_ids=ids))
            else:
                errors.append(f"{draft.id}: quote not found: {quote[:90]!r}")
        premise = find(draft.document, draft.nli_probe.premise_quote)
        if not premise:
            errors.append(f"{draft.id}: premise not found: {draft.nli_probe.premise_quote[:90]!r}")
        if draft.category == "multi_hop" and len(evidence) == len(draft.quotes):
            sets = [set(e.chunk_ids) for e in evidence]
            if all(a & b for i, a in enumerate(sets) for b in sets[i + 1 :]):
                errors.append(f"{draft.id}: multi_hop evidence lies in one chunk")
        for term in draft.absence_terms:
            if re.search(term, corpus_text, flags=re.I):
                errors.append(f"{draft.id}: absence term occurs in corpus: {term!r}")
        if errors and errors[-1].startswith(draft.id):
            continue
        cases.append(
            EvalCase(
                id=draft.id,
                category=draft.category,
                subtype=draft.subtype,
                language=draft.language,
                document=draft.document,
                document_id=parsed[draft.document]["document_id"],
                question=draft.question,
                ground_truth_answer=draft.ground_truth_answer,
                expected_behavior=draft.expected_behavior,
                expected_nli_verdict=draft.nli_probe.label,
                evidence=evidence,
                gold_chunk_ids=sorted({cid for e in evidence for cid in e.chunk_ids}),
                nli_probe=NLIProbe(
                    premise_chunk_id=premise[0],
                    premise_quote=draft.nli_probe.premise_quote,
                    hypothesis_en=draft.nli_probe.hypothesis_en,
                    hypothesis_ru=draft.nli_probe.hypothesis_ru,
                    label=draft.nli_probe.label,
                ),
                answer_must_contain=draft.answer_must_contain,
                absence_terms=draft.absence_terms,
                distortion=draft.distortion,
            )
        )
    return cases, errors


def duplicates(drafts: list[DraftCase]) -> list[str]:
    errors: list[str] = []
    prior = [tokens(q) for q in prior_questions()]
    seen: list[tuple[str, set[str]]] = []
    for draft in drafts:
        own = tokens(draft.question)
        if any(jaccard(own, other) >= 0.8 for other in prior):
            errors.append(f"{draft.id}: near-duplicate of an earlier dataset question")
        for other_id, other in seen:
            if jaccard(own, other) >= 0.8:
                errors.append(f"{draft.id}: near-duplicate of {other_id}")
        seen.append((draft.id, own))
    return errors


def statistics(cases: list[EvalCase]) -> dict[str, Any]:
    def count(key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(getattr(c, key)) for c in cases).items()))

    return {
        "cases": len(cases),
        "by_category": count("category"),
        "by_language": count("language"),
        "by_document": count("document"),
        "by_expected_behavior": count("expected_behavior"),
        "by_expected_nli_verdict": count("expected_nli_verdict"),
        "by_subtype": count("subtype"),
        "evidence_units": sum(len(c.evidence) for c in cases),
        "multi_chunk_cases": sum(len(c.evidence) >= 2 for c in cases),
    }


def main(argv: Sequence[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--check", action="store_true", help="Validate only, do not write output")
    args = cli.parse_args(argv)
    mapping = derive_html()
    files = corpus_files(mapping)
    drafts = load_drafts(DRAFTS)
    parsed = parse_all(files)
    cases, errors = bind(drafts, parsed)
    errors += duplicates(drafts)
    if errors:
        print(f"{len(errors)} problem(s):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    stats = statistics(cases)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    if args.check:
        return 0
    dataset = EvalDatasetV3(
        description=(
            "Stratified FactRAG-NLI evaluation set v3: single/multi-hop, tabular & code, "
            "adversarial, unanswerable and lexical-trap questions over data/test_docs. "
            "AI-authored drafts bound to production chunks; human review pending."
        ),
        corpus_root="data/test_docs",
        chunking={**CHUNKING, "tokenizer": "models/bge-m3 (BAAI/bge-m3)"},
        documents=[
            CorpusEntry(
                path=name,
                sha256=parsed[name]["sha256"],
                document_id=parsed[name]["document_id"],
                chunks=len(parsed[name]["chunks"]),
                derived_from=mapping.get(name),
            )
            for name in files
        ],
        cases=cases,
        stats=stats,
    )
    OUTPUT.write_text(dataset.model_dump_json(indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({len(cases)} cases)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
