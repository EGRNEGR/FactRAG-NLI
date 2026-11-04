"""Bind v3 drafts to production chunks and write data/eval_dataset_v3.json.

Checks (all errors are collected and reported in one pass):
* every evidence quote and NLI premise quote occurs verbatim (whitespace/case-insensitive)
  in a chunk produced by the production parser (BGE-M3 tokenizer, 320/384/50);
* multi_hop evidence spans at least two disjoint chunk sets;
* absence_terms of refusal cases match nowhere in the indexed corpus;
* no near-duplicates within v3 or against earlier datasets (token Jaccard >= 0.8).

Parser v2 (optimization_5, --parser v2) writes its own derived HTML (derived_v2/), parse
cache and dataset (data/eval_dataset_v3_parser_v2.json) and binds quotes ignoring table
markup: the drafts quote the v1 rendering of a table ("**CA**|||Assessment ..."), v2 renders
the same cells as "| CA | Assessment ... |". Questions, answers and quotes stay as drafted.

Usage:  python scripts/build_eval_dataset_v3.py [--check] [--parser v1|v2]
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
from scripts.download_quality_corpus import MarkdownExtractor, TextExtractor  # noqa: E402
from settings import RAGSettings  # noqa: E402

CORPUS = ROOT / "data" / "test_docs"
DERIVED = CORPUS / "derived"
DERIVED_V2 = CORPUS / "derived_v2"
DRAFTS = ROOT / "data" / "eval_v3" / "drafts"
OUTPUT = ROOT / "data" / "eval_dataset_v3.json"
CACHE = ROOT / ".artifacts" / "eval_v3_parse_cache"
OUTPUT_V2 = ROOT / "data" / "eval_dataset_v3_parser_v2.json"
CACHE_V2 = ROOT / ".artifacts" / "eval_v3_parse_cache_parser_v2"
_TABLE_MARKUP = re.compile(r"[|*]+")
_CHUNK_HEADER = re.compile(r"^Документ:[^\n]*\nРаздел:[^\n]*\n")
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


def derive_html(version: str = "v1") -> dict[str, str]:
    """Deterministic HTML -> Markdown conversion (v1: the extractor of the golden corpus)."""
    folder = DERIVED_V2 if version == "v2" else DERIVED
    folder.mkdir(exist_ok=True)
    mapping: dict[str, str] = {}
    for html in sorted(CORPUS.glob("*.html")):
        extractor = MarkdownExtractor() if version == "v2" else TextExtractor()
        extractor.feed(html.read_bytes().decode("utf-8-sig"))
        target = folder / f"{html.stem}.md"
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


def parse_all(
    files: Sequence[str], corpus: Path = CORPUS, cache: Path = CACHE, parser_version: str = "v1"
) -> dict[str, dict[str, Any]]:
    settings = RAGSettings().model_copy(update={**CHUNKING, "parser_version": parser_version})
    parser: DocumentParser | None = None
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer = settings.embedding_model_path.name
    parsed: dict[str, dict[str, Any]] = {}
    for name in files:
        path = corpus / name
        digest = sha256(path)
        identity: list[Any] = [digest, CHUNKING, tokenizer]
        if parser_version != "v1":  # v1 keeps the cache keys of every earlier run
            identity.append(parser_version)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        cached = cache / f"{key}.json"
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


def v2_documents(drafts: list[DraftCase]) -> list[DraftCase]:
    """Parser v2 reads HTML through derived_v2/: the same question, the v2 rendering."""
    prefix = DERIVED.relative_to(CORPUS).as_posix() + "/"
    target = DERIVED_V2.relative_to(CORPUS).as_posix() + "/"
    return [
        d.model_copy(update={"document": target + d.document[len(prefix) :]})
        if d.document.startswith(prefix)
        else d
        for d in drafts
    ]


def without_table_markup(text: str) -> str:
    """Normalized text without cell separators and bold markers (parser v2 binding)."""
    return " ".join(_TABLE_MARKUP.sub(" ", normalize(text)).split())


def merge_overlap(left: str, right: str) -> str:
    """Two consecutive chunk bodies as one text, the repeated overlap dropped."""
    for size in range(min(len(left), len(right)), 0, -1):
        if left.endswith(right[:size]):
            return left + right[size:]
    return f"{left} {right}"


def bind(
    drafts: list[DraftCase],
    parsed: dict[str, dict[str, Any]],
    *,
    ignore_table_markup: bool = False,
    chunk_runs: bool = False,
    warnings: list[str] | None = None,
) -> tuple[list[EvalCase], list[str]]:
    """chunk_runs (parser v2): a quote across a chunk boundary binds to the run of consecutive
    chunks that contains it, and multi_hop evidence in one chunk is a warning, not an error -
    the chunking changed, the question did not."""
    errors: list[str] = []
    norm = without_table_markup if ignore_table_markup else normalize
    norm_chunks = {
        name: [(c["chunk_id"], norm(c["text"])) for c in rec["chunks"]]
        for name, rec in parsed.items()
    }
    corpus_text = "\n".join(text for chunks in norm_chunks.values() for _, text in chunks)
    bodies = {
        name: [(c["chunk_id"], norm(_CHUNK_HEADER.sub("", c["text"], count=1))) for c in rec["chunks"]]
        for name, rec in parsed.items()
    }

    def find(name: str, quote: str) -> list[str]:
        needle = norm(quote)
        ids = [cid for cid, text in norm_chunks[name] if needle in text]
        if ids or not chunk_runs:
            return ids
        chunks = bodies[name]
        for width in (2, 3):
            for start in range(len(chunks) - width + 1):
                run = chunks[start : start + width]
                text = run[0][1]
                for _, following in run[1:]:
                    text = merge_overlap(text, following)
                if needle in text:
                    return [cid for cid, _ in run]
        return []

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
                message = f"{draft.id}: multi_hop evidence lies in one chunk"
                if chunk_runs and warnings is not None:
                    warnings.append(message)
                else:
                    errors.append(message)
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


def build(
    parser_version: str, parsed: dict[str, dict[str, Any]], files: Sequence[str], mapping: dict[str, str]
) -> tuple[EvalDatasetV3 | None, list[str], list[str]]:
    """Bind the drafts to parsed chunks: (dataset or None on errors, errors, warnings)."""
    v2 = parser_version == "v2"
    drafts = load_drafts(DRAFTS)
    if v2:
        drafts = v2_documents(drafts)
    warnings: list[str] = []
    cases, errors = bind(
        drafts, parsed, ignore_table_markup=v2, chunk_runs=v2, warnings=warnings
    )
    errors += duplicates(drafts)
    if errors:
        return None, errors, warnings
    dataset = EvalDatasetV3(
        description=(
            "Stratified FactRAG-NLI evaluation set v3: single/multi-hop, tabular & code, "
            "adversarial, unanswerable and lexical-trap questions over data/test_docs. "
            "AI-authored drafts bound to production chunks; human review pending."
        ),
        corpus_root="data/test_docs",
        chunking={
            **CHUNKING,
            "tokenizer": "models/bge-m3 (BAAI/bge-m3)",
            **({"parser_version": "v2"} if v2 else {}),
        },
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
        stats=statistics(cases),
    )
    return dataset, errors, warnings


def main(argv: Sequence[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--check", action="store_true", help="Validate only, do not write output")
    cli.add_argument("--parser", choices=("v1", "v2"), default="v1", help="Document parser")
    args = cli.parse_args(argv)
    v2 = args.parser == "v2"
    mapping = derive_html(args.parser)
    files = corpus_files(mapping)
    parsed = parse_all(files, cache=CACHE_V2 if v2 else CACHE, parser_version=args.parser)
    dataset, errors, warnings = build(args.parser, parsed, files, mapping)
    for warning in warnings:
        print(f"  warning: {warning}", file=sys.stderr)
    if dataset is None:
        print(f"{len(errors)} problem(s):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(json.dumps(dataset.stats, ensure_ascii=False, indent=1))
    if args.check:
        return 0
    output = OUTPUT_V2 if v2 else OUTPUT
    output.write_text(dataset.model_dump_json(indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {output.relative_to(ROOT)} ({len(dataset.cases)} cases)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
