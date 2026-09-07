"""Bind imported AI/human QA drafts to actual parsed documents and immutable hashes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_parser import DocumentParser
from evaluate_quality import CorpusDocument, GoldenDataset, Judgment, QAExample, digest
from settings import RAGSettings


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def prepare(root: Path, rows: list[dict[str, Any]], settings: RAGSettings) -> GoldenDataset:
    parser = DocumentParser.from_settings(settings)
    documents: dict[str, CorpusDocument] = {}
    examples = []
    for row in rows:
        source = str(row["source"])
        path = settings.resolve_document(source)
        parsed = parser.parse(path)
        document = CorpusDocument(
            path=source, sha256=digest(path), document_id=parsed.document_id, url=row["source_url"]
        )
        documents[source] = document
        quotes = row.get("quotes", [])
        if not isinstance(quotes, list) or any(
            not isinstance(q, str) or not q.strip() for q in quotes
        ):
            raise ValueError("quotes must be a list of nonempty verbatim fragments")
        negative = bool(row.get("is_negative", False))
        relevant: dict[str, Judgment] = {}
        for quote in quotes:
            matches = [
                c
                for c in parsed.chunks
                if normalize(quote) in normalize(c.text)
                and str(row.get("section_contains", "")) in " ".join(c.section_path)
            ]
            if not matches:
                raise ValueError(f"Quote not found in parsed chunks: {row['id']}: {quote[:80]}")
            for chunk in matches:
                relevant[chunk.chunk_id] = Judgment(
                    document_id=parsed.document_id, chunk_id=chunk.chunk_id
                )
        if negative and quotes:
            raise ValueError("Negative examples must have no evidence quote")
        examples.append(
            QAExample(
                id=row["id"],
                question=row["question"],
                source_url=row["source_url"],
                document_sha256=document.sha256,
                ground_truth_answer=row["ground_truth_answer"],
                relevant_fragment_quote="\n\n".join(quotes),
                relevant=list(relevant.values()),
                is_negative=negative,
                kind=row.get("kind", "negative" if negative else "fact"),
                created_by=row.get("created_by", "ai_pregenerated"),
                review_status=row.get("review_status", "ai_draft"),
            )
        )
    return GoldenDataset(
        description="Source-bound QA drafts; human review required unless marked reviewed",
        corpus_root=str(root),
        documents=list(documents.values()),
        examples=examples,
        parser_settings={
            "chunk_target_tokens": settings.chunk_target_tokens,
            "chunk_max_tokens": settings.chunk_max_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--corpus", type=Path, required=True)
    cli.add_argument(
        "--qa-file",
        type=Path,
        required=True,
        help="JSON draft rows with source, quotes and QA fields",
    )
    cli.add_argument("--output", type=Path, default=Path("data/golden_dataset.json"))
    args = cli.parse_args(argv)
    try:
        root = args.corpus.resolve(strict=True)
        rows = json.loads(args.qa_file.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError("Expected nonempty QA array")
        settings = RAGSettings.model_validate({**RAGSettings().model_dump(), "document_root": root})
        dataset = prepare(root, rows, settings)
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        dataset.corpus_root = os.path.relpath(root, output.parent)
        output.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
        print(
            f"Saved {len(dataset.examples)} examples and {len(dataset.documents)} documents to {output}"
        )
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
