"""Parse the v3 corpus with parser v1 or v2 into a parse cache (the record format of the index build).

The records are the ones scripts/build_eval_dataset_v3.py caches and evaluate_suite reads
(document_id, sha256, chunks with chunk_id and text) plus section paths and kinds for the study.
The parser is built like DocumentParser.from_settings in production: the BGE-M3 tokenizer and
the v3 chunking 320/384/50, without RAGSettings (it validates every model directory).

Usage:  python -m research.optimization_5.corpus v1 <cache dir>
        python -m research.optimization_5.corpus v2 <cache dir>
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

from document_parser import ChunkingConfig, DocumentParser, LocalTokenizer, ParserVersion
from scripts import build_eval_dataset_v3 as builder

ROOT = Path(__file__).resolve().parents[2]
# Models of the checkout; a git worktree has none of its own (OPT5_MODELS=<checkout>/models).
MODELS = Path(os.environ.get("OPT5_MODELS", ROOT / "models"))
TOKENIZER = MODELS / "bge-m3"


def parser(version: str) -> DocumentParser:
    chunking = builder.CHUNKING
    return DocumentParser(
        ChunkingConfig(
            target_tokens=chunking["chunk_target_tokens"],
            max_tokens=chunking["chunk_max_tokens"],
            overlap_tokens=chunking["chunk_overlap_tokens"],
        ),
        tokenizer=LocalTokenizer(TOKENIZER),
        version=cast(ParserVersion, version),
    )


def files(version: str) -> tuple[Path, list[str]]:
    """Corpus root and document paths: v2 reads HTML through the v2 converter (derived_v2/)."""
    mapping = builder.derive_html(version)
    return builder.CORPUS, builder.corpus_files(mapping)


def parse(version: str, cache: Path) -> dict[str, dict[str, Any]]:
    cache.mkdir(parents=True, exist_ok=True)
    corpus, names = files(version)
    instance = parser(version)
    records: dict[str, dict[str, Any]] = {}
    for name in names:
        started = time.perf_counter()
        document = instance.parse(corpus / name)
        records[name] = {
            "sha256": builder.sha256(corpus / name),
            "document_id": document.document_id,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "section_path": list(c.section_path),
                    "kinds": list(c.metadata.get("kinds", ())),
                }
                for c in document.chunks
            ],
        }
        print(f"{name}: {len(document.chunks)} chunks, {time.perf_counter() - started:.0f} s",
              file=sys.stderr, flush=True)
    (cache / "corpus.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return records


def load(cache: Path) -> dict[str, dict[str, Any]]:
    corpus: dict[str, dict[str, Any]] = json.loads(
        (cache / "corpus.json").read_text(encoding="utf-8")
    )
    return corpus


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in {"v1", "v2"}:
        print(__doc__)
        raise SystemExit(2)
    parse(sys.argv[1], Path(sys.argv[2]))
