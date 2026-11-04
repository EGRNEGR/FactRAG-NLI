"""Regression of parser v2 outside the English tables: corpora whose chunks must not change.

Parses every document of a corpus with v1 and v2 (the production tokenizer and the corpus's
chunking) and compares chunk ids and texts. XQuAD (40 Russian Wikipedia articles, the 1087
questions of the benchmark) must be byte-identical: then its index, answers and scores cannot
change with the parser switch. The golden corpus (data/documents) is reported as is.

Usage:  python -m research.optimization_5.regression <out json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from eval_dataset import load_dataset

from research.optimization_5 import corpus

ROOT = Path(__file__).resolve().parents[2]
CORPORA = {
    "xquad-full": ROOT / "data" / "xquad_dataset_full.json",
}
GOLDEN = ROOT / "data" / "documents"


def chunks(version: str, path: Path) -> list[tuple[str, str, tuple[str, ...]]]:
    return [(c.chunk_id, c.text, c.section_path) for c in corpus.parser(version).parse(path).chunks]


def compare_files(paths: list[Path]) -> dict[str, Any]:
    rows = {}
    for path in paths:
        one, two = chunks("v1", path), chunks("v2", path)
        rows[path.name] = {
            "v1_chunks": len(one),
            "v2_chunks": len(two),
            "identical": one == two,
            "same_text_different_sections": [a[1] for a in one] == [b[1] for b in two] and one != two,
        }
    return {
        "documents": len(rows),
        "identical": sum(r["identical"] for r in rows.values()),
        "chunks_v1": sum(r["v1_chunks"] for r in rows.values()),
        "chunks_v2": sum(r["v2_chunks"] for r in rows.values()),
        "files": rows,
    }


def main(argv: list[str]) -> int:
    result: dict[str, Any] = {}
    for name, path in CORPORA.items():
        dataset = load_dataset(path)
        root = ROOT / dataset.corpus_root
        result[name] = compare_files([root / d.path for d in dataset.documents])
    result["golden"] = compare_files(sorted(p for p in GOLDEN.iterdir() if p.suffix in {".md", ".txt"}))
    Path(argv[0]).parent.mkdir(parents=True, exist_ok=True)
    Path(argv[0]).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, block in result.items():
        print(name, {k: v for k, v in block.items() if k != "files"})
        for file, row in block["files"].items():
            if not row["identical"]:
                print("   differs:", file, row)
    return 0 if result["xquad-full"]["identical"] == result["xquad-full"]["documents"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
