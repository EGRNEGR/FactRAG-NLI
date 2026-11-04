"""The v3 dataset for parser v2 from the research parse caches (scripts/build_eval_dataset_v3.build).

v1 must give data/eval_dataset_v3.json byte for byte (the refactored builder and the research
parse are the production ones). v2 writes data/eval_dataset_v3_parser_v2.json and checks
that only the binding moved: per case the question, answers, expected behaviour, quotes and
probe hypotheses equal the v1 dataset; the document path changes only for derived HTML.

Usage:  python -m research.optimization_5.dataset <v1 cache> <v2 cache> <out json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from eval_dataset import load_dataset
from scripts import build_eval_dataset_v3 as builder

from research.optimization_5 import corpus

LABEL_FIELDS = (
    "category",
    "subtype",
    "language",
    "question",
    "ground_truth_answer",
    "expected_behavior",
    "expected_nli_verdict",
    "answer_must_contain",
    "absence_terms",
    "distortion",
)


def render(version: str, cache: Path) -> tuple[str | None, list[str], list[str]]:
    _, files = corpus.files(version)
    mapping = builder.derive_html(version)
    dataset, errors, warnings = builder.build(version, corpus.load(cache), files, mapping)
    text = dataset.model_dump_json(indent=1) + "\n" if dataset else None
    return text, errors, warnings


def main(argv: list[str]) -> int:
    v1_cache, v2_cache, out = (Path(a) for a in argv)
    v1_text, v1_errors, _ = render("v1", v1_cache)
    reference = builder.OUTPUT.read_text(encoding="utf-8")
    v2_text, v2_errors, v2_warnings = render("v2", v2_cache)
    result: dict[str, Any] = {
        "v1_identical_to_dataset": v1_text == reference,
        "v1_errors": v1_errors,
        "v2_errors": v2_errors,
        "v2_warnings": v2_warnings,
    }
    if v2_text is not None:
        builder.OUTPUT_V2.write_text(v2_text, encoding="utf-8", newline="\n")
        one = {c.id: c for c in load_dataset(builder.OUTPUT).cases}
        two = {c.id: c for c in load_dataset(builder.OUTPUT_V2).cases}
        label_diffs = [
            cid
            for cid in one
            if cid not in two
            or any(getattr(one[cid], f) != getattr(two[cid], f) for f in LABEL_FIELDS)
            or [e.quote for e in one[cid].evidence] != [e.quote for e in two[cid].evidence]
            or one[cid].nli_probe.model_dump(exclude={"premise_chunk_id"})
            != two[cid].nli_probe.model_dump(exclude={"premise_chunk_id"})
        ]
        result.update(
            v2_cases=len(two),
            v2_label_differences=label_diffs,
            v2_quotes_in_several_chunks=sorted(
                cid for cid, c in two.items() if any(len(e.chunk_ids) > 1 for e in c.evidence)
            ),
            v1_quotes_in_several_chunks=sorted(
                cid for cid, c in one.items() if any(len(e.chunk_ids) > 1 for e in c.evidence)
            ),
            v2_document_paths_changed=sorted(cid for cid in one if one[cid].document != two[cid].document),
        )
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v if not isinstance(v, list) or len(v) < 12 else f"{len(v)} items"
                      for k, v in result.items()}, ensure_ascii=False, indent=1))
    ok = result["v1_identical_to_dataset"] and not v2_errors and not result.get("v2_label_differences")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
