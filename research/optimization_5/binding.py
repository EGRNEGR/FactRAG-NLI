"""Bind the v3 drafts to parser v1 and v2 chunks with the dataset builder's own checks.

Checks of scripts/build_eval_dataset_v3.py (bind): every evidence and premise quote is found
in a chunk of its document, multi_hop evidence spans two chunk sets, absence terms of
refusal cases occur nowhere in the corpus. v1 must reproduce data/eval_dataset_v3.json
exactly (same gold chunk ids); the table-insensitive matching of v2 is also run on v1 to
show that the matching rule alone changes no binding.

Usage:  python -m research.optimization_5.binding <v1 cache dir> <v2 cache dir> <out json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from eval_dataset import load_dataset, load_drafts
from scripts import build_eval_dataset_v3 as builder

from research.optimization_5 import corpus


def bind(cache: Path, *, v2: bool, ignore_table_markup: bool) -> tuple[dict[str, Any], list[str]]:
    drafts = load_drafts(builder.DRAFTS)
    if v2:
        drafts = builder.v2_documents(drafts)
    cases, errors = builder.bind(drafts, corpus.load(cache), ignore_table_markup=ignore_table_markup)
    return {c.id: c for c in cases}, errors


def main(argv: list[str]) -> int:
    v1_cache, v2_cache, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    reference = {c.id: c for c in load_dataset(builder.OUTPUT).cases}
    v1, v1_errors = bind(v1_cache, v2=False, ignore_table_markup=False)
    v1_loose, v1_loose_errors = bind(v1_cache, v2=False, ignore_table_markup=True)
    v2, v2_errors = bind(v2_cache, v2=True, ignore_table_markup=True)
    v1_same = [cid for cid, c in reference.items() if cid in v1 and v1[cid].gold_chunk_ids == c.gold_chunk_ids
               and v1[cid].nli_probe.premise_chunk_id == c.nli_probe.premise_chunk_id]
    loose_same = [cid for cid, c in v1.items() if cid in v1_loose
                  and v1_loose[cid].gold_chunk_ids == c.gold_chunk_ids]
    result = {
        "reference_cases": len(reference),
        "v1": {"cases": len(v1), "errors": v1_errors, "identical_to_dataset": len(v1_same)},
        "v1_table_insensitive": {"cases": len(v1_loose), "errors": v1_loose_errors,
                                 "same_gold_as_v1": len(loose_same),
                                 "changed": sorted(set(v1) - set(loose_same))},
        "v2": {"cases": len(v2), "errors": v2_errors,
               "missing": sorted(set(reference) - set(v2))},
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: (vv if not isinstance(vv, list) or len(vv) < 15 else f"{len(vv)} items")
                          for kk, vv in v.items()} if isinstance(v, dict) else v
                      for k, v in result.items()}, ensure_ascii=False, indent=1))
    return 0 if not v1_errors and not v2_errors and len(v1_same) == len(reference) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
