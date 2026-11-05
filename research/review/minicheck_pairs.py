"""MiniCheck scores computed on another machine (the RTX 3090) and imported into the cache.

  export   (no GPU) every (premise, claim) pair research.review.compare will ask MiniCheck
           about: the first drafts of 1222Z and the stress set -> reports/review/
           minicheck_pairs.jsonl.gz
  score    (CUDA) MiniCheck-Flan-T5-Large on those pairs (resumable: the cache is committed in
           parts) -> reports/review/minicheck_scores.jsonl.gz, one {"k", "s"} per chunk pair
  import   (no GPU) the scores file into .artifacts/review/minicheck.sqlite; afterwards
           compare runs from the cache without loading the model

Usage:  python -m research.review.minicheck_pairs export|score|import [file]
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

from generator_verifier import SourceContext

from research.review import OUT  # noqa: I001 - selects the run first
from research.review import verifiers as V

PAIRS = OUT / "minicheck_pairs.jsonl.gz"
SCORES = OUT / "minicheck_scores.jsonl.gz"


def export() -> Path:
    from research.opt3 import data
    from research.review.compare import first_draft, stress_items, verify

    records = data.records()
    texts = [(cid, d.raw_text) for cid, r in records.items() if (d := first_draft(r))]
    texts += [(i["id"], i["text"]) for i in stress_items(records)]
    pairs: list[tuple[str, str]] = []

    def collect(v: V.ClaimFilter, hypothesis: str, context: SourceContext) -> tuple[float, bool]:
        pairs.append((v._premise(context), hypothesis))
        return 0.0, False

    collector = V.ClaimFilter("collect", collect)
    for case_id, text in texts:
        draft = first_draft(records[case_id])
        if draft is not None:
            verify(collector, draft, text)
    unique = list(dict.fromkeys(pairs))
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(PAIRS, "wt", encoding="utf-8") as handle:
        for doc, claim in unique:
            handle.write(json.dumps({"doc": doc, "claim": claim}, ensure_ascii=False) + "\n")
    print(f"{len(unique)} pairs -> {PAIRS}")
    return PAIRS


def score(path: Path = PAIRS) -> Path:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        pairs = [(row["doc"], row["claim"]) for row in map(json.loads, handle)]
    mc = V.minicheck()
    started = time.perf_counter()
    for start in range(0, len(pairs), 512):
        mc.score_many(pairs[start : start + 512])
        print(f"{min(start + 512, len(pairs))}/{len(pairs)} pairs, "
              f"{time.perf_counter() - started:.0f} s", flush=True)
    rows = mc._db.execute("SELECT k, s FROM p").fetchall()
    with gzip.open(SCORES, "wt", encoding="utf-8") as handle:
        for k, s in rows:
            handle.write(json.dumps({"k": k, "s": s}) + "\n")
    print(f"{len(rows)} chunk scores -> {SCORES}")
    return SCORES


def load(path: Path = SCORES) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [(row["k"], row["s"]) for row in map(json.loads, handle)]
    mc = V.minicheck()
    mc._db.executemany("INSERT OR REPLACE INTO p VALUES (?, ?)", rows)
    mc._db.commit()
    print(f"{len(rows)} chunk scores imported into the MiniCheck cache")
    return len(rows)


def main(args: list[str]) -> None:
    command = args[0] if args else ""
    path = Path(args[1]) if len(args) > 1 else None
    if command == "export":
        export()
    elif command == "score":
        score(path or PAIRS)
    elif command == "import":
        load(path or SCORES)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
