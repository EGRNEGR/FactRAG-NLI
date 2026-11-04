"""Table stress set: does a variant accept false claims about table rows?

Built from the Markdown tables of every context chunk the run 1222Z gave its v3 drafts
(parser v2), deterministically. A table qualifies with a full header of two to eight
columns, at least three body rows and a key column (the first one whose values are filled
and distinct across rows). For each of the first MAX_ROWS rows and each other column with
at least two distinct values (MAX_COLUMNS of them), four claims cite the chunk as [S1]:

* positive_en / positive_ru: the row's own value of the column for its key;
* swap_en / swap_ru: the value of the same column from the nearest following row whose value
  neither contains nor is contained in the true one (a fluent false claim whose value occurs
  in the table: the typical error of reading the neighbouring row).

Claims are templated, not LLM text: the set measures the verifier's acceptance of a row fact
against a wrong-row fact under the same wording, not the drafting quality.

Usage:  python -m research.optimization_6.table_stress [B00 T1 T2 T12]
Output: reports/optimization_6/offline/table_stress.json (acceptance per kind and variant,
        every accepted false claim) and .artifacts/opt6/table_stress.jsonl (the set).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Any

import research.optimization_6  # noqa: F401 - selects the baseline run and the output folder
from generator_verifier import RawGenerationResult
from grounding import markdown_tables
from research.opt3 import data
from research.opt3.replay import verifier
from research.opt4.offline import CONFIG, RESULTS
from research.optimization_6 import ROOT
from research.optimization_6.offline import REGISTRY

PATH = ROOT / ".artifacts" / "opt6" / "table_stress.jsonl"
MAX_ROWS = 4
MAX_COLUMNS = 2


def _key_column(header: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> int | None:
    for c in range(len(header)):
        values = [r[c] for r in rows]
        if all(values) and len(set(values)) == len(values):
            return c
    return None


def _distinct(a: str, b: str) -> bool:
    """A wrong-row value that is not part of the true one (nor the reverse): "RFC 9562" for
    "[RFC4122], RFC 9562" would make a true claim."""
    x, y = a.casefold(), b.casefold()
    return x not in y and y not in x


def _claims(caption: str | None, header: tuple[str, ...], key: int, row: tuple[str, ...],
            column: int, value: str) -> dict[str, str]:
    table = f"«{caption}»" if caption else "таблице"
    table_en = caption or "the table"
    return {
        "en": f"In {table_en}, the {header[column]} for {header[key]} {row[key]} is {value} [S1].",
        "ru": f"Согласно {table if caption else 'таблице'}, для {header[key]} {row[key]} "
              f"значение столбца «{header[column]}» равно {value} [S1].",
    }


def build() -> list[dict[str, Any]]:
    chunks: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {}
    for record in sorted(data.records().values(), key=lambda r: r.id):
        if data.dataset_of(record.id) != "v3":
            continue
        for attempt in record.attempts[CONFIG]:
            for context in attempt.contexts.values():
                chunks.setdefault(context["chunk_id"], (record.id, context, attempt.metadata))
    items: list[dict[str, Any]] = []
    for chunk_id in sorted(chunks):
        record_id, context, metadata = chunks[chunk_id]
        for t, table in enumerate(markdown_tables(context["text"])):
            header, rows = table.header, table.rows
            if not (2 <= len(header) <= 8 and all(header) and len(rows) >= 3):
                continue
            key = _key_column(header, rows)
            if key is None:
                continue
            columns = [
                c for c in range(len(header))
                if c != key and all(r[c] for r in rows) and len({r[c] for r in rows}) >= 2
            ][:MAX_COLUMNS]
            for column in columns:
                for i, row in enumerate(rows[:MAX_ROWS]):
                    other = next(
                        (rows[(i + d) % len(rows)][column] for d in range(1, len(rows))
                         if _distinct(rows[(i + d) % len(rows)][column], row[column])),
                        None,
                    )
                    if other is None:
                        continue
                    for kind, value in (("positive", row[column]), ("swap", other)):
                        for lang, text in _claims(
                            table.caption, header, key, row, column, value
                        ).items():
                            items.append({
                                "chunk_id": chunk_id, "record": record_id, "table": t,
                                "kind": f"{kind}_{lang}", "text": text,
                                "context": context, "metadata": metadata,
                            })
    return items


def load() -> list[dict[str, Any]]:
    if not PATH.exists():
        PATH.parent.mkdir(parents=True, exist_ok=True)
        with PATH.open("w", encoding="utf-8") as handle:
            for item in build():
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines()]


def evaluate(name: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    v = verifier(CONFIG, **REGISTRY[name].overrides)
    counts: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    accepted_false: list[dict[str, str]] = []
    for item in items:
        context = {**item["context"], "citation": item["context"].get("citation") or "S1"}
        raw = RawGenerationResult.model_validate(
            {"text": item["text"], "contexts": {"S1": context}, "metadata": item["metadata"]}
        )
        ok = not v.verify(raw).refused
        counts[item["kind"]] += 1
        hits[item["kind"]] += ok
        if ok and item["kind"].startswith("swap"):
            accepted_false.append({"chunk_id": item["chunk_id"], "text": item["text"]})
    return {
        "acceptance": {k: {"k": hits[k], "n": n, "rate": hits[k] / n} for k, n in sorted(counts.items())},
        "accepted_false": accepted_false,
    }


def main(names: list[str]) -> None:
    items = load()
    tables = {(i["chunk_id"], i["table"]) for i in items}
    print(len(items), "claims from", len(tables), "tables,", Counter(i["kind"] for i in items))
    target = RESULTS / "table_stress.json"
    result = json.loads(target.read_text("utf-8")) if target.exists() else {}
    for name in names or list(REGISTRY):
        result[name] = evaluate(name, items)
        line = "; ".join(
            f"{k} {100 * r['rate']:.1f}% ({r['k']}/{r['n']})"
            for k, r in result[name]["acceptance"].items()
        )
        print(f"{name}: {line}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
