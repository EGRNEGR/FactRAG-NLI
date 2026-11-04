"""Structure of the v3 corpus under parser v1 and v2, per document (parse caches of corpus.py).

Counted per document: chunks; table chunks; table chunks that name their table (a caption
line "Table N" / "TABLE N" / "Таблица N" in the chunk); table chunks without a data row
(header only); distinct section paths; chunks whose section is a numbered line that
continues a sentence (lowercase after the number, a sentence end, or a semicolon); the median
chunk length in characters.

Usage:  python -m research.optimization_5.structure <v1 cache> <v2 cache> <out json>
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from research.optimization_5 import corpus

_CAPTION_LINE = re.compile(r"^(?:Table|TABLE|Таблица)\s+[A-Z]?\d", re.M)
_ROW = re.compile(r"^\|.*\|$")
_RULE = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|)+$")
_JUNK_SECTION = re.compile(r"^\d+\.?\s+(?:[a-zа-я(\[]|.*[.,;:]$|.*;)")


def document_stats(record: dict[str, Any]) -> dict[str, Any]:
    chunks = record["chunks"]
    tables = [c for c in chunks if "table" in c["kinds"]]

    def data_rows(text: str) -> int:
        rows = [ln for ln in text.splitlines() if _ROW.match(ln) and not _RULE.match(ln)]
        return max(0, len(rows) - 1)

    sections = {" > ".join(c["section_path"]) for c in chunks}
    return {
        "chunks": len(chunks),
        "table_chunks": len(tables),
        "table_chunks_captioned": sum(bool(_CAPTION_LINE.search(c["text"])) for c in tables),
        "table_chunks_header_only": sum(data_rows(c["text"]) == 0 for c in tables),
        "sections": len(sections),
        "chunks_in_junk_section": sum(
            any(_JUNK_SECTION.match(s) for s in c["section_path"]) for c in chunks
        ),
        "median_chars": int(statistics.median(len(c["text"]) for c in chunks)),
    }


def compare(v1: Path, v2: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    one, two = corpus.load(v1), corpus.load(v2)
    for name, record in one.items():
        other = name.replace("derived/", "derived_v2/", 1)
        result[name] = {"v1": document_stats(record), "v2": document_stats(two[other])}
    totals: dict[str, Any] = {}
    for version in ("v1", "v2"):
        keys = result[next(iter(result))][version].keys()
        totals[version] = {
            k: sum(doc[version][k] for doc in result.values()) for k in keys if k != "median_chars"
        }
    result["total"] = totals
    return result


if __name__ == "__main__":
    out = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[3]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, row in out.items():
        print(name, json.dumps(row, ensure_ascii=False))
