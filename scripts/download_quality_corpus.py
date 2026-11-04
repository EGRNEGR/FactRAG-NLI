"""Online acquisition only; evaluation itself uses the resulting offline corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Sequence

import httpx

SOURCES = {
    "rfc9110": "https://www.rfc-editor.org/rfc/rfc9110.txt",
    "postgresql16-ha": "https://www.postgresql.org/docs/16/high-availability.html",
    "nist-800-63b": "https://pages.nist.gov/800-63-3/sp800-63b.html",
}


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav", "header", "footer"}:
            self.hidden += 1
        if not self.hidden:
            if tag in {"p", "div", "li", "tr", "br", "section"}:
                self.parts.append("\n\n")
            if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
            if tag in {"td", "th"}:
                self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "header", "footer"}:
            self.hidden = max(0, self.hidden - 1)
        if not self.hidden and tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)

    def text(self) -> str:
        value = re.sub(r"[ \t]+", " ", "".join(self.parts))
        return re.sub(r"\n\s*\n", "\n\n", value).strip() + "\n"


_HIDDEN = {"script", "style", "nav", "header", "footer"}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def _flat(parts: list[str]) -> str:
    return " ".join("".join(parts).split())


class MarkdownExtractor(HTMLParser):
    """HTML -> Markdown for parser v2 (optimization_5): one-line headings and pipe tables.

    TextExtractor (v1) writes the heading text on the line after "#" and every table cell on
    a line of its own, so the parser sees neither the heading nor a table row. Here a heading
    is "## Title" and a table is a Markdown table, one line per row: colspan repeats a cell
    to the right, rowspan down; leading header rows (thead or th only) merge into one
    ("Group / Column"); a nested table becomes text of its outer cell.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0
        self.heading: tuple[int, list[str]] | None = None
        self.depth = 0
        self.rows: list[dict[str, object]] = []
        self.row: dict[str, object] | None = None
        self.cell: dict[str, object] | None = None
        self.caption: list[str] | None = None
        self.in_head = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _HIDDEN:
            self.hidden += 1
        if self.hidden:
            return
        if tag == "table":
            self.depth += 1
            if self.depth == 1:
                self.rows, self.row, self.cell, self.caption = [], None, None, []
                self.in_head = False
                return
        if self.depth > 1 or (self.depth == 1 and self.cell is not None):
            if tag in {"p", "div", "li", "br", "tr", "td", "th"} and self.cell is not None:
                self._cell_text().append(" ")
            if self.depth > 1 or tag not in {"td", "th", "tr"}:
                return
        if self.depth == 1:
            if tag == "thead":
                self.in_head = True
            elif tag in {"tbody", "tfoot"}:
                self.in_head = False
            elif tag == "tr":
                self._end_row()
                self.row = {"cells": [], "head": self.in_head}
            elif tag in {"td", "th"}:
                if self.row is None:
                    self.row = {"cells": [], "head": self.in_head}
                values = dict(attrs)
                self.cell = {
                    "text": [],
                    "th": tag == "th",
                    "colspan": _span(values.get("colspan")),
                    "rowspan": _span(values.get("rowspan")),
                }
                self._row_cells().append(self.cell)
            return
        if tag in _HEADINGS:
            self.heading = (int(tag[1]), [])
        elif tag in {"p", "div", "li", "tr", "br", "section"}:
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _HIDDEN:
            self.hidden = max(0, self.hidden - 1)
            return
        if self.hidden:
            return
        if tag == "table" and self.depth:
            self.depth -= 1
            if self.depth == 0:
                self._end_row()
                caption = _flat(self.caption or [])
                table = _render_table(self.rows)
                if table:
                    self.parts.append("\n\n" + (caption + "\n\n" if caption else "") + table + "\n\n")
                self.rows, self.row, self.cell, self.caption = [], None, None, None
            return
        if self.depth == 1:
            if tag in {"td", "th"}:
                self.cell = None
            elif tag == "tr":
                self._end_row()
            return
        if self.depth:
            return
        if tag in _HEADINGS and self.heading is not None:
            level, buffer = self.heading
            self.heading = None
            text = _flat(buffer)
            if text:
                self.parts.append("\n\n" + "#" * level + " " + text + "\n\n")
        elif tag in {"p", "div", "li", "tr"}:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self.hidden:
            return
        if self.depth:
            if self.cell is not None:
                self._cell_text().append(data)
            elif self.caption is not None and self.depth == 1:
                self.caption.append(data)
        elif self.heading is not None:
            self.heading[1].append(data)
        else:
            self.parts.append(data)

    def _cell_text(self) -> list[str]:
        assert self.cell is not None
        text = self.cell["text"]
        assert isinstance(text, list)
        return text

    def _row_cells(self) -> list[dict[str, object]]:
        assert self.row is not None
        cells = self.row["cells"]
        assert isinstance(cells, list)
        return cells

    def _end_row(self) -> None:
        if self.row is not None and self._row_cells():
            self.rows.append(self.row)
        self.row, self.cell = None, None

    def text(self) -> str:
        value = re.sub(r"[ \t]+", " ", "".join(self.parts))
        return re.sub(r"\n\s*\n", "\n\n", value).strip() + "\n"


def _span(value: str | None) -> int:
    return max(1, min(int(value), 50)) if value and value.strip().isdigit() else 1


def _render_table(rows: list[dict[str, object]]) -> str:
    grid: list[list[str]] = []
    heads: list[bool] = []
    pending: dict[int, tuple[str, int]] = {}  # column -> (text, rows still covered)

    def carry(column: int, out: list[str]) -> None:
        text, left = pending.pop(column)
        out.append(text)
        if left > 1:
            pending[column] = (text, left - 1)

    for row in rows:
        cells = row["cells"]
        assert isinstance(cells, list)
        out: list[str] = []
        column = 0
        for cell in cells:
            while column in pending:
                carry(column, out)
                column += 1
            text = _flat(cell["text"]).replace("|", "&#124;")
            for _ in range(cell["colspan"]):
                out.append(text)
                if cell["rowspan"] > 1:
                    pending[column] = (text, cell["rowspan"] - 1)
                column += 1
        while pending and column <= max(pending):
            if column in pending:
                carry(column, out)
            else:
                out.append("")
            column += 1
        if any(out):
            grid.append(out)
            heads.append(bool(row["head"]) or all(cell["th"] for cell in cells))
    if not grid:
        return ""
    header_rows = next((i for i, head in enumerate(heads) if not head), len(heads)) or 1
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    header = [
        " / ".join(dict.fromkeys(r[c] for r in grid[:header_rows] if r[c])) for c in range(width)
    ]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(r) + " |" for r in grid[header_rows:])
    return "\n".join(lines)


def download(output: Path) -> None:
    raw = output / "raw"
    corpus = output / "documents"
    raw.mkdir(parents=True, exist_ok=True)
    corpus.mkdir(parents=True, exist_ok=True)
    records = []
    with httpx.Client(timeout=60, trust_env=False, follow_redirects=False) as client:
        for name, url in SOURCES.items():
            response = client.get(url)
            response.raise_for_status()
            content = response.content
            if not content or len(content) > 20 * 1024 * 1024:
                raise ValueError("Unexpected document size")
            original = raw / (name + (".txt" if url.endswith(".txt") else ".html"))
            original.write_bytes(content)
            text = content.decode("utf-8-sig")
            if url.endswith(".html"):
                extractor = TextExtractor()
                extractor.feed(text)
                text = extractor.text()
            target = corpus / (name + (".txt" if url.endswith(".txt") else ".md"))
            target.write_text(text, encoding="utf-8")
            records.append(
                {
                    "url": url,
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "raw_path": str(original),
                    "document_path": str(target),
                    "raw_sha256": hashlib.sha256(content).hexdigest(),
                    "document_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "edition_note": "User-selected historical 800-63-3 edition, superseded by 800-63-4"
                    if name.startswith("nist")
                    else "Snapshot of specified source URL",
                }
            )
    (output / "corpus_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=Path("data"))
    args = cli.parse_args(argv)
    download(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
