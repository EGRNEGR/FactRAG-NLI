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
