"""Incrementally ingest user files; embedded storage requires exclusive process access."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_processor import DocumentProcessor, SUPPORTED_SUFFIXES, index_status
from hybrid_retriever import HybridRetriever, RetrievalError
from settings import RAGSettings


def main(argv: Sequence[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    action = cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--add", type=Path)
    action.add_argument("--scan", type=Path)
    action.add_argument("--status", action="store_true")
    args = cli.parse_args(argv)
    try:
        settings = RAGSettings()
        if args.status:
            print(json.dumps(asdict(index_status(settings))))
            return 0
        paths = [args.add] if args.add is not None else []
        if args.scan is not None:
            root = args.scan.expanduser().resolve(strict=True)
            if not root.is_dir():
                raise ValueError("--scan requires a directory")
            paths = sorted(
                p
                for p in root.rglob("*")
                if p.is_file() and not p.is_symlink() and p.suffix.lower() in SUPPORTED_SUFFIXES
            )
        failures = 0
        if not paths:
            print(json.dumps({"status": "empty_scan", "files": 0}))
            return 0
        if args.add is not None:
            candidate = args.add.expanduser().resolve(strict=True)
            if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError("Мы добавляем через --add файлы TXT, MD, PDF и DOCX")
        processor = DocumentProcessor(settings)
        with HybridRetriever(settings=settings) as index:
            for path in paths:
                try:
                    result = processor.add(path, index)
                    print(json.dumps(asdict(result), ensure_ascii=False), flush=True)
                except RetrievalError:
                    raise  # Storage/inference failures poison this client: reopen before retry.
                except (ValueError, OSError, RuntimeError) as exc:
                    failures += 1
                    print(
                        json.dumps(
                            {"source": str(path), "status": "error", "error": str(exc)},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        return 1 if failures else 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(
            json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
