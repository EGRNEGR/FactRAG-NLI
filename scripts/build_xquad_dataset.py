"""Build an external FactRAG-NLI test set from XQuAD (google/xquad, CC BY-SA 4.0).

XQuAD is 240 SQuAD v1.1 paragraphs from 48 Wikipedia articles with 1190 questions,
professionally translated into Russian; the RU and EN versions are parallel by question ID.
The dataset is converted to the v3 schema (eval_dataset.EvalDatasetV3) so that
evaluate_suite.py runs retrieval, NLI and end-to-end modes on it unchanged:

* corpus: the Russian paragraphs of 40 articles, one Markdown file per article
  (data/xquad/docs); 8 seeded articles are held out of the corpus;
* single_hop (answer): questions about corpus articles, half asked in Russian and half in
  English (cross-lingual retrieval); the evidence quote is the Russian sentence that holds
  the answer span, bound to production chunks by build_eval_dataset_v3.bind;
* unanswerable (refusal): questions about held-out articles whose answer string occurs
  nowhere in the corpus (checked by bind through absence_terms);
* NLI probes: entailment = the RU answer sentence (hypothesis_ru, sanity control) and its
  parallel EN sentence (hypothesis_en, cross-lingual); neutral = a held-out sentence against
  an unrelated corpus sentence.

Answer scoring: XQuAD answers are short spans, but Russian inflects them in a generated
answer, so answer_must_contain is one order-free regex over word stems of the RU span
(alternatively the EN span).

Usage:
  python scripts/build_xquad_dataset.py            # download if needed, build, write JSON
  python scripts/build_xquad_dataset.py --check    # rebuild and require identical JSON
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_dataset import (  # noqa: E402
    CorpusEntry,
    DraftCase,
    EvalDatasetV3,
    NLIProbeDraft,
    normalize,
)
from scripts.build_eval_dataset_v3 import CHUNKING, bind, parse_all, statistics  # noqa: E402

BASE = ROOT / "data" / "xquad"
RAW = BASE / "raw"
DOCS = BASE / "docs"
OUTPUT = ROOT / "data" / "xquad_dataset.json"
CACHE = ROOT / ".artifacts" / "xquad_parse_cache"
SOURCES = {
    "xquad.ru.parquet": "https://huggingface.co/datasets/google/xquad/resolve/main/"
    "xquad.ru/validation-00000-of-00001.parquet",
    "xquad.en.parquet": "https://huggingface.co/datasets/google/xquad/resolve/main/"
    "xquad.en/validation-00000-of-00001.parquet",
    # Article titles: XQuAD keeps the SQuAD v1.1 question IDs but drops the title column.
    "squad_v1.validation.parquet": "https://huggingface.co/datasets/rajpurkar/squad/resolve/"
    "main/plain_text/validation-00000-of-00001.parquet",
}
SEED = 42
HELD_OUT_ARTICLES = 8
PER_ARTICLE_ANSWERABLE = 4
PER_ARTICLE_UNANSWERABLE = 5
MAX_ANSWER_CHARS = 60
# A boundary needs three word characters before the stop: initials and abbreviations
# ("The T. Rex", "St. Louis", "т.е.") do not end a sentence.
SENTENCE_END = re.compile(r"(?<=[\w)»\"]{3}[.!?…])\s+(?=[A-ZА-ЯЁ0-9«\"(])")


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        target = RAW / name
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"Downloading {url}", file=sys.stderr, flush=True)
        partial = target.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
            out.write(response.read())
        partial.replace(target)


def load_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    import pyarrow.parquet as pq

    def rows(name: str) -> list[dict[str, Any]]:
        table: list[dict[str, Any]] = pq.read_table(RAW / name).to_pylist()
        return table

    ru = rows("xquad.ru.parquet")
    en = {row["id"]: row for row in rows("xquad.en.parquet")}
    titles = {row["id"]: row["title"] for row in rows("squad_v1.validation.parquet")}
    for row in ru:
        row["context"] = row["context"].lstrip("﻿")
    missing = [row["id"] for row in ru if row["id"] not in en or row["id"] not in titles]
    if missing:
        raise ValueError(f"{len(missing)} XQuAD IDs without EN pair or title: {missing[:3]}")
    return ru, en, titles


def sentence_at(text: str, offset: int) -> str:
    """The sentence of `text` that contains character `offset`."""
    start = 0
    for match in SENTENCE_END.finditer(text):
        if match.start() > offset:
            return text[start : match.start()].strip()
        start = match.end()
    return text[start:].strip()


def stem(token: str) -> str:
    if any(ch.isdigit() for ch in token):
        return rf"(?<!\d){re.escape(token)}(?!\d)"
    if len(token) >= 6:
        return rf"\b{re.escape(token[:-2])}\w*"
    if len(token) >= 4:
        return rf"\b{re.escape(token[:-1])}\w*"
    return rf"\b{re.escape(token)}\w*"


def span_pattern(answer: str) -> str:
    """Order-free conjunction of the stems of the (at most 4 longest) content words."""
    words = re.findall(r"\w+", normalize(answer))
    # Function words ("the", "of", "в", "год") are skipped unless nothing else is left.
    content = [w for w in words if len(w) > 3 or any(ch.isdigit() for ch in w)] or words
    digits = [w for w in content if any(ch.isdigit() for ch in w)]
    longest = sorted((w for w in content if w not in digits), key=len, reverse=True)
    chosen = list(dict.fromkeys(digits + longest[: max(0, 4 - len(digits))]))
    return "".join(rf"(?=[\s\S]*{stem(w)})" for w in chosen)


def answer_pattern(ru_answer: str, en_answer: str) -> str:
    options = dict.fromkeys(span_pattern(answer) for answer in (ru_answer, en_answer))
    return "re:^(?:" + "|".join(o for o in options if o) + ")"


def slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").lower()


def write_corpus(articles: dict[str, list[str]], keep: Sequence[str]) -> list[str]:
    """One Markdown file per corpus article; returns file names relative to DOCS."""
    DOCS.mkdir(parents=True, exist_ok=True)
    names = []
    for title in keep:
        name = f"{slug(title)}.md"
        body = f"# {title.replace('_', ' ')}\n\n" + "\n\n".join(articles[title]) + "\n"
        target = DOCS / name
        if not target.exists() or target.read_text(encoding="utf-8") != body:
            target.write_text(body, encoding="utf-8", newline="\n")
        names.append(name)
    stale = {p.name for p in DOCS.glob("*.md")} - set(names)
    for name in stale:
        (DOCS / name).unlink()
    return names


def drafts(
    ru: list[dict[str, Any]],
    en: dict[str, dict[str, Any]],
    titles: dict[str, str],
    keep: Sequence[str],
    held_out: Sequence[str],
    corpus_norm: str,
) -> list[DraftCase]:
    rng = random.Random(SEED)
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ru:
        by_article[titles[row["id"]]].append(row)
    doc_of = {title: f"{slug(title)}.md" for title in keep}
    cases: list[DraftCase] = []

    def en_sentence(qid: str) -> tuple[str, str]:
        row = en[qid]
        return row["answers"]["text"][0], sentence_at(
            row["context"], row["answers"]["answer_start"][0]
        )

    for title in keep:
        pool = [
            r
            for r in by_article[title]
            if len(r["answers"]["text"][0]) <= MAX_ANSWER_CHARS
            and len(sentence_at(r["context"], r["answers"]["answer_start"][0])) >= 12
            and len(en_sentence(r["id"])[1]) >= 12
        ]
        for i, row in enumerate(rng.sample(pool, min(PER_ARTICLE_ANSWERABLE, len(pool)))):
            answer = row["answers"]["text"][0]
            quote = sentence_at(row["context"], row["answers"]["answer_start"][0])
            en_answer, en_quote = en_sentence(row["id"])
            language: Literal["ru", "en"] = "ru" if i % 2 == 0 else "en"
            cases.append(
                DraftCase(
                    id=f"V3-XQ{language.upper()}-{len(cases) + 1:03d}",
                    category="single_hop",
                    subtype=f"xquad_span_{language}",
                    language=language,
                    document=doc_of[title],
                    question=row["question"] if language == "ru" else en[row["id"]]["question"],
                    ground_truth_answer=answer,
                    expected_behavior="answer",
                    quotes=[quote],
                    nli_probe=NLIProbeDraft(
                        premise_quote=quote,
                        hypothesis_en=en_quote,
                        hypothesis_ru=quote,
                        label="entailment",
                    ),
                    answer_must_contain=[answer_pattern(answer, en_answer)],
                )
            )
    premises = [
        (doc_of[title], s)
        for title in keep
        for paragraph in dict.fromkeys(r["context"] for r in by_article[title])
        for s in SENTENCE_END.split(paragraph)
        if len(s) >= 40
    ]
    for title in held_out:
        candidates: list[tuple[dict[str, Any], list[str], str]] = []
        for row in by_article[title]:
            answer = row["answers"]["text"][0]
            en_answer, en_quote = en_sentence(row["id"])
            terms = [re.escape(normalize(answer))]
            if len(en_answer) >= 3 and normalize(en_answer) != normalize(answer):
                terms.append(re.escape(normalize(en_answer)))
            if len(answer) >= 3 and not any(re.search(t, corpus_norm) for t in terms):
                candidates.append((row, terms, en_quote))
        for row, terms, en_quote in rng.sample(
            candidates, min(PER_ARTICLE_UNANSWERABLE, len(candidates))
        ):
            language = "ru" if len(cases) % 2 == 0 else "en"
            document, premise = premises[rng.randrange(len(premises))]
            cases.append(
                DraftCase(
                    id=f"V3-XQNA-{len(cases) + 1:03d}",
                    category="unanswerable",
                    subtype=f"out_of_corpus_{language}",
                    language=language,
                    document=document,
                    question=row["question"] if language == "ru" else en[row["id"]]["question"],
                    ground_truth_answer=(
                        f"В корпусе нет сведений: статья «{title.replace('_', ' ')}» "
                        "исключена из индекса."
                    ),
                    expected_behavior="refusal",
                    nli_probe=NLIProbeDraft(
                        premise_quote=premise,
                        hypothesis_en=en_quote,
                        hypothesis_ru=sentence_at(
                            row["context"], row["answers"]["answer_start"][0]
                        ),
                        label="neutral",
                    ),
                    absence_terms=terms,
                )
            )
    return cases


def main(argv: Sequence[str] | None = None) -> int:
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cli.add_argument("--check", action="store_true", help="Require the rebuilt JSON to match")
    args = cli.parse_args(argv)
    download()
    ru, en, titles = load_rows()
    articles: dict[str, list[str]] = defaultdict(list)
    for row in ru:
        paragraphs = articles[titles[row["id"]]]
        if row["context"] not in paragraphs:
            paragraphs.append(row["context"])
    order = list(articles)
    held_out = sorted(random.Random(SEED).sample(order, HELD_OUT_ARTICLES))
    keep = [title for title in order if title not in held_out]
    files = write_corpus(articles, keep)
    parsed = parse_all(files, corpus=DOCS, cache=CACHE)
    corpus_norm = "\n".join(normalize(c["text"]) for r in parsed.values() for c in r["chunks"])
    draft_cases = drafts(ru, en, titles, keep, held_out, corpus_norm)
    cases, errors = bind(draft_cases, parsed)
    if errors:
        # A quote split across a chunk boundary cannot be bound; such cases are dropped.
        print(f"Dropped {len(errors)} case(s):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
    stats = statistics(cases)
    stats["held_out_articles"] = held_out
    dataset = EvalDatasetV3(
        description=(
            "External FactRAG-NLI test set built from XQuAD (google/xquad, CC BY-SA 4.0): "
            "Russian Wikipedia paragraphs of 40 articles as the corpus, RU and EN span questions "
            "(single_hop) and out-of-corpus questions about 8 held-out articles (unanswerable). "
            "Generated by scripts/build_xquad_dataset.py."
        ),
        corpus_root=DOCS.relative_to(ROOT).as_posix(),
        chunking={**CHUNKING, "tokenizer": "models/bge-m3 (BAAI/bge-m3)"},
        documents=[
            CorpusEntry(
                path=name,
                sha256=parsed[name]["sha256"],
                document_id=parsed[name]["document_id"],
                chunks=len(parsed[name]["chunks"]),
            )
            for name in files
        ],
        cases=cases,
        stats=stats,
    )
    text = dataset.model_dump_json(indent=1) + "\n"
    print(
        json.dumps(
            {k: v for k, v in stats.items() if k != "by_document"}, ensure_ascii=False, indent=1
        )
    )
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != text:
            print(f"{OUTPUT.relative_to(ROOT)} differs from the rebuild", file=sys.stderr)
            return 1
        print(f"{OUTPUT.relative_to(ROOT)} matches the rebuild", file=sys.stderr)
        return 0
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({len(cases)} cases)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
