"""Text-level evidence helpers for claim verification: pure functions, no models.

Sentence segmentation, citation inheritance, NLI premise cleanup and windows, lexical
anchors (numbers, codes, identifiers), verbatim code spans and meta-statement detection.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Literal

SegmentationVersion = Literal["v1", "v2"]

_CITATION = re.compile(r"\[S([1-9]\d*)\]")
_ANY_CITATION = re.compile(r"\[S[^\]\n]*\]")

# v1: the original conservative Russian-only abbreviation list.
_PROTECT_V1 = re.compile(r"\b(?:пп?|рис|табл|стр|см|им|г)\.|\bт\.\s*[ед]\.", re.I)
# v2: abbreviations that never end a factual sentence in technical text ...
_ALWAYS = (
    r"(?:пп?|рис|табл|стр|см|им|г|ст|гл|разд|прил|др|пр|т\.\s*[едпкч]|"
    r"e\.\s*g|i\.\s*e|cf|vs|al|approx|resp|incl|esp|dr|mr|mrs|ms|prof|jr|sr|inc|ltd|corp)\."
)
# ... and numbered references that are abbreviations only before a number ("Rev. 5", "No. 1").
_NUMBERED = (
    r"(?:rev|no|nos|sec|sect|secs|fig|figs|vol|ch|pt|pts|art|par|para|p|v|ver|eq|ex|tab)\."
    r"(?=\s*(?:\d|[IVX]+\b))"
)
_PROTECT_V2 = re.compile(
    rf"\b{_ALWAYS}|\b{_NUMBERED}|\betc\.(?=\s*[a-zа-яё0-9,;)])|\.{{3,}}|…", re.I
)
_TAIL_V1 = re.compile(r"(?:\s*\[S[^\]\n]*\])+")
_TAIL_V2 = re.compile(r"(?:\s*,?\s*\[S[^\]\n]*\])+")


def segment_claims(text: str, version: SegmentationVersion = "v1") -> list[str]:
    """Split sentences, preserving decimals, section numbers and trailing citations.

    v1 splits on semicolons and protects only Russian abbreviations. v2 keeps a sentence
    with semicolons whole, protects English numbered references ("SP 800-53 Rev. 5"),
    common abbreviations and ellipses, and attaches comma-separated citations.
    Compound propositions inside one sentence are not semantically atomized.
    """
    text = re.sub(r"^\s*(?:ответ|answer)\s*:\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", "", text)
    text = re.sub(r"(?m)^\s*```[^\n]*$", "", text)
    legacy = version == "v1"
    protected: set[int] = set()
    for match in (_PROTECT_V1 if legacy else _PROTECT_V2).finditer(text):
        protected.update(
            i for i in range(match.start(), match.end()) if not legacy or text[i] == "."
        )
    breaks = ".!?;\n。！？" if legacy else ".!?\n。！？"
    tail_pattern = _TAIL_V1 if legacy else _TAIL_V2
    pieces: list[str] = []
    start, cursor = 0, 0
    while cursor < len(text):
        char = text[cursor]
        if char not in breaks or cursor in protected:
            cursor += 1
            continue
        if (
            char == "."
            and cursor > 0
            and cursor + 1 < len(text)
            and text[cursor - 1].isdigit()
            and text[cursor + 1].isdigit()
        ):
            cursor += 1
            continue
        end = cursor + 1
        # Consume a run of punctuation and following citations with the preceding fact.
        while end < len(text) and text[end] in ".!?。！？" and (legacy or end not in protected):
            end += 1
        tail = tail_pattern.match(text[end:])
        if tail:
            end += tail.end()
        if char != "\n" and end < len(text) and not text[end].isspace() and text[end] != "[":
            cursor += 1
            continue
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        start = cursor = end
    if text[start:].strip():
        pieces.append(text[start:].strip())
    return pieces


@dataclass(frozen=True, slots=True)
class CitedClaim:
    text: str
    source_ids: tuple[str, ...]
    inherited: bool = False


def own_citations(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(f"S{match}" for match in _CITATION.findall(text)))


def cited_claims(
    text: str, version: SegmentationVersion = "v2", *, inherit: bool = True
) -> list[CitedClaim]:
    """Segment an answer and give uncited sentences the citation of their neighbours.

    LLMs often cite only the last sentence of a paragraph or the first one of a list.
    An uncited sentence inherits, in this order, the nearest following and the nearest
    preceding citation in the same paragraph, then the nearest preceding and following
    citation in the answer. An answer without any citation stays uncited.
    """
    if not inherit:
        return [CitedClaim(c, own_citations(c)) for c in segment_claims(text, version)]
    items = [
        (index, claim)
        for index, block in enumerate(b for b in re.split(r"\n+", text) if b.strip())
        for claim in segment_claims(block, version)
    ]
    own = [own_citations(claim) for _, claim in items]
    result = []
    for i, (paragraph, claim) in enumerate(items):
        ids, inherited = own[i], False
        if not ids and not _ANY_CITATION.search(claim):
            after = range(i + 1, len(items))
            before = range(i - 1, -1, -1)
            for scope in (
                (j for j in after if items[j][0] == paragraph),
                (j for j in before if items[j][0] == paragraph),
                iter(before),
                iter(after),
            ):
                found = next((own[j] for j in scope if own[j]), ())
                if found:
                    ids, inherited = found, True
                    break
        result.append(CitedClaim(claim, ids, inherited))
    return result


# ----------------------------------------------------------------------------- premises

_HEADER = re.compile(r"^Документ:[^\n]*\nРаздел:[^\n]*\n")
_MARKUP = re.compile(
    r"</?(?:a|abbr|b|blockquote|br|code|dd|div|dl|dt|em|h[1-6]|hr|i|img|kbd|li|ol|p|pre|q|s|"
    r"samp|section|small|span|strong|sub|sup|table|tbody|td|th|thead|tr|tt|u|ul|var|nav|header|"
    r"footer|main|article|aside|figure|figcaption|caption)\b[^<>]*>",
    re.I,
)


def clean_premise(text: str, *, strip_markup: bool = True) -> str:
    """Drop the parser header, HTML markup and entities; collapse whitespace."""
    text = _HEADER.sub("", text)
    if strip_markup:
        text = html.unescape(_MARKUP.sub(" ", text))
    return " ".join(text.split())


def premise_windows(premise: str, max_sentences: int = 2) -> list[str]:
    """Contiguous windows of 1..max_sentences sentences, excluding the whole premise."""
    sentences = segment_claims(premise, "v2")
    windows = [
        " ".join(sentences[i : i + size])
        for size in range(1, max_sentences + 1)
        for i in range(len(sentences) - size + 1)
    ]
    return [w for w in dict.fromkeys(windows) if w != premise]


# ----------------------------------------------------------------------------- titles

_HTML_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_RFC_NUMBER = re.compile(r"Request for Comments:\s*(\d+)")
_LETTER = re.compile(r"[^\W\d_]")


def title_from_text(head: str, metadata_title: str | None = None) -> str | None:
    """Document title from the head of its text, by generic rules (no per-document table).

    HTML <title> without the site suffix; an RFC header gives "RFC N: <line before
    Abstract>"; otherwise the first non-empty line, with a different metadata title
    (PDF/DOCX properties) appended.
    """
    head = head.lstrip(chr(0xFEFF))
    match = _HTML_TITLE.search(head)
    if match:
        title = " ".join(html.unescape(match.group(1)).split()).split(" | ")[0]
        return title[:160] or None
    lines = [line.strip().lstrip("#").strip() for line in head.splitlines()]
    lines = [line for line in lines if line]
    rfc = _RFC_NUMBER.search(head)
    if rfc:
        name = next(
            (prev for prev, line in zip(lines, lines[1:]) if line.casefold() == "abstract"), None
        )
        return f"RFC {rfc.group(1)}: {name}" if name else f"RFC {rfc.group(1)}"
    first = next((line for line in lines if len(_LETTER.findall(line)) >= 3), None)
    if first and metadata_title and metadata_title.casefold() not in first.casefold():
        first = f"{first}: {metadata_title}"
    return first[:160] if first else metadata_title or None


def premise_header(title: str | None, section_path: tuple[str, ...] | list[str]) -> str:
    """'Title. Section.' prefix for an NLI premise; noisy section names are dropped.

    Parser section paths of plain-text and PDF documents sometimes hold ASCII diagrams or
    sentence fragments; a section name is kept only if letters make up half of it.
    """
    section = section_path[-1].strip() if section_path else ""
    letters = len(_LETTER.findall(section))
    if letters < 3 or letters < 0.5 * len(section.replace(" ", "")):
        section = ""
    parts = [part.strip().rstrip(".") for part in (title, section) if part and part.strip()]
    return ". ".join(parts) + "." if parts else ""


_WORD = re.compile(r"\w+")
_STOPWORDS = frozenset(
    "the and for are with that this from its has have been was were will can all any each "
    "into than then which when also such their there these those not but only must should "
    "may shall это для что как при или его она они если также быть может должен должна "
    "должны которые который которая только тоже".split()
)


def content_overlap(claim: str, source: str) -> float:
    """Share of the claim's content words (3+ characters, or containing a digit) in `source`."""
    words = {
        word.casefold()
        for word in _WORD.findall(_ANY_CITATION.sub(" ", claim))
        if (len(word) >= 3 or any(ch.isdigit() for ch in word))
        and word.casefold() not in _STOPWORDS
    }
    if not words:
        return 0.0
    vocabulary = {word.casefold() for word in _WORD.findall(source)}
    return sum(word in vocabulary for word in words) / len(words)


# ----------------------------------------------------------------------------- anchors

# Document designators are provenance, not facts the chunk text has to contain.
_DESIGNATOR = re.compile(
    r"\b(?:NIST\s+)?(?:RFC|SP|PEP|FIPS|ISO|IEC|IETF|PostgreSQL|Python|CPython|СП)\s*-?\s*"
    r"\d+(?:[-.][0-9A-Za-z]+)*(?:\s*,?\s*(?:Rev\.?|Revision|редакци\w*)\s*\d+)?"
    r"|\b(?:Rev|Revision|Version|Ver)\.?\s*\d+\b|\b(?:редакци|верси)\w*\s+\d+\b",
    re.I,
)
_ANCHOR = re.compile(
    r"(?<![\w-])(?:"
    r"0x[0-9a-fA-F]+|0b[01]+"  # hex and binary literals
    r"|\d+(?:[ ,.  ]\d+)*"  # numbers with group separators or decimals
    r"|--[A-Za-z][\w-]*"  # command-line flags
    r"|[A-Za-z]\w*(?:[_=/+][\w.+-]*)+"  # max_ack_delay, PYTHONGIL=0, application/problem+json
    r"|[A-Za-z]+-\d+[A-Za-z]?|[A-Za-z]+\d\w*"  # AC-2k, SI-2, UUIDv7, IPv6
    r"|[a-z]+[A-Z]\w*"  # camelCase: traceId
    r")(?![\w])"
)
_GROUPED = re.compile(r"(?<=\d)[ ,  ](?=\d{3}\b)")


def claim_anchors(text: str) -> frozenset[str]:
    """Numbers, literals and code identifiers that a supporting text must contain."""
    text = _DESIGNATOR.sub(" ", _ANY_CITATION.sub(" ", text))
    found = {_GROUPED.sub("", m.group(0)).casefold().rstrip(".,") for m in _ANCHOR.finditer(text)}
    return frozenset(found - {""})


def anchors_present(anchors: frozenset[str], text: str) -> bool:
    haystack = _GROUPED.sub("", text).casefold()
    for anchor in anchors:
        if anchor.replace(".", "").isdigit():
            if not re.search(rf"(?<![\w.]){re.escape(anchor)}(?!\w)", haystack):
                return False
        elif anchor not in haystack:
            return False
    return True


# ----------------------------------------------------------------------------- code spans

_QUOTES = str.maketrans({"“": '"', "”": '"', "«": '"', "»": '"', "‘": "'", "’": "'", "\\": ""})
_OPERATOR = re.compile(r"[=*/#()\[\]<>+_]")


def _code_normal(text: str) -> str:
    text = " ".join(text.translate(_QUOTES).split())
    return re.sub(r"\s*([()\[\]/=,*#\"<>+|])\s*", r"\1", text)


def verbatim_code_span(hypothesis: str, premise: str, min_chars: int = 12) -> bool:
    """True if an ASCII code-like span of the claim (ABNF rule, assignment, literal)
    occurs verbatim in the premise, ignoring whitespace around operators."""
    if min_chars <= 0:
        return False
    haystack = _code_normal(premise)
    for match in re.finditer(r"[\x20-\x7e]{%d,}" % min_chars, hypothesis.translate(_QUOTES)):
        span = " ".join(match.group(0).split()).strip(" .,;:")
        if len(span) >= min_chars and _OPERATOR.search(span) and len(span.split()) >= 2:
            if _code_normal(span) in haystack:
                return True
    return False


# ----------------------------------------------------------------------------- meta statements

_ABSENCE = re.compile(
    r"(?:не\s+(?:содерж\w*|приводит\w*|предоставля\w*)\s+(?:\w+\s+){0,3}?"
    r"(?:информаци|сведени|данных|указани)\w*"
    r"|(?:отсутству\w*|нет)\s+(?:\w+\s+){0,2}?(?:сведени|информаци|данн|упоминани|указани)\w*"
    r"|не\s+(?:упомина|рассматрива|освеща)\w*"
    r"|no\s+(?:specific\s+|direct\s+)?(?:information|details|mention)"
    r"|does\s+not\s+(?:contain|provide|include)\s+(?:any\s+)?(?:specific\s+)?"
    r"(?:information|details|data)"
    r"|(?:is|are)\s+not\s+(?:mentioned|covered|discussed)"
    r"|does\s+not\s+(?:mention|discuss))",
    re.I,
)
_VERDICT = re.compile(r"^(?:нет|неверно|это\s+не\s+так|no|incorrect|false)\b[,.!:]?", re.I)
_NEGATION = re.compile(r"\b(?:не|нет|not|no|never)\b", re.I)


def is_absence_statement(text: str) -> bool:
    """A statement that the sources lack information (not a domain negation)."""
    return bool(_ABSENCE.search(text))


def is_verdict_marker(text: str) -> bool:
    """A bare verdict such as "Нет." or "Неверно." that only introduces the next claim."""
    words = re.findall(r"\w+", _ANY_CITATION.sub("", text))
    return bool(_VERDICT.match(text.strip())) and len(words) <= 3


def has_negation(text: str) -> bool:
    return bool(_NEGATION.search(text))
