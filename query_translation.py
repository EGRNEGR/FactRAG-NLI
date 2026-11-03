"""Question translation into the language of the sources after a refusal (query_translation).

An English question to a Russian corpus (XQuAD EN questions over the RU paragraphs) is often
refused: the reranker scores the cross-language pair low (relevance gate) or the LLM does not
match the question's terms with the sources' (LLM refusal). After such a refusal the question
is translated by the LLM into the script of the best sources, retrieved again and answered
once more. The translation is only the query of retrieval and drafting: the answer is verified
against the sources as usual and published only if verification passes.

optimization_4, experiment G3 (reports/optimization_4): on the 151 questions of run
20261103T1310Z this rule selects, +5 / -0 correct (4 by the scorer, one scorer miss), no
unanswerable question published.
"""

from __future__ import annotations

import re
from typing import Literal

Script = Literal["ru", "en"]

_CYRILLIC = re.compile(r"[а-яё]", re.I)
_LATIN = re.compile(r"[a-z]", re.I)

TRANSLATE = (
    "Переведи вопрос пользователя на {language}. Сохрани смысл и все предпосылки вопроса, "
    "имена, числа, коды и идентификаторы. Ничего не добавляй и не отвечай на вопрос. "
    "Верни только перевод вопроса одной строкой."
)
LANGUAGE: dict[Script, str] = {"ru": "русский язык", "en": "английский язык"}
# The refusals after which the question is translated: the relevance gate and the LLM's own.
TRIGGERS = frozenset({"relevance_threshold_not_met", "llm_refusal"})
# Gate refusal: the script of the sources is read from this many best-reranked candidates.
GATE_CANDIDATES = 5


def text_script(text: str) -> Script | None:
    """The dominant script of a text by letter count (Cyrillic wins ties); None without letters."""
    cyrillic, latin = len(_CYRILLIC.findall(text)), len(_LATIN.findall(text))
    if not cyrillic and not latin:
        return None
    return "ru" if cyrillic >= latin else "en"


def translation_prompt(target: Script) -> str:
    return TRANSLATE.format(language=LANGUAGE[target])


def parse_rewrite(reply: str, question: str) -> str | None:
    """The first non-empty line of the reply without quotes; None if empty or far longer than
    the question (the model answered instead of translating)."""
    line = next((x.strip() for x in reply.splitlines() if x.strip()), "")
    line = line.strip("«»\"' ")
    if not line or len(line) > 3 * len(question) + 40:
        return None
    return line
