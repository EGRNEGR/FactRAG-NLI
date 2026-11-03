"""DIAGNOSTIC_REPORT.md of optimization_4: one section per problem question of run 1310Z.

Evidence comes from the dossiers of research.opt4.diagnose (reports/optimization_4/
diagnostics/*.jsonl); the cause and the proposed fix of each question are hand-written in
research/opt4/annotations.json after reading the dossier. A question without an annotation
is an error (the report must not ship an empty cause).

Usage:  python -m research.opt4.report
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from research.opt4.diagnose import OUT, load

ANNOTATIONS = Path(__file__).with_name("annotations.json")
ROLE = {
    "draft": "черновик",
    "repair": "правка",
    "think": "рассуждение",
    "think_repair": "правка с рассуждением",
    "recovery": "восстановление (T=0.7)",
}
OUTCOME = {"fixed": "исправлен", "already_correct": "уже был верен", "still_wrong": "остался неверным"}


def quote(text: str | None) -> str:
    if not text:
        return "> (нет)"
    return "\n".join("> " + line if line.strip() else ">" for line in text.strip().splitlines())


def criteria(d: dict[str, Any]) -> str:
    facts = ", ".join(f"`{f}`" for f in d["answer_must_contain"]) or "-"
    if d["expected"] == "answer":
        return f"верен, если опубликованный текст содержит все факты: {facts}"
    if d["expected"] == "refusal":
        return "верен отказ или явное «нет сведений»"
    return f"верен отказ/поправка предпосылки или все факты: {facts}"


def withheld(attempt: dict[str, Any]) -> list[str]:
    lines = []
    for c in attempt["claims"]:
        if c["status"] == "verified":
            continue
        missing = ", ".join(c["anchors_missing_in_cited"]) or "-"
        ev = "; ".join(
            f"{e['source_id']}: mDeBERTa E={e['p_entailment']} C={e['p_contradiction']}, "
            f"bge-m3 E={e['secondary_entailment']}"
            for e in c["evidence"]
        )
        lines.append(
            f"  - «{c['claim']}» - {c['status']}, {c['reason']}, степень {c['grade']}, "
            f"судья {c['judge_score']}, источники {', '.join(c['cited']) or '-'}, "
            f"якоря не в цитируемом: {missing}" + (f" ({ev})" if ev else "")
        )
    return lines


def section(d: dict[str, Any], note: dict[str, Any], codes: dict[str, str]) -> list[str]:
    head = f"### {d['id']} ({d['split']}, {d['category']}/{d['subtype']}, {d['language']})"
    outcome = OUTCOME.get(d.get("think_outcome", ""), "")
    lines = [
        head,
        "",
        f"Итог прогона: {'верно' if d['correct'] else 'неверно'}, класс `{d['class']}`, путь "
        f"`{d['path']}`" + (f", рассуждение после обрезки: {outcome}" if outcome else "") + ".",
        "",
        f"**Вопрос.** {d['question']}",
        "",
        f"**Эталон.** {d['ground_truth']}",
        "",
        f"**Критерий оценки.** {criteria(d)}.",
        "",
        "**Найденный контекст.** "
        + (
            f"золото на позициях {', '.join(map(str, d['gold_ranks']))}, "
            if d["gold_ranks"]
            else "золота нет в топ-10, "
        )
        + f"лучший балл реранкера {d['top_rerank']}; "
        + ", ".join(f"{s['source_id']} {s['rerank']}{' (золото)' if s['gold'] else ''}"
                    for s in d["sources"]) + ".",
        "",
        "**Релевантные фрагменты.**",
    ]
    for f in d["fragments"]:
        state = "передан LLM" if f["given_to_llm"] else "не передан LLM"
        lines.append(f"- `{f['chunk_id']}` ({state}): «{f['sentence'] or f['quote']}»")
    lines.append("")
    for a in d["attempts"]:
        verdict = "верен" if a["draft_correct"] else "неверен"
        lines += [
            f"**{ROLE.get(a['role'], a['role']).capitalize()}** ({a['completion_tokens']} токенов, "
            f"{a['generation_ms'] / 1000:.1f} с; текст сам по себе {verdict}):",
            "",
            quote(a["raw_text"]),
            "",
        ]
        removed = withheld(a)
        if a["llm_refusal"]:
            lines += ["Отказ LLM.", ""]
        elif removed:
            lines += ["Удалённые утверждения:", *removed, ""]
            if a["published_text"] is not None:
                lines += ["Опубликовано после удаления:", "", quote(a["published_text"]), ""]
            else:
                lines += ["Ответ не опубликован.", ""]
    lines += [
        f"**Опубликованный ответ.** {d['answer'] if d['answer'] else '(отказ)'}",
        "",
        "**Причина.** " + ("; ".join(codes[c] for c in note["code"]) + ". " if note["code"] else "")
        + note["cause"],
        "",
        f"**Способ исправления.** {note['fix']}",
        "",
    ]
    return lines


def main() -> None:
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    codes, notes = annotations["codes"], annotations["cases"]
    think, refusals = load("think_on_trim"), load("xquad_refusals")
    missing = [d["id"] for d in think + refusals if d["id"] not in notes]
    if missing:
        raise SystemExit(f"no annotation for {missing}")
    outcomes = Counter(d["think_outcome"] for d in think)
    primary = Counter(notes[d["id"]]["code"][0] for d in refusals)
    any_code = Counter(c for d in refusals for c in notes[d["id"]]["code"])
    languages = Counter(d["language"] for d in refusals)
    lines = [
        "# DIAGNOSTIC_REPORT: проблемные вопросы прогона 20261103T1310Z",
        "",
        "Базовая линия - живой прогон `20261103T1310Z_rtx3090_main` (judge, код f083e2d): 1184 из "
        "1307 верных (90.6%), v3 88.6%, XQuAD 91.0%, галлюцинации 3 из 269, ложные отказы 79. "
        "Разбор построен по его логам (`research/opt4/diagnose.py`, досье в "
        "`diagnostics/*.jsonl`); причины и способы исправления записаны вручную после чтения "
        "досье (`research/opt4/annotations.json`). Логи прогона не изменялись.",
        "",
        "## 1. Рассуждение после обрезки (llm_think_on_trim)",
        "",
        f"Правило сработало на {len(think)} вопросах: исправлено {outcomes['fixed']}, "
        f"было верным и осталось {outcomes['already_correct']}, осталось неверным "
        f"{outcomes['still_wrong']}. «Не помогло» - {len(think) - outcomes['fixed']} вопросов: "
        f"{outcomes['already_correct']} уже были верны (рассуждение в 3 из них заменило ответ "
        "более полным проверенным, в 4 ответ остался прежним) и "
        f"{outcomes['still_wrong']} неверных. У пяти неверных пять разных механизмов, ни один не "
        "лечится повторным рассуждением на том же контексте:",
        "",
        "| Вопрос | Что мешает | Где |",
        "|---|---|---|",
        "| V3-MH-024 | метка AC-7(4) в соседнем чанке, якорь не найден (судья 0.643 выше порога) | верификатор, якоря |",
        "| V3-MH-002 | NLI ставит противоречие 0.98 верному утверждению с «MAY» | NLI-модель |",
        "| V3-MH-018 | перевод «SHALL NOT impose» оценён судьёй 0.038; ответ с рассуждением смешал языки | NLI, язык ответа |",
        "| V3-MH-021 | чанк третьей части вопроса не в топ-10 | поиск |",
        "| V3-MH-036 | значение PYTHONGIL=1 отрезано границей чанка, следующий чанк не найден | чанкинг, поиск |",
        "",
        "## 2. Упорные отказы на XQuAD",
        "",
        f"{len(refusals)} отвечаемых вопросов XQuAD, где все попытки (черновик, рассуждение, "
        f"восстановление) - канонический отказ; {languages['ru']} на русском, {languages['en']} "
        "на английском. Доля отказов почти одинакова для языков (RU 28 из 464 = 6.0%, EN 24 из "
        "447 = 5.4%), а балл реранкера у золота отказанных вопросов намного ниже (медиана EN "
        "0.089 против 0.802 у остальных, RU 0.588 против 0.988): поиск сам видит слабую связь "
        "вопроса с текстом. Бюджет рассуждения не причина: после отказа отказ дали 83 из 121 "
        "прохода, не дошедших до 640 токенов, и 53 из 120 исчерпавших его.",
        "",
        "Главная причина каждого вопроса (первый код аннотации):",
        "",
        "| Причина | Вопросов | Устранима без риска для предпосылок |",
        "|---|---|---|",
    ]
    fixable = {
        "premise_not_in_context": "нет: отказ обоснован",
        "dataset_noise": "нет: ошибка эталона или перевода",
        "question_not_standalone": "нет: вопросу нужен контекст статьи",
        "strict_premise_synonym": "только ослаблением правила предпосылки (риск)",
        "question_mt_garbled": "да, гипотеза H1 (нормализация вопроса), опыт G4",
        "mt_lexical_gap": "да, гипотеза H3 (перевод вопроса), опыт G3",
        "needs_inference": "частично, рассуждением",
        "entity_alias": "нет без словаря алиасов",
        "retrieval_miss": "поиск",
    }
    for code, n in primary.most_common():
        lines.append(f"| {codes[code]} | {n} | {fixable.get(code, '-')} |")
    lines += [
        "",
        "С учётом вторичных кодов: " + ", ".join(f"{c} {n}" for c, n in any_code.most_common()) + ".",
        "",
        "Вывод: 22 из 52 отказов - обоснованные отказы или ошибки данных (текст не утверждает "
        "того, о чём спрашивают; эталон противоречит тексту; вопрос без контекста статьи). "
        "Реалистичная цель - 12 вопросов с проблемой перевода (7 искажённых вопросов и 5 "
        "лексических расхождений EN-RU) и часть из 9 вопросов со строгим чтением синонима.",
        "",
        "## 3. Вопросы рассуждения после обрезки",
        "",
    ]
    for d in sorted(think, key=lambda d: (d["think_outcome"] != "still_wrong", d["id"])):
        lines += section(d, notes[d["id"]], codes)
    lines += ["## 4. Упорные отказы XQuAD", ""]
    for d in refusals:
        lines += section(d, notes[d["id"]], codes)
    (OUT / "DIAGNOSTIC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{OUT / 'DIAGNOSTIC_REPORT.md'}: {len(think)} + {len(refusals)} questions")


if __name__ == "__main__":
    main()
