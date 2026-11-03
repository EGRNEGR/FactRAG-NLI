# optimization_4: диагностика прогона 20261103T1310Z

Начинать с `FINAL_REPORT.md`.

| Файл | Содержание |
|---|---|
| FINAL_REPORT.md | итог: базовая линия, диагностика, эксперименты, быстрый контур, рекомендации |
| DIAGNOSTIC_REPORT.md | разбор каждого проблемного вопроса: 16 рассуждений после обрезки, 52 отказа XQuAD |
| EXPERIMENT_PLAN.md | гипотезы H1-H5, методика, метрики успеха, регрессионные проверки, критерии принятия |
| EXPERIMENTS.csv, experiments/ | офлайн-эксперименты: повтор B00, E1, E2 (1310Z и 0749Z), поиск E3-E5 |
| diagnostics/ | досье вопросов (jsonl): контекст, все попытки, утверждения с баллами NLI и якорями |
| live/ | результаты запусков `scripts/linux/run_experiments.sh` на RTX 3090: `20261103T1801Z_opt4_quick` (D0, G3, G4; SUMMARY.md, run.log, analysis.json, steps/) |

Код: `research/opt4/`, перевод вопроса в конвейере - `query_translation.py`
(`query_translation=refusal`, по умолчанию выключен). Запуск на RTX 3090:
`PLAN=g3 bash scripts/linux/run_experiments.sh` (см. FINAL_REPORT.md, раздел 7).
