# optimization_4: диагностика прогона 20261103T1310Z

Начинать с `FINAL_REPORT.md`.

| Файл | Содержание |
|---|---|
| FINAL_REPORT.md | итог: базовая линия, диагностика, эксперименты, быстрый контур, рекомендации |
| DIAGNOSTIC_REPORT.md | разбор каждого проблемного вопроса: 16 рассуждений после обрезки, 52 отказа XQuAD |
| EXPERIMENT_PLAN.md | гипотезы H1-H5, методика, метрики успеха, регрессионные проверки, критерии принятия |
| EXPERIMENTS.csv, experiments/ | офлайн-эксперименты: повтор B00, E1, E2 (1310Z и 0749Z), поиск E3-E5 |
| diagnostics/ | досье вопросов (jsonl): контекст, все попытки, утверждения с баллами NLI и якорями |
| live/ | результаты запусков `scripts/linux/run_experiments.sh` на RTX 3090: `20261103T1801Z_opt4_quick` (D0, G3, G4 из сохранённого состояния), `20261103T1839Z_opt4_g3` (G3 в настоящем конвейере); в каждом SUMMARY.md, run.log, analysis.json |
| BEST_CONFIG.yaml | рабочая конфигурация после optimization_4: конфигурация optimization_3 + `query_translation: refusal` |

Код: `research/opt4/`, перевод вопроса в конвейере - `query_translation.py`
(`query_translation=refusal`; в настройках по умолчанию выключен, в профиле
`qwen38-27b-hybrid` включён). Финальная оценка на RTX 3090:
`PLAN=main bash scripts/linux/run_benchmark.sh` (см. FINAL_REPORT.md, раздел 7).
