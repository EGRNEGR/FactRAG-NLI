# optimization_5: разбор английских таблиц и заголовков, G2/E1 на новом индексе

Ветка `factrag-opt5`, базовая линия - полный прогон `20261103T1929Z_rtx3090_main`
(optimization_4, 1188 / 1307).

| Файл | Содержание |
|---|---|
| PARSER_REPORT.md | этапы 1-2: причина дефекта разбора, исправление (parser v2), тесты и регрессия, сравнение структуры, поиск на 220 вопросах v3, датасет v2 |
| FULL_RUN_1222Z.md | этап 2: прогон `PARSER=v2 PLAN=main` на RTX 3090 против 1929Z - 1193 / 1307, +11 / -6 (p = 0.33), разбор каждого изменения, E1 офлайн |
| offline/ | офлайн-повтор B00 и E1 на прогоне 1222Z (`research.opt4.offline`) |
| EXPERIMENT_PLAN.md | базовый прогон на новом индексе, G2 и E1 в исходных определениях, критерии принятия, время |
| parser/ | `structure_v1_v2.json` (структура корпуса), `regression_v1_v2.json` (XQuAD и золотой корпус v1 против v2), `binding_v1_v2.json` и `dataset_v2_check.json` (привязка датасета) |
| retrieval/ | `retrieval_v1_v2.json` - поиск по каждому вопросу v3 для v1 и v2; вариант без заголовков контролей |
| smoke_5060/ | сквозная проверка на ноутбуке: индекс v2 через конвейер, 3 вопроса с LLM |
| live/ | запуски на RTX 3090: `20261104T1222Z_rtx3090_main` (этап 2: COMPARISON.md, попарное сравнение с 1929Z, время этапов) |

Код: `document_parser.py` (`version="v2"`), `scripts/download_quality_corpus.py`
(`MarkdownExtractor`), `scripts/build_eval_dataset_v3.py --parser v2`, `research/optimization_5/`.

## Запуск на RTX 3090

```bash
cd ~/Downloads/FactRAG-NLI-3090/FactRAG-NLI-rtx3090-linux
UPDATE_BRANCH=factrag-opt5 bash scripts/linux/update.sh
PARSER=v2 PLAN=main bash scripts/linux/run_benchmark.sh          # этап 2, ~3 ч 15 мин
PLAN=opt5 BASELINE=<RUN_ID этапа 2> bash scripts/linux/run_experiments.sh   # этап 3, ~30-40 мин
```

`PARSER=v2` нужен в первой команде явно: по умолчанию разбор v1, как во всех прежних прогонах.
Результаты:
- этап 2: `reports/runs/<RUN_ID>/`;
- этап 3: `reports/optimization_5/live/<RUN_ID>/`.
