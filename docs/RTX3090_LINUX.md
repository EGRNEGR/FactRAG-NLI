# FactRAG-NLI на RTX 3090 (24 ГБ) под Linux

Поставка ветки `rtx3090-linux`: перенос конвейера с ноутбучной RTX 5060 (8 ГБ, Windows) на
настольную RTX 3090 (Ampere, sm_86, 24 ГБ, драйвер 595, CUDA 13.2) под Linux, замена LLM
Qwen2.5-14B на Qwen3.8-27B, внешний тестовый датасет XQuAD и прогон с отдыхом GPU.

## 1. Что изменилось

| Компонент | Было (RTX 5060 8 ГБ) | Стало (RTX 3090 24 ГБ) |
|---|---|---|
| LLM | Qwen2.5-14B-Instruct Q4_K_M, 25 слоёв на GPU | Qwen3.8-27B UD-Q4_K_XL, все слои на GPU |
| llama.cpp | b10819, Windows | b12002, `ubuntu-cuda-12.8-x64` |
| Окно контекста | 4096 | 8192 |
| Батчи BGE-M3 / реранкера | 2 / 2 | 16 / 8 |
| Реранкер | bge-reranker-large | bge-reranker-large (по умолчанию) или bge-reranker-v2-m3 |
| Запуск | `scripts/start_rag.ps1` | `scripts/linux/*.sh` |

Пороги верификатора и поиска не менялись: они калибровались на dev-половине `eval_dataset_v3`.

Изменения в коде (всё обратно совместимо, умолчания прежние):

- `settings.py`, `generator_verifier.py`: `RAG_LLM_DISABLE_THINKING=true` добавляет к запросу
  `chat_template_kwargs: {"enable_thinking": false}`. Qwen3.x по умолчанию рассуждает в блоке
  `<think>`, который съел бы `max_tokens` и сломал разбор ссылок `[S#]`. Сервер дополнительно
  запускается с `--reasoning off`.
- `evaluate_suite.py`: `FACTRAG_INDEX` и `FACTRAG_PARSE_CACHE` задают индекс и кэш разбора для
  другого датасета (иначе `open_pipeline` перестроил бы индекс v3); `FACTRAG_EMBEDDING_BATCH`
  и `FACTRAG_RERANK_BATCH` поднимают батчи; в `retrieval/cases.jsonl` пишется
  `top_rerank_score` для калибровки гейта.
- `scripts/build_eval_dataset_v3.py`: `parse_all` принимает каталог корпуса и кэша.
- Новые файлы: `scripts/build_xquad_dataset.py`, `scripts/calibrate_gate.py`,
  `scripts/compare_runs.py`, `scripts/linux/`, `.env.rtx3090.example`, тесты
  `tests/test_gpu_duty_cycle.py` и `test_thinking_switch_in_request`.

## 2. Выбор LLM

Кандидаты на октябрь 2026 года, которые помещаются в 24 ГБ вместе с BGE-M3, реранкером и
mDeBERTa (около 3.5 ГБ):

| Модель | Тип | GGUF Q4 | Вывод |
|---|---|---|---|
| **Qwen3.8-27B** (Qwen, 2026-08) | dense 27B, гибрид Gated DeltaNet + attention | 16.4 ГБ (UD-Q4_K_XL) | **выбрана** |
| Qwen3.6-27B (2026-04) | dense 27B | ~16.8 ГБ | предыдущее поколение той же линейки |
| Qwen3.6-35B-A3B | MoE, 3B активных | ~20-22 ГБ | быстрее, но не помещается рядом с кодировщиками |
| Qwen3.8-Flash-Next | MoE | ~200 ГБ | не для одной карты |

Почему Qwen3.8-27B:

- самая новая dense-модель Qwen, помещающаяся целиком на 3090; линейка Qwen остаётся сильнейшей
  открытой по русскому языку и следованию инструкциям (цитаты `[S#]`, отказ при нехватке данных);
- только 16 из 64 слоёв хранят KV-кэш (4 KV-головы по 256), поэтому окно 8192 стоит около
  0.5 ГБ;
- dense, а не MoE: при temperature 0 ответы стабильнее, а скорость (~35 ток/с на 3090) для
  RAG-ответа в 150-300 токенов достаточна.

Бюджет VRAM: рабочий стол занимает ~0.55 ГБ; Qwen3.8 UD-Q4_K_XL 16.4 ГБ, KV и буферы ~1.5 ГБ,
кодировщики ~3.5 ГБ; итого около 22 ГБ. Профиль `qwen38-27b-q5` (UD-Q5_K_M, 18.4 ГБ) только
без графической оболочки: `sudo systemctl isolate multi-user.target`, управление по SSH.

## 3. Внешний датасет: XQuAD

[google/xquad](https://huggingface.co/datasets/google/xquad) (CC BY-SA 4.0): 240 абзацев
SQuAD v1.1 из 48 статей Википедии и 1190 вопросов с профессиональным переводом на русский;
версии RU и EN параллельны по ID вопроса. `scripts/build_xquad_dataset.py` переводит его в схему
v3, поэтому `evaluate_suite.py` работает с ним без изменений:

- корпус: русские абзацы 40 статей, по файлу `.md` на статью (`data/xquad/docs`); 8 статей
  (выбраны с seed 42) в корпус не входят;
- `single_hop` (158): вопросы по статьям корпуса, половина на русском, половина на английском
  (межъязыковой поиск); цитата-доказательство: русское предложение с ответом, привязанное к
  чанкам тем же кодом, что и v3;
- `unanswerable` (40): вопросы по исключённым статьям, ответ на которые не встречается в корпусе
  (проверено через `absence_terms`);
- NLI-пробы: entailment = предложение с ответом (RU, контроль) и его английская параллель
  (межъязыковая), neutral = предложение исключённой статьи против постороннего предложения
  корпуса;
- проверка ответа: XQuAD даёт короткие фрагменты, а русский их склоняет, поэтому
  `answer_must_contain` содержит одно regex-условие на основы слов фрагмента (RU или EN) в
  любом порядке.

2 кейса из 200 отброшены: их общая цитата попала на границу чанков. Исходные parquet лежат в
`data/xquad/raw`, готовый `data/xquad_dataset.json` в репозитории. `--check` пересобирает и
сверяет его побайтно (так `setup.sh` строит кэш разбора на новой машине).

## 4. Замеры поиска (без LLM, RTX 5060, 2026-10-30)

| Датасет | Реранкер, порог гейта | Recall@5 | MRR@10 | Гейт отсекает: невозможные / отвечаемые |
|---|---|---|---|---|
| v3 | bge-reranker-large, 0.05 (production) | 0.820 | 0.761 | 45.8% / 2.3% |
| v3 | bge-reranker-v2-m3, 0.05 | 0.828 | 0.767 | 12.5% / 0.0% |
| v3 | bge-reranker-v2-m3, 0.2082 | 0.799 | 0.756 | 35.4% / 1.7% |
| XQuAD | bge-reranker-large, 0.05 | 0.671 | 0.591 | 62.5% / 16.5% |
| XQuAD | bge-reranker-v2-m3, 0.05 | 0.962 | 0.956 | 97.5% / 3.2% |
| XQuAD | bge-reranker-v2-m3, 0.2082 | 0.905 | 0.899 | 97.5% / 8.9% |

Dense-поиск BGE-M3 и гибрид RRF на XQuAD дают Recall@5 = 1.000: потери создаёт англо-китайский
bge-reranker-large, который на русском тексте поднимает наверх чужие чанки. Многоязычный
bge-reranker-v2-m3 устраняет это, но его шкала баллов другая: при пороге 0.05 гейт пропускает
почти все невозможные вопросы v3. `scripts/calibrate_gate.py` подобрал 0.2082 на dev-половине v3
по той же доле ложных отказов, что у bge-reranker-large; на test-половине v3 гейт отсекает те же
40.9% невозможных вопросов при 2.3% ложных отказов (было 3.4%). Цена: меньший Recall на XQuAD.

Поэтому по умолчанию остаётся калиброванный bge-reranker-large, а `PLAN=full` прогоняет
bge-reranker-v2-m3 (0.2082) третьей конфигурацией: решение примет сквозная оценка.

NLI-пробы XQuAD (mDeBERTa, правило production): neutral-пробы пойманы в 82.5%, ложное отклонение
entailment-проб 9.8%, совпадение статусов 84.8% (n = 396).

## 5. Установка на Linux-ПК

Нужно: драйвер NVIDIA (`nvidia-smi`), Python ≥ 3.11 с `venv`, `curl`, ~50 ГБ на диске.

```bash
sudo apt install -y python3.12 python3.12-venv curl     # Ubuntu 24.04
unzip FactRAG-NLI-3090.zip && cd FactRAG-NLI-3090        # или: git clone -b rtx3090-linux ...
bash scripts/linux/setup.sh
```

`setup.sh` по шагам: проверка GPU и Python; `.venv` с PyTorch cu128; llama.cpp b12002 в
`tools/llama` (проверка, что видит CUDA); модели с Hugging Face (BGE-M3, оба реранкера, mDeBERTa,
Qwen3.8-27B и Qwen2.5-14B для сравнения, около 33 ГБ; GGUF качается `curl -C -`, обрыв
продолжается повторным запуском, затем проверка SHA-256); `.env` из `.env.rtx3090.example` с
новым API-токеном; сборка кэша XQuAD; тесты. Без базовой модели:
`LLM_PROFILES=qwen38-27b bash scripts/linux/setup.sh`.

## 6. Прогон с отдыхом GPU

```bash
bash scripts/linux/run_benchmark.sh                 # PLAN=full
PLAN=quick bash scripts/linux/run_benchmark.sh      # 10 кейсов на датасет, проверка за ~15 мин
bash scripts/linux/run_benchmark.sh --resume        # продолжить после остановки или перезагрузки
bash scripts/linux/stop_benchmark.sh                # остановить прогон и llama-server
```

Прогон отвязан от терминала (`setsid nohup`): закрытие AnyDesk или окна его не прерывает,
Ctrl+C прекращает только просмотр лога. Каждые 45 минут работы `gpu_duty_cycle.py`
замораживает текущую стадию (SIGSTOP) на 5 минут и затем продолжает её (SIGCONT). Отдых
начинается и раньше, если GPU нагрелся до 83 °C. llama-server не замораживается: он дописывает
начатый ответ, и клиент получает его после паузы без таймаута. Параметры: `WORK_MINUTES`,
`REST_MINUTES`, `GPU_TEMP_LIMIT`.

План `full` (оценка 4-6 часов вместе с отдыхом):

1. проверка датасета XQuAD, NLI-пробы XQuAD;
2. поиск v3 и XQuAD для каждого реранкера;
3. для каждой конфигурации (`qwen38-27b+bge-reranker-large`, `qwen25-14b+bge-reranker-large`,
   `qwen38-27b+bge-reranker-v2-m3`): перезапуск llama-server с нужной моделью, e2e v3 (220),
   e2e XQuAD (198), повтор кейсов с инфраструктурными ошибками;
4. остановка сервера и `COMPARISON.md`.

Результаты: `reports/runs/<RUN_ID>/COMPARISON.md`; полные логи кейсов в подкаталогах.
Состояние во время работы: `.artifacts/duty_cycle/status.json` (стадия, работа/отдых,
температура, мощность, VRAM). Лог: `reports/runs/<RUN_ID>/benchmark.log`.

Как читать сравнение: строка `qwen25-14b+bge-reranker-large` показывает старую модель на том же
железе и с тем же кодом, поэтому разница с `qwen38-27b+bge-reranker-large` целиком относится к
LLM. Строка-эталон с ноутбука (20261026T2139Z_v3) получена верификатором v1; replay тех же
черновиков верификатором v2 дал 78.6%.

## 7. Ручной запуск

```bash
bash scripts/linux/llama_server.sh start      # status | stop | restart
source scripts/linux/env.rtx3090.sh
.venv/bin/python -X utf8 demo.py --ensure-index --query "Какой код состояния означает ..."
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

Другая модель: `LLM_PROFILE=qwen25-14b bash scripts/linux/llama_server.sh restart`.
Другой реранкер: `RERANKER=bge-reranker-v2-m3` (порог гейта подставляется сам).

## 8. Перенос на другой ПК

Архив `FactRAG-NLI-3090.zip` содержит код, данные и отчёты (без моделей и `.venv`; их ставит
`setup.sh`). Его можно взять с Google Диска или клонировать ветку `rtx3090-linux` с GitHub. Архив
собран `git archive`, поэтому bash-скрипты в нём с LF.

## 9. Неполадки

| Симптом | Что делать |
|---|---|
| `CUDA out of memory` при старте llama-server | закрыть браузер; `LLAMA_CTX=6144`; без GUI: `sudo systemctl isolate multi-user.target` |
| `llama-server does not see a CUDA device` | `nvidia-smi`; переустановить драйвер; `rm -rf tools/llama` и снова `setup.sh` |
| загрузка модели оборвалась | повторить `setup.sh`: curl продолжит с места обрыва |
| `SHA-256 ... FAILED` | удалить файл из `models/llm` и повторить `setup.sh` |
| много `errors` в e2e | `tail .artifacts/llama/server.log`; затем `run_benchmark.sh --resume` |
| GPU постоянно > 83 °C | проверить обдув; `GPU_TEMP_LIMIT=80 WORK_MINUTES=30` |
