# FactRAG-NLI

[![tests](https://github.com/EGRNEGR/FactRAG-NLI/actions/workflows/tests.yml/badge.svg)](https://github.com/EGRNEGR/FactRAG-NLI/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![Inference](https://img.shields.io/badge/inference-100%25%20local-2ea44f)
![LLM](https://img.shields.io/badge/LLM-Qwen2.5--14B%20Q4__K__M%20%C2%B7%20llama.cpp-8A2BE2)
![Retrieval](https://img.shields.io/badge/retrieval-BGE--M3%20%2B%20BM25%20%2B%20bge--reranker-1f6feb)
![NLI](https://img.shields.io/badge/NLI-mDeBERTa--v3-orange)
![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)

**Локальная RAG-система для технической документации, которая не публикует утверждения LLM без подтверждения источником.**
Ответ генерирует Qwen2.5-14B через llama.cpp, затем каждое утверждение проверяется моделью логического вывода mDeBERTa-v3 по процитированным фрагментам: неподтверждённое удаляется, а при нехватке подтверждений вместо ответа выдаётся отказ; весь конвейер работает на одной GPU 8 ГБ без внешних API.

- [Как это работает](#как-это-работает)
- [Ключевые возможности](#ключевые-возможности)
- [Результаты](#результаты)
- [Быстрый старт](#быстрый-старт)
- [Конфигурация](#конфигурация)
- [Структура репозитория](#структура-репозитория)
- [Ограничения](#ограничения)

## Как это работает

```mermaid
graph TD
    Q["Вопрос"] --> N["Нормализация NFKC,<br/>фильтр prompt-injection"]
    N --> D["Dense: BGE-M3 → Qdrant<br/>top-60"]
    N --> S["Sparse: BM25Okapi<br/>top-60"]
    D --> F["Взвешенный RRF, k = 60"]
    S --> F
    F --> R["bge-reranker-large<br/>окна 384 токена, 0.8·RRF + 0.2·rerank"]
    R --> G1{"rerank ≥ 0.05?"}
    G1 -- "нет" --> X1["Отказ без вызова LLM<br/>relevance_threshold_not_met"]
    G1 -- "да" --> C["Сборка контекста: top-5 источников S1…S5,<br/>бюджет 4096 − 512 − 128 токенов"]
    C --> L["Qwen2.5-14B-Instruct Q4_K_M<br/>llama-server, temperature 0"]
    L --> SP["Разбиение черновика на утверждения<br/>с цитатами [Si]"]
    SP --> V["mDeBERTa-v3 NLI:<br/>утверждение × каждый цитируемый чанк"]
    V --> G2{"P(con) ≥ 0.2 → CONTRADICTION<br/>P(ent) > 0.4 у всех или ≥ 0.7 → VERIFIED<br/>иначе UNVERIFIED"}
    G2 -- "VERIFIED" --> O["Ответ из подтверждённых утверждений<br/>с цитатами [Si]"]
    G2 -- "UNVERIFIED, CONTRADICTION" --> DROP["Утверждение удаляется"]
    DROP --> G3{"0 подтверждённых или<br/>заблокировано > 50 %?"}
    G3 -- "да" --> X2["Отказ<br/>nli_verification_failed"]
    G3 -- "нет" --> O
```

Посылкой для NLI служит полный текст процитированного чанка без обрезки: пара длиннее 512 токенов получает UNVERIFIED, а не проверку по усечённому тексту, чтобы не потерять исключения и отрицания в конце фрагмента.

## Ключевые возможности

- **Полностью локальный контур.** Модели загружаются только из локальных файлов (`local_files_only`, `trust_remote_code=False`), LLM доступна только по loopback; документы и вопросы не покидают машину.
- **Гибридный поиск.** Dense-векторы BGE-M3 находят перефразировки, BM25 точные термины, идентификаторы и фрагменты кода; RRF объединяет кандидатов, cross-encoder переупорядочивает их, порог релевантности отсекает часть вопросов вне базы до вызова LLM.
- **NLI-верификация каждого утверждения.** Детерминированное правило (temperature 0, фиксированные пороги) пропускает к пользователю только утверждения, которые NLI-модель признала следующими из процитированных источников; отказ всегда типизирован (`relevance_threshold_not_met`, `llm_refusal`, `nli_verification_failed`).
- **Защита от ложных предпосылок.** Системный промпт запрещает принимать предпосылку вопроса за факт (premise guard); эффект измерен отдельно от NLI.
- **Документы PDF, DOCX, TXT, Markdown** со структурой разделов и таблиц, инкрементальная загрузка с дедупликацией по SHA-256, саммаризация документа с NLI-фильтром.
- **Интерфейсы:** Streamlit, FastAPI, CLI.
- **Воспроизводимая оценка:** датасет v3 из 220 стратифицированных кейсов, абляции поиска и верификации, сырые логи прогонов и генератор отчёта.

## Результаты

<!-- results:start -->
Сквозной прогон на 220 вопросах датасета v3, прод-профиль. *Standard RAG* - черновик той же LLM до NLI-фильтра (тот же поиск, порог релевантности и промпт), поэтому сравнение парное.

| Метрика | Standard RAG (без NLI) | FactRAG-NLI |
| :--- | ---: | ---: |
| Корректный исход, все вопросы ↑ | 81.4 % (179/220; ДИ 76-86 %) | 56.8 % (125/220; ДИ 50-63 %) |
| Ключевые факты верны (отвечаемые) ↑ | 69.3 % (88/127; ДИ 61-77 %) | 27.6 % (35/127; ДИ 21-36 %) |
| Отказы на отвечаемых вопросах ↓ | 11.8 % (15/127; ДИ 7-19 %) | 47.2 % (60/127; ДИ 39-56 %) |
| Галлюцинации на вопросах с известной истиной ↓ | 2.2 % (2/93; ДИ 1-8 %) | 3.2 % (3/93; ДИ 1-9 %) |

В текущей конфигурации NLI-фильтр **снизил** долю корректных исходов (только Standard RAG верен в 55 кейсах, только FactRAG-NLI в 1; точный тест Мак-Немара p = 1.6e-15) и не уменьшил число галлюцинаций (2 → 3 из 93). Из 274 утверждений черновиков VERIFIED получили 118; 52 отклонены без вызова NLI, потому что после разбиения на предложения остались без маркера источника; 68 получили CONTRADICTION (P(con) ≥ 0.2); 36 не подтверждены NLI. Разбор причин: [раздел 4 отчёта](reports/EVALUATION_REPORT.md#4-разбор-ошибок). NLI-верификация добавляет p50 18 мс к запросу; основное время занимает генерация LLM (p50 19 199 мс).

NLI-модуль на 440 контрольных парах EN/RU: перехват 88.2 %, ложная отбраковка 25.2 %, macro-F1 0.716.

Поиск, Recall@5: dense 0.767, BM25 0.645, RRF 0.695, RRF + реранкинг 0.799.
<!-- results:end -->

Полный отчёт с методикой, абляциями, разбором ошибок и таймингами: [`reports/EVALUATION_REPORT.md`](reports/EVALUATION_REPORT.md); сводные таблицы: [`reports/tables/`](reports/tables/); сырые логи прогона: [`reports/runs/`](reports/runs/).

Стенд: RTX 5060 Laptop 8 ГБ, Qwen2.5-14B-Instruct Q4_K_M (llama.cpp, `-ngl 25`), корпус из 9 открытых документов (RFC 9110, RFC 9000, RFC 9457, RFC 9562, NIST SP 800-63B-4, NIST SP 800-53 Rev. 5, PostgreSQL 18, PEP 703), вопросы на русском и английском. Разметка датасета подготовлена ИИ и проверена механически (дословные цитаты, отсутствие терминов в корпусе), экспертной проверки не было.

## Быстрый старт

Проверено на Windows 11, Python 3.12, NVIDIA RTX 5060 Laptop (sm_120).

### 1. Окружение

```powershell
git clone https://github.com/EGRNEGR/FactRAG-NLI.git
cd FactRAG-NLI
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install "transformers>=4.40" httpx==0.28.1
```

PyTorch с CUDA устанавливается отдельно; для RTX 50xx нужна сборка с поддержкой sm_120 (проверено: `2.12.0.dev20260408+cu128`, см. [`docs/evidence/release_runtime.txt`](docs/evidence/release_runtime.txt)). Проверка реального вычисления на GPU:

```powershell
python -c "import torch; x=torch.randn(64,64,device='cuda'); print(torch.__version__, (x@x).shape)"
```

### 2. Веса моделей

| Каталог | Источник |
|---|---|
| `models/bge-m3` | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| `models/bge-reranker-large` | [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) |
| `models/mdeberta-nli` | [MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) |
| `models/llm/Qwen2.5-14B-Instruct-Q4_K_M.gguf` | [Qwen/Qwen2.5-14B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF) |

`llama-server.exe` и его DLL из [релиза llama.cpp](https://github.com/ggml-org/llama.cpp/releases) помещаются в `tools/llama/`. Веса и бинарники в репозиторий не входят; порядок доставки и сверки хешей описан в [`docs/RELEASE_GUIDE.md`](docs/RELEASE_GUIDE.md). Аудит окружения: `python scripts/verify_environment.py`.

### 3. Конфигурация

```powershell
Copy-Item .env.example .env
python -c "import json, secrets; print(json.dumps([secrets.token_urlsafe(48)]))"   # значение для RAG_API_TOKENS
```

`.env.example` содержит профиль, на котором получены опубликованные метрики.

### 4. LLM-сервер

```powershell
.\tools\llama\llama-server.exe -m .\models\llm\Qwen2.5-14B-Instruct-Q4_K_M.gguf -ngl 25 -c 4096 -b 128 -ub 64 -np 1 --cache-ram 0 --port 8080 --host 127.0.0.1
```

`-ngl 25` подобран замером на RTX 5060 Laptop 8 ГБ: рядом с llama-server на GPU работают BGE-M3, реранкер и NLI (около 2.7 ГБ в процессе PyTorch). Прежний профиль `-ngl 35` с llama.cpp сборки 10819 в 8 ГБ вместе с ними не помещается: драйвер Windows вытесняет часть буферов llama-server в общую память, и со второго запроса обработка промпта замедляется примерно в 4 раза; при 25 слоях скорость стабильна. Таблица замеров: [отчёт, раздел 5](reports/EVALUATION_REPORT.md#5-задержки-и-ресурсы). `.\scripts\start_rag.ps1` запускает сервер с тем же профилем; прежний задаётся параметром `-GpuLayers 35`.

`--cache-ram 0` отключает кэш промптов в ОЗУ (в сборке 10819 он по умолчанию до 8 ГиБ): при сотнях разных вопросов он не даёт повторного использования, а рост памяти выталкивает CPU-слои модели из ОЗУ. Флаг поддерживается только новыми сборками llama.cpp.

### 5. Индексация и запуск

```powershell
python baseline_rag.py --index ./data/documents            # индексировать каталог документов
python scripts/manage_index.py --add "C:\Docs\manual.pdf"  # добавить файл инкрементально
.\scripts\start_rag.ps1                                    # llama-server + Streamlit: http://127.0.0.1:8501
uvicorn api:app --host 127.0.0.1 --port 8000               # или HTTP API: /health, /ingest, /query
python demo.py --query "Что означает код HTTP 204 согласно RFC 9110?" --ensure-index
.\scripts\stop_rag.ps1 -Force                              # остановить llama-server
```

Встроенный Qdrant допускает одного владельца индекса: Web, API, CLI и оценку над одним индексом одновременно не запускают. Подробное руководство по эксплуатации: [`docs/OPERATIONS.md`](docs/OPERATIONS.md).

### 6. Тесты и оценка

```powershell
python -m pytest                       # модульные тесты без весов моделей и GPU
python -m ruff check .
python -m mypy --strict .

python scripts/build_eval_dataset_v3.py --check                    # проверить датасет v3
python evaluate_suite.py --run-dir reports/runs/my-run retrieval   # абляция поиска (без LLM)
python evaluate_suite.py --run-dir reports/runs/my-run nli         # NLI на контрольных парах
python evaluate_suite.py --run-dir reports/runs/my-run e2e --profile guard_on
python evaluate_suite.py --run-dir reports/runs/my-run e2e --profile guard_off
python scripts/render_report.py reports/runs/my-run                # отчёт, таблицы, графики
```

Прогон фиксирует seed, параметры сэмплирования, пороги NLI, хеши датасета, кода и весов, командную строку llama-server и пишет сырые логи по каждому кейсу; прерванный режим продолжается с последнего кейса.

## Конфигурация

Все параметры задаются переменными `RAG_*` в `.env` (полный список: [`settings.py`](settings.py)).

| Параметр | Значение | Назначение |
|---|---|---|
| `RAG_MODE`, `RAG_ALLOW_FALLBACK` | `production`, `false` | Без эвристических заглушек моделей |
| `RAG_LLM_API_BASE` | `http://127.0.0.1:8080/v1` | OpenAI-совместимый endpoint llama-server, только loopback |
| `RAG_CHUNK_TARGET_TOKENS`, `RAG_CHUNK_MAX_TOKENS` | `320`, `384` | Нарезка документов (токенизатор BGE-M3) |
| `RAG_DENSE_CANDIDATES`, `RAG_SPARSE_CANDIDATES` | `60`, `60` | Кандидаты dense и BM25 перед RRF |
| `RAG_RERANK_RRF_ALPHA` | `0.8` | Доля нормированного RRF в итоговой оценке |
| `RAG_RERANK_SCORE_THRESHOLD` | `0.05` | Порог релевантности: ниже него отказ без генерации |
| `RAG_RERANK_TOP_K` | `5` | Источников в контексте LLM |
| `RAG_LLM_CONTEXT_WINDOW`, `RAG_MAX_TOKENS` | `4096`, `512` | Бюджет контекста и ответа |
| `RAG_NLI_ENTAILMENT_THRESHOLD` | `0.4` | Минимальная P(entailment) от каждого цитируемого источника |
| `RAG_NLI_CONTRADICTION_THRESHOLD` | `0.2` | P(contradiction) от любого источника, блокирующая утверждение |
| `RAG_NLI_STRONG_ENTAILMENT_THRESHOLD` | `0.7` | P(entailment) одного источника, достаточная для подтверждения |
| `RAG_GENERATION_LANGUAGE` | `ru` | Язык ответа |
| `RAG_EMBEDDING_BATCH_SIZE`, `RAG_RERANK_BATCH_SIZE` | `2`, `2` | Батчи, при которых процесс PyTorch укладывается в ~3 ГБ рядом с llama-server |

## Структура репозитория

```text
FactRAG-NLI/
├── settings.py              # типизированная конфигурация RAG_*, проверки профиля
├── document_parser.py       # PDF/DOCX/TXT/MD → структурные чанки
├── document_processor.py    # инкрементальная загрузка, SHA-256-дедупликация
├── hybrid_retriever.py      # BGE-M3 + Qdrant, BM25, RRF, cross-encoder, порог релевантности
├── generator_verifier.py    # промпт, llama-server, разбиение на утверждения, NLI-правило
├── pipeline.py              # оркестрация QA-маршрута и типизированный ответ
├── summarization.py         # саммаризация документа с NLI-фильтром
├── app.py, web_bridge.py    # Streamlit-интерфейс
├── api.py                   # FastAPI
├── baseline_rag.py, demo.py # CLI
├── eval_dataset.py          # схема датасета v3
├── evaluate_suite.py        # оценка v3: поиск, NLI, сквозной A/B
├── evaluate_quality.py      # оценка на golden/holdout (v1)
├── benchmark.py             # GPU/latency-бенчмарк
├── data/
│   ├── eval_dataset_v3.json # 220 кейсов с gold_context и NLI-пробами
│   ├── eval_v3/drafts/      # черновики кейсов по документам
│   ├── test_docs/           # корпус оценки v3 (+ derived/ из HTML)
│   ├── documents/, raw/     # корпус golden/holdout
│   └── *_dataset*.json      # наборы v1 и v2
├── reports/
│   ├── EVALUATION_REPORT.md # отчёт об оценке (генерируется)
│   ├── summary.json         # ключевые числа прогона
│   ├── tables/, figures/    # CSV и SVG
│   ├── runs/<run_id>/       # конфигурация и сырые логи прогона
│   └── legacy/              # прежние отчёты
├── scripts/                 # сборка датасета, рендер отчёта, запуск, индекс, релиз
├── tests/                   # модульные тесты (без весов моделей)
├── docs/                    # руководство по эксплуатации, протоколы этапов, доклад
└── .github/workflows/       # CI: pytest, ruff, mypy --strict на Windows
```

## Ограничения

- В оценённой конфигурации NLI-фильтр снижает долю корректных исходов по сравнению с черновиком той же LLM и не уменьшает число галлюцинаций (см. [Результаты](#результаты)). Измеренные причины: утверждения без маркера источника после разбиения на предложения (в том числе разрез «SP 800-53 Rev. 5» после «Rev.») и ложные противоречия mDeBERTa (P(con) ≥ 0.2) на верных утверждениях; разбор в [отчёте](reports/EVALUATION_REPORT.md#4-разбор-ошибок).
- NLI-модель ошибается в обе стороны: блокирует часть верных утверждений и пропускает часть искажённых. Вердикт VERIFIED не гарантирует фактической точности.
- Ответы формируются преимущественно на русском (`RAG_GENERATION_LANGUAGE=ru`), а источники в корпусе оценки английские; межъязыковые пары измерены отдельно.
- Сканированные PDF без текстового слоя (OCR) и формат `.doc` не поддерживаются.
- Интерфейс рассчитан на доверенную локальную машину: многопользовательской авторизации в Web нет, `api.py` не проверяет токены.
- Конфигурация проверена на одном стенде (RTX 5060 Laptop 8 ГБ, Windows); переносимость на другие GPU и ОС не измерялась.

## Документация

- [`reports/EVALUATION_REPORT.md`](reports/EVALUATION_REPORT.md): оценка на датасете v3
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md): эксплуатация (Web, CLI, загрузка, саммаризация, холодный старт)
- [`docs/RELEASE_GUIDE.md`](docs/RELEASE_GUIDE.md): развёртывание и доставка весов
- [`docs/CONFERENCE_REPORT.md`](docs/CONFERENCE_REPORT.md): научно-технический доклад
- [`docs/evidence/`](docs/evidence/): сохранённые результаты прежних этапов

## Лицензия

Код распространяется по лицензии [BSD 3-Clause](LICENSE). Веса моделей и документы корпуса распространяются по лицензиям их правообладателей.
