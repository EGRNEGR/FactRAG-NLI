# Оценка FactRAG-NLI на датасете v3

Прогон [`reports/runs/20261026T1410Z_v3`](runs/20261026T1410Z_v3), дата 2026-10-26, коммит `d32a40bb8de8`. Все числа сгенерированы `scripts/render_report.py` из сырых логов прогона.

## 1. Краткие выводы

- **Сквозной результат.** На 220 вопросах доля корректных исходов у FactRAG-NLI снизилась с 81.4 % (Vanilla RAG: черновик той же LLM до NLI-фильтра) до 56.8 %: фильтр сделал неверными 55 исходов и исправил 1 (точный тест Мак-Немара, p = 1.6e-15). Из потерянных: ответ заменён отказом по итогам NLI-проверки - 33; ответ опубликован, но фильтр удалил утверждение с ключевым фактом - 22. Отказы на отвечаемых вопросах: 11.8 % → 47.2 %.
- **Галлюцинации.** На вопросах с известной истиной (без ответа, с ложной предпосылкой, lexical_trap без ответа) ответов без опоры на корпус: guard вкл.: Vanilla 2 из 93, FactRAG-NLI 3 из 93; guard выкл.: Vanilla 5 из 93, FactRAG-NLI 4 из 93. Доля мала уже у черновика: из 93 таких вопросов в прод-профиле 21 отсёк порог релевантности (LLM не вызывается), на 32 LLM ответила отказом по инструкции промпта. В 2 из 3 галлюцинаций FactRAG-NLI в прод-профиле черновик был верным: фильтр удалил фразу об отсутствии сведений или исправление предпосылки и опубликовал оставшееся подтверждённое утверждение.
- **Причины на уровне утверждений.** Из 274 утверждений черновиков 52 (19.0 %) остались без маркера источника после разбиения на предложения и отклонены без вызова NLI; 68 (24.8 %) получили CONTRADICTION. На 440 контрольных парах с эталонной посылкой правило перехватывает 88.2 % искажённых и неподтверждённых утверждений, но блокирует 25.2 % верных; 54 из 254 верных гипотез получают P(con) ≥ 0.2. Ложная отбраковка по языку гипотезы: EN 25.2 %, RU 25.2 %.
- **Поиск и задержка.** Гибрид с реранкингом: Recall@5 0.799, MRR@10 0.723 против 0.767 / 0.644 у dense и 0.645 / 0.535 у BM25; RRF без реранкинга (0.695) ниже dense, то есть выигрыш гибрида даёт реранкер. NLI-верификация занимает p50 18 мс на запрос при полном времени p50 19 972 мс, из которых генерация LLM 19 199 мс.

## 2. Методология и датасет

### 2.1. Датасет v3

`data/eval_dataset_v3.json`: 220 кейсов (в 4 раза больше набора v2 из 55 кейсов), 242 единицы эталонного контекста, 68 кейсов с доказательствами в двух и более чанках.

| Категория | Кейсов | RU / EN | Ожидаемое поведение |
| :--- | ---: | ---: | :--- |
| single_hop | 30 | 14 / 16 | answer: 30 |
| multi_hop | 40 | 20 / 20 | answer: 40 |
| tabular_code | 35 | 17 / 18 | answer: 35 |
| adversarial | 45 | 23 / 22 | refusal_or_correction: 45 |
| unanswerable | 40 | 22 / 18 | refusal: 40 |
| lexical_trap | 30 | 13 / 17 | answer: 22, refusal: 8 |
| **Итого** | 220 | 109 / 111 |  |

Каждый кейс содержит эталонный ответ, эталонные чанки (`gold_context`: дословные цитаты, привязанные к `chunk_id` боевого парсера с токенизатором BGE-M3 и нарезкой 320/384/50), ожидаемое поведение и NLI-пробу: посылку (чанк) и гипотезу на английском и русском с ожидаемым вердиктом Entailment / Contradiction / Neutral. Скрипт `scripts/build_eval_dataset_v3.py` проверяет, что каждая цитата и посылка найдены дословно, что для multi_hop доказательства лежат в разных чанках, что термины из вопросов без ответа не встречаются нигде в корпусе, и отсекает почти-дубликаты вопросов из прежних наборов.

Разметку подготовил ИИ (`review_status = ai_draft`); экспертная проверка смысла не проводилась.

| Документ | Чанков | Конвертирован из | Кейсов |
| :--- | ---: | :--- | ---: |
| nist_sp800_53r5.pdf | 1896 |  | 23 |
| pep703_gil.txt | 172 |  | 21 |
| rfc9000_quic.txt | 910 |  | 32 |
| rfc9110_http_semantics.txt | 1306 |  | 41 |
| rfc9457_problem_details.txt | 65 |  | 16 |
| rfc9562_uuid.txt | 347 |  | 27 |
| derived/nist_sp800_63b.md | 520 | nist_sp800_63b.html | 33 |
| derived/postgresql_create_table.md | 118 | postgresql_create_table.html | 24 |
| derived/postgresql_indexes.md | 24 | postgresql_indexes.html | 3 |

### 2.2. Стенд и воспроизводимость

| Параметр | Значение |
| :--- | :--- |
| Коммит кода | `d32a40bb8de8` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB, 596.21 |
| Python / платформа | 3.12.9 / Windows-11-10.0.26300-SP0 |
| Пакеты | torch 2.12.0.dev20260408+cu128, transformers 5.16.1, sentence-transformers 6.0.1, qdrant-client 1.19.0, rank-bm25 0.2.2 |
| LLM | `Qwen2.5-14B-Instruct-Q4_K_M.gguf` (8 572 МиБ), `llama-server.exe -m Qwen2.5-14B-Instruct-Q4_K_M.gguf -ngl 25 -c 4096 -b 128 -ub 64 -np 1 --cache-ram 0 --port 8080 --host 127.0.0.1` |
| Эмбеддинги / реранкер / NLI | BGE-M3 / bge-reranker-large / mDeBERTa-v3 (XNLI, `models/mdeberta-nli`) |
| Сэмплирование | temperature 0.0, top_p 0.9, seed 42, max_tokens 512, контекст 4096 |
| Пороги NLI | entailment > 0.4, contradiction ≥ 0.2, strong entailment ≥ 0.7 |
| Датасет | `data/eval_dataset_v3.json`, SHA-256 `33cb533ee7cc46ad…` |
| Сырые логи | `reports/runs/20261026T1410Z_v3/` (config.json, cases.jsonl, summary.json по режимам) |

### 2.3. Режимы и метрики

- **Поиск** (без LLM): для каждого вопроса одним вызовом получены пять ранжирований: dense (BGE-M3, Qdrant), sparse (BM25), гибрид RRF, RRF с реранкингом до порога и боевой список после порога релевантности 0.05. Recall@K считается по единицам эталонного контекста (единица найдена, если в топ-K есть любой её чанк), MRR@10 и nDCG@K с бинарной релевантностью.
- **NLI** (без LLM): боевая функция `NLIFactVerifier._claim` на эталонной посылке и гипотезе. Отдельно оценены argmax трёх классов (матрица ошибок, P/R/F1) и решение боевого правила: *Hallucination Catch Rate* (доля проб Contradiction/Neutral, не получивших VERIFIED) и *False Rejection Rate* (доля проб Entailment, не получивших VERIFIED).
- **Сквозной прогон**: полный конвейер на 220 вопросах. *Vanilla RAG* = черновик LLM до NLI-фильтра (`raw_answer`), *FactRAG-NLI* = опубликованный ответ; оба из одного вызова LLM, поэтому сравнение парное. Отказ = канонический текст отказа; воздержание = отказ или явная фраза об отсутствии сведений. Корректность: для отвечаемых вопросов все ключевые факты (`answer_must_contain`) присутствуют; для вопросов без ответа воздержание; для adversarial воздержание или верный факт-исправление. Галлюцинация определена только там, где истина зафиксирована датасетом: ответ на вопрос без ответа или adversarial-ответ без верного факта.

## 3. Количественные результаты

### 3.1. Поиск: абляция режимов

| Режим | Recall@1 | Recall@3 | Recall@5 | Recall@10 | nDCG@5 | MRR@10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense (BGE-M3) | 0.436 | 0.683 | 0.767 | 0.831 | 0.621 | 0.644 |
| Sparse (BM25) | 0.349 | 0.555 | 0.645 | 0.733 | 0.513 | 0.535 |
| Hybrid RRF | 0.416 | 0.642 | 0.695 | 0.808 | 0.578 | 0.618 |
| RRF + rerank (без порога) | 0.526 | 0.692 | 0.791 | 0.852 | 0.672 | 0.710 |
| RRF + rerank (прод) | 0.529 | 0.730 | 0.799 | 0.834 | 0.687 | 0.723 |

![Recall@K по режимам поиска](figures/retrieval_recall.svg)

Таблица: [`tables/retrieval_overall.csv`](tables/retrieval_overall.csv). Recall@5 по категориям и языку вопроса:

| Срез | Dense (BGE-M3) | Sparse (BM25) | Hybrid RRF | RRF + rerank (без порога) | RRF + rerank (прод) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| adversarial | 0.89 | 0.68 | 0.76 | 0.81 | 0.88 |
| lexical_trap | 0.73 | 0.68 | 0.77 | 0.82 | 0.68 |
| multi_hop | 0.72 | 0.62 | 0.68 | 0.75 | 0.78 |
| single_hop | 0.83 | 0.73 | 0.70 | 0.93 | 0.97 |
| tabular_code | 0.63 | 0.53 | 0.59 | 0.67 | 0.66 |
| EN | 0.81 | 0.82 | 0.80 | 0.84 | 0.81 |
| RU | 0.72 | 0.45 | 0.58 | 0.73 | 0.79 |

**Порог релевантности (фильтр 1).** Отсечено 41.7 % (20/48; ДИ 29-56 %) вопросов без ответа и 2.3 % (4/172; ДИ 1-6 %) отвечаемых вопросов (по ним LLM не вызывается). Задержка поиска p50 61 мс, реранкинга p50 738 мс.

### 3.2. NLI-верификация на контрольных парах

440 пар (220 проб × EN/RU), ни одна не превысила контекст 512 токенов.

**Классификатор (argmax трёх классов), P / R / F1:**

| Срез | Entailment | Neutral | Contradiction | Macro-F1 | Accuracy |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Все | 0.88 / 0.81 / 0.85 | 0.71 / 0.62 / 0.66 | 0.56 / 0.74 / 0.64 | 0.716 | 0.759 |
| EN-гипотеза | 0.91 / 0.80 / 0.85 | 0.76 / 0.71 / 0.73 | 0.56 / 0.78 / 0.65 | 0.744 | 0.777 |
| RU-гипотеза | 0.85 / 0.83 / 0.84 | 0.65 / 0.54 / 0.59 | 0.56 / 0.71 / 0.63 | 0.686 | 0.741 |

![Матрица ошибок, EN](figures/nli_confusion_en.svg) ![Матрица ошибок, RU](figures/nli_confusion_ru.svg)

**Боевое правило (P(con) ≥ 0.2 в любом источнике → CONTRADICTION; иначе VERIFIED при P(ent) > 0.4 во всех источниках или ≥ 0.7 в одном):**

| Срез | Пар | Catch rate | Contradiction | Neutral | False rejection |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Все | 440 | 88.2 % | 90.0 % | 86.5 % | 25.2 % |
| EN-гипотеза | 220 | 90.3 % | 88.9 % | 91.7 % | 25.2 % |
| RU-гипотеза | 220 | 86.0 % | 91.1 % | 81.2 % | 25.2 % |
| adversarial | 90 | 90.0 % | 90.0 % | н/д | н/д |
| lexical_trap | 60 | 93.8 % | н/д | 93.8 % | 25.0 % |
| multi_hop | 80 | н/д | н/д | н/д | 13.8 % |
| single_hop | 60 | н/д | н/д | н/д | 26.7 % |
| tabular_code | 70 | н/д | н/д | н/д | 37.1 % |
| unanswerable | 80 | 85.0 % | н/д | 85.0 % | н/д |

Распределение вердиктов правила (строка = эталон):

| Эталон | VERIFIED | UNVERIFIED | CONTRADICTION |
| :--- | ---: | ---: | ---: |
| Entailment | 190 | 10 | 54 |
| Neutral | 13 | 54 | 29 |
| Contradiction | 9 | 9 | 72 |

**Чувствительность к порогу противоречия** (порог entailment 0.4, strong 0.7):

| Порог contradiction | Catch rate | False rejection |
| :--- | ---: | ---: |
| 0.1 | 90.3 % | 29.9 % |
| 0.2 (прод) | 88.2 % | 25.2 % |
| 0.3 | 86.6 % | 22.4 % |
| 0.5 | 84.9 % | 18.5 % |
| 0.7 | 84.4 % | 18.1 % |

Полная сетка: [`tables/nli_threshold_sweep.csv`](tables/nli_threshold_sweep.csv). Время одной пары: p50 14.9 мс, p95 23.3 мс.

### 3.3. Сквозной прогон: Vanilla RAG и FactRAG-NLI

| Вариант | Галлюцинации | Отказы на отвечаемых | Ключевые факты верны | Корректный исход |
| :--- | ---: | ---: | ---: | ---: |
| Vanilla RAG, premise guard вкл. (прод) | 2.2 % (2/93; ДИ 1-8 %) | 11.8 % (15/127; ДИ 7-19 %) | 69.3 % (88/127; ДИ 61-77 %) | 81.4 % (179/220; ДИ 76-86 %) |
| FactRAG-NLI, premise guard вкл. (прод) | 3.2 % (3/93; ДИ 1-9 %) | 47.2 % (60/127; ДИ 39-56 %) | 27.6 % (35/127; ДИ 21-36 %) | 56.8 % (125/220; ДИ 50-63 %) |
| Vanilla RAG, premise guard выкл. | 5.4 % (5/93; ДИ 2-12 %) | 10.2 % (13/127; ДИ 6-17 %) | 71.7 % (91/127; ДИ 63-79 %) | 81.4 % (179/220; ДИ 76-86 %) |
| FactRAG-NLI, premise guard выкл. | 4.3 % (4/93; ДИ 2-11 %) | 44.9 % (57/127; ДИ 37-54 %) | 29.9 % (38/127; ДИ 23-38 %) | 57.7 % (127/220; ДИ 51-64 %) |

![Корректный исход, галлюцинации и отказы](figures/e2e_hallucination.svg)

- premise guard вкл. (прод): только Vanilla корректна в 55 кейсах (ответ заменён отказом по итогам NLI-проверки: 33, ответ опубликован, но фильтр удалил утверждение с ключевым фактом: 22), только FactRAG-NLI в 1; точный тест Мак-Немара p = 1.6e-15. Галлюцинации: Vanilla 2 из 93, FactRAG-NLI 3 из 93; только у Vanilla: V3-ADV-021, только у FactRAG-NLI: V3-ADV-006, V3-UNA-025, у обоих: V3-UNA-018.
- premise guard выкл.: только Vanilla корректна в 54 кейсах (ответ заменён отказом по итогам NLI-проверки: 34, ответ опубликован, но фильтр удалил утверждение с ключевым фактом: 20), только FactRAG-NLI в 2; точный тест Мак-Немара p = 4.4e-14. Галлюцинации: Vanilla 5 из 93, FactRAG-NLI 4 из 93; только у Vanilla: V3-ADV-007, V3-ADV-031, только у FactRAG-NLI: V3-UNA-025, у обоих: V3-ADV-021, V3-UNA-018, V3-ADV-008.
- Premise guard (вкл. против выкл.), Vanilla: лучше с guard в 4 кейсах, без guard в 4; точный тест Мак-Немара p = 1.000.
- Premise guard (вкл. против выкл.), FactRAG-NLI: лучше с guard в 7 кейсах, без guard в 9; точный тест Мак-Немара p = 0.804.

По категориям (premise guard вкл. (прод)), Vanilla → FactRAG-NLI:

| Категория | Корректный исход | Галлюцинации | Отказы |
| :--- | ---: | ---: | ---: |
| single_hop | 93 % → 50 % | не опр. | 0 % → 43 % |
| multi_hop | 65 % → 15 % | не опр. | 5 % → 42 % |
| tabular_code | 69 % → 20 % | не опр. | 14 % → 49 % |
| adversarial | 98 % → 98 % | 2 % → 2 % | 22 % → 58 % |
| unanswerable | 98 % → 95 % | 2 % → 5 % | 92 % → 95 % |
| lexical_trap | 60 % → 50 % | 0 % → 0 % | 53 % → 70 % |

Вердикты по утверждениям черновиков (доля от всех утверждений профиля):

| Статус и причина | premise guard вкл. (прод) | premise guard выкл. |
| :--- | ---: | ---: |
| VERIFIED | 118 (43.1 %) | 127 (45.5 %) |
| UNVERIFIED: нет маркера [S#] после разбиения на предложения, NLI не вызывался | 52 (19.0 %) | 47 (16.8 %) |
| UNVERIFIED: NLI не подтвердил (P(ent) ниже порога, P(con) < 0.2) | 36 (13.1 %) | 32 (11.5 %) |
| CONTRADICTION: P(con) ≥ 0.2 хотя бы в одном источнике | 68 (24.8 %) | 73 (26.2 %) |

Faithfulness micro (доля VERIFIED, premise guard вкл. (прод)): 43.1 %; пар «утверждение × источник», не поместившихся в контекст NLI 512 токенов: 0. Причины отказов FactRAG-NLI: nli_verification_failed: 64, llm_refusal: 44, relevance_threshold_not_met: 24. Эталонный чанк среди источников ответа: 85.5 % (147/172; ДИ 79-90 %). Таблица: [`tables/e2e_claim_reasons.csv`](tables/e2e_claim_reasons.csv).

## 4. Разбор ошибок

### 4.1. NLI: где mDeBERTa-v3 ошибается

**Ложные противоречия** (54 из 254 проб Entailment): верное утверждение получает P(contradiction) ≥ 0.2 и блокируется. По категориям: tabular_code: 21, single_hop: 12, lexical_trap: 11, multi_hop: 10; по языку гипотезы: EN: 26, RU: 28.

| Пара | Гипотеза | P(ent) | P(con) | Посылка, симв. |
| :--- | :--- | ---: | ---: | ---: |
| V3-LEX-008:en | Receiving a valid Stateless Reset makes the peer end the connection immediately. | 0.001 | 0.998 | 1280 |
| V3-SH-008:en | UDP datagrams carrying Initial packets must be expanded to at least 1200 bytes. | 0.000 | 0.993 | 1209 |
| V3-SH-008:ru | UDP-датаграммы с пакетами Initial должны быть дополнены как минимум до 1200 байт. | 0.000 | 0.993 | 1209 |
| V3-TC-012:ru | Если max_ack_delay не передан, принимается значение 25 миллисекунд. | 0.011 | 0.975 | 1129 |
| V3-LEX-025:ru | Смещённый подсчёт ссылок опирается на то, что большинство объектов используются только одним потоком. | 0.021 | 0.974 | 1298 |
| V3-TC-012:en | When max_ack_delay is absent, 25 milliseconds is assumed. | 0.006 | 0.974 | 1129 |
| V3-LEX-016:en | The shared and group account credential change enhancement was incorporated into AC-2k. | 0.019 | 0.970 | 1291 |
| V3-LEX-021:ru | Ключевое слово GLOBAL перед TEMPORARY в PostgreSQL ни на что не влияет. | 0.030 | 0.966 | 1149 |

Ложная отбраковка верных утверждений в зависимости от длины посылки (символов в чанке):

| Длина посылки | Пар Entailment | Ложные противоречия | Не VERIFIED |
| :--- | ---: | ---: | ---: |
| 0-800 | 48 | 14.6 % | 14.6 % |
| 800-1400 | 174 | 24.7 % | 30.5 % |
| 1400-∞ | 32 | 12.5 % | 12.5 % |

**Пропущенные противоречия** (9): искажённое утверждение получает VERIFIED. Типы искажений: premise_distortion: 2, version_substitution: 2, modality_inversion: 2, condition_inversion: 2, config_param_substitution: 1.

| Пара | Тип искажения | Гипотеза | P(ent) |
| :--- | :--- | :--- | ---: |
| V3-ADV-019:en | premise_distortion | Syncable authenticators are permitted at AAL3. | 0.989 |
| V3-ADV-019:ru | premise_distortion | Синхронизируемые аутентификаторы разрешены на уровне AAL3. | 0.989 |
| V3-ADV-022:en | config_param_substitution | Session cookies must be readable by JavaScript. | 0.875 |
| V3-ADV-015:en | version_substitution | This version of QUIC is identified by the number 0x00000000. | 0.935 |
| V3-ADV-015:ru | version_substitution | Эта версия QUIC обозначается номером 0x00000000. | 0.962 |
| V3-ADV-043:en | modality_inversion | Clients must reject problem details that contain unrecognized extension members. | 0.984 |
| V3-ADV-043:ru | modality_inversion | Клиенты обязаны отклонять сведения о проблеме с нераспознанными членами-расширениями. | 0.987 |
| V3-ADV-030:en | condition_inversion | Systems without legacy UUIDv1 should prefer UUIDv6 over UUIDv7. | 0.986 |
| V3-ADV-030:ru | condition_inversion | Системам без унаследованных UUIDv1 следует предпочитать UUIDv6, а не UUIDv7. | 0.988 |

**Neutral, принятые за Entailment** (13): утверждение, которого нет в корпусе, проходит проверку против тематически близкой посылки. По категориям: unanswerable: 12, lexical_trap: 1.

| Пара | Гипотеза | P(ent) |
| :--- | :--- | ---: |
| V3-UNA-038:ru | Массивы errors GraphQL необходимо преобразовывать в объекты problem details. | 0.984 |
| V3-UNA-038:en | GraphQL errors arrays must be converted into problem details objects. | 0.912 |
| V3-UNA-035:ru | PEP 779 объявил сборку без GIL официально поддерживаемой в Python 3.14. | 0.890 |
| V3-UNA-003:en | Sec-WebSocket-Accept is the base64-encoded SHA-1 hash of the client key concatenated with a fixed GUID. | 0.872 |
| V3-UNA-032:en | The copy-and-patch JIT compiles hot bytecode into machine code. | 0.758 |
| V3-UNA-016:ru | Пароль Wi-Fi WPA3 должен содержать не менее 12 символов. | 0.726 |
| V3-UNA-003:ru | Sec-WebSocket-Accept представляет собой base64 от SHA-1 ключа клиента, склеенного с фиксированным GUID. | 0.664 |
| V3-UNA-032:ru | JIT copy-and-patch компилирует горячий байткод в машинный код. | 0.650 |

**Пограничные Neutral / Entailment** (10): верное утверждение, где модель выбирает Neutral (P(ent) ≤ 0.4 при P(con) < 0.2). По категориям: tabular_code: 5, single_hop: 4, multi_hop: 1.

| Пара | Гипотеза | P(ent) | P(neu) |
| :--- | :--- | ---: | ---: |
| V3-TC-020:en | CA-8 lets the organization define how often penetration testing is conducted. | 0.361 | 0.492 |
| V3-SH-025:en | The ABI tag of free-threaded builds contains the letter t, which stands for threading. | 0.315 | 0.636 |
| V3-SH-028:en | Problem details serialized as JSON use the application/problem+json media type. | 0.301 | 0.609 |
| V3-TC-019:en | SI-2 requires installing security-relevant updates within an organization-defined period after their release. | 0.286 | 0.541 |
| V3-TC-031:ru | При меньшем fillfactor команда INSERT заполняет страницы таблицы только до указанного процента. | 0.221 | 0.725 |
| V3-TC-032:en | Single-threaded overhead is 6% on one machine and 5% on the other. | 0.157 | 0.841 |

### 4.2. Сквозной прогон (premise guard вкл. (прод))

**Перехваченные галлюцинации** (1): черновик LLM отвечает на вопрос без ответа или принимает ложную предпосылку, NLI-фильтр заменяет ответ отказом или удаляет утверждения.

| Кейс | Вопрос | Черновик LLM (Vanilla) | Вердикты NLI |
| :--- | :--- | :--- | :--- |
| V3-ADV-021 | SP 800-63B-4 caps consecutive failed authentication attempts with one authenticator at 10… | Документ SP 800-63B-4 ограничивает количество последовательных неудачных попыток аутентификации с использованием одного… | unverified (P(ent) 0.05, P(con) 0.01) |

**Галлюцинации FactRAG-NLI** (3): опубликован ответ на вопрос без ответа или ответ без исправления ложной предпосылки. Если черновик был верным, галлюцинацию внёс фильтр: он публикует только VERIFIED-утверждения, а фраза об отсутствии сведений или исправление предпосылки опорой в источнике не подтверждается.

| Кейс | Вопрос | Черновик LLM (Vanilla) | Опубликованный ответ | Черновик верен |
| :--- | :--- | :--- | :--- | :--- |
| V3-UNA-018 | Какой уровень сертификации FIDO Certified (L1, L2 или L3) SP 800-63B-4 требует … | Single-factor and multi-factor authenticators used at AAL3 SHALL be validated to meet the requirements of [FIPS140] Level 1 or higher overall [S1]. | Single-factor and multi-factor authenticators used at AAL3 SHALL be validated to meet the requirements of [FI… | нет |
| V3-ADV-006 | RFC 9110 requires clients to automatically retry requests with non-idempotent m… | RFC 9110 does not require clients to automatically retry requests with non-idempotent methods after a connection failure. [S2] A client SHOULD NOT au… | RFC 9110 does not require clients to automatically retry requests with non-idempotent methods after a connect… | да |
| V3-UNA-025 | What does RFC 9562 say about the Java method UUID.randomUUID() and its random n… | RFC 9562 не содержит информации о методе UUID.randomUUID() в Java и его генераторе случайных чисел. Однако документ указывает, что UUID версии 4 (UUI… | Однако документ указывает, что UUID версии 4 (UUIDv4) предназначен для генерации UUID на основе истинно случа… | да |

**Потерянные верные ответы** (53 из 55 исходов, которые фильтр сделал неверными; остальные - галлюцинации, внесённые фильтром, см. выше): черновик на отвечаемый вопрос содержал все ключевые факты, опубликованный ответ их не содержит. Исход: ответ заменён отказом по итогам NLI-проверки - 33, ответ опубликован, но фильтр удалил утверждение с ключевым фактом - 20. Вердикты утверждений в этих черновиках: CONTRADICTION: P(con) ≥ 0.2 хотя бы в одном источнике - 47, VERIFIED - 31, UNVERIFIED: нет маркера [S#] после разбиения на предложения, NLI не вызывался - 25, UNVERIFIED: NLI не подтвердил (P(ent) ниже порога, P(con) < 0.2) - 17.

| Кейс | Черновик LLM | Вердикты NLI (первые три утверждения) |
| :--- | :--- | :--- |
| V3-SH-016 | Security and privacy controls of SP 800-53 Rev. 5 are organized into 20 families [S2]. | unverified (missing_citation); contradiction (P(ent) 0.02, P(con) 0.87) |
| V3-SH-018 | Each control family is uniquely identified by a two-character identifier. For example, the identifier for the Personnel Security … | unverified (missing_citation); verified (P(ent) 1.00, P(con) 0.00) |
| V3-MH-022 | Контроли SP 800-53 Rev. 5 разделены на 20 семейств. Из них 17 семейств соответствуют минимальным требованиям FIPS 200. Семейства,… | unverified (missing_citation); unverified (missing_citation); unverified (missing_citation) |
| V3-MH-024 | Базовый контроль AC-7 вводит ограничение на количество организацией определённых последовательных неудачных попыток входа в систе… | unverified (missing_citation); contradiction (P(ent) 0.01, P(con) 0.90) |
| V3-MH-025 | Control families in SP 800-53 Rev. 5 that contain controls directed at developers are System and Services Acquisition (SA) and Su… | unverified (missing_citation); verified (P(ent) 0.80, P(con) 0.19); unverified (P(ent) 0.13, P(con) 0.13) |
| V3-TC-019 | SI-2 требует установки обновлений программного обеспечения и прошивки, относящихся к устранению уязвимостей, в течение [определён… | unverified (P(ent) 0.05, P(con) 0.19) |

**Разбиение на утверждения.** 52 утверждения не получили маркер [S#]. Из них 10 - обрывки предложений (эвристика: оканчиваются на «Rev.», «No.», «;» или начинаются с цифры), например: «Security and privacy controls of SP 800-53 Rev.», «Контроли SP 800-53 Rev.», «5 разделены на 20 семейств.». Защита сокращений в `segment_claims` (`generator_verifier.py`) охватывает только русские сокращения (пп., рис., т. е. и др.), поэтому «SP 800-53 Rev. 5» режется после «Rev.», и первая часть остаётся без цитаты. Точка с запятой разделяет утверждения по правилу, а LLM ставит маркер в конце предложения, поэтому часть до «;» тоже остаётся без цитаты. Остальные 42 - целые предложения, после которых LLM не поставила маркер источника.

## 5. Задержки и ресурсы

### Выбор числа GPU-слоёв LLM

Три последовательных запроса конвейера (реранкинг → генерация → NLI) при процессе PyTorch с моделями на той же карте (2 748 МиБ reserved); llama.cpp сборки 10819 (6a1a922d2), флаги `-c 4096 -b 128 -ub 64 -np 1 --cache-ram 0`. Скорости из журнала llama-server, ток/с по запросам:

| -ngl | VRAM llama, МиБ | После прогона, МиБ | Промпт, ток/с | Генерация, ток/с | Предупреждение «failed to fit» |
| :--- | ---: | ---: | :--- | :--- | ---: |
| 35 | 6 658 | 4 844 | 213 → 54 → 54 | 14.5 → 4.5 → 4.5 | да |
| 27 | 5 292 | 4 940 | 153 → 153 → 153 | 11.2 → 11.0 → 11.2 | нет |
| 25 (выбрано) | 4 946 | 4 980 | 135 → 135 → 136 | 10.7 → 10.4 → 10.4 | нет |

Уменьшение VRAM llama-server после прогона означает, что драйвер вытеснил её буферы в общую память: со второго запроса скорость падает. Сквозной прогон выполнен с `-ngl 25`.

Запросы, дошедшие до генерации (без первого запроса с ленивой загрузкой моделей); мс:

| Профиль | Этап | n | Среднее | p50 | p95 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| premise guard вкл. (прод) | Поиск | 195 | 59 | 58 | 72 |
| premise guard вкл. (прод) | Реранкинг | 195 | 735 | 740 | 812 |
| premise guard вкл. (прод) | Генерация LLM | 195 | 18 133 | 19 199 | 26 659 |
| premise guard вкл. (прод) | NLI-верификация | 195 | 23 | 18 | 67 |
| premise guard вкл. (прод) | Итого | 195 | 18 950 | 19 972 | 27 444 |
| premise guard выкл. | Поиск | 195 | 58 | 58 | 69 |
| premise guard выкл. | Реранкинг | 195 | 728 | 732 | 803 |
| premise guard выкл. | Генерация LLM | 195 | 18 047 | 19 166 | 26 190 |
| premise guard выкл. | NLI-верификация | 195 | 24 | 19 | 62 |
| premise guard выкл. | Итого | 195 | 18 858 | 19 986 | 27 025 |

TTFT LLM не измерялся: в оценённом профиле `RAG_LLM_API_STREAM=false`, ответ llama-server приходит целиком.

![Время этапов](figures/latency_breakdown.svg)

Однократная загрузка моделей при первом запросе: 1 746 мс. Отказ по порогу релевантности обходится без вызова LLM и NLI.

Память (пики за прогон, МиБ; GPU = вся карта по nvidia-smi, включая llama-server):

| Прогон | GPU, вся карта | PyTorch max allocated | RSS процесса оценки | RSS llama-server |
| :--- | ---: | ---: | ---: | ---: |
| Поиск (абляция) | 7 725 | 2 654 | 3 072 | не участвует |
| NLI-пробы | 7 719 | 2 754 | 2 817 | не участвует |
| E2E, premise guard вкл. (прод) | 7 881 | 2 758 | 3 087 | 9 058 |
| E2E, premise guard выкл. | 7 881 | 2 757 | 3 403 | 8 767 |

## 6. Ограничения

- *Vanilla RAG* в этом отчёте - черновик той же LLM с тем же промптом (включая premise guard в прод-профиле) и тем же порогом релевантности, а не внешняя базовая система; сравнение изолирует вклад NLI-фильтра.
- Все документы корпуса англоязычные. При `RAG_GENERATION_LANGUAGE=ru` 153 из 196 черновиков написаны преимущественно кириллицей (на английские вопросы: 65 из 101), поэтому NLI в основном сравнивает русскую гипотезу с английской посылкой; на контрольных парах доля ложной отбраковки для EN- и RU-гипотез приведена в разделе 3.2.
- Разметка датасета подготовлена ИИ; механические проверки (дословные цитаты, отсутствие терминов в корпусе, дедупликация) не заменяют экспертную проверку смысла.
- Корректность ответа оценивается по ключевым фактам (`answer_must_contain`), а не экспертом: ответ с верным числом, но лишним неподтверждённым утверждением считается верным; правильный ответ другими словами может быть засчитан как неверный.
- Галлюцинации на уровне кейса определены для вопросов без ответа и adversarial; лишние неподтверждённые утверждения в ответах на отвечаемые вопросы этой метрикой не учитываются (их видно в статусах утверждений).
- Один прогон на профиль при temperature 0 и фиксированном seed; разброс между прогонами не измерялся.
- Корпус открытый (RFC, NIST, PostgreSQL, PEP), вопросы RU/EN; ГОСТ/СП и другие русскоязычные нормативные документы не измерялись.

## 7. Воспроизведение

```powershell
python scripts/build_eval_dataset_v3.py            # пересобрать и проверить датасет
.\tools\llama\llama-server.exe -m .\models\llm\Qwen2.5-14B-Instruct-Q4_K_M.gguf -ngl 25 -c 4096 -b 128 -ub 64 -np 1 --cache-ram 0 --port 8080 --host 127.0.0.1
python evaluate_suite.py --run-dir reports/runs/20261026T1410Z_v3 retrieval
python evaluate_suite.py --run-dir reports/runs/20261026T1410Z_v3 nli
python evaluate_suite.py --run-dir reports/runs/20261026T1410Z_v3 e2e --profile guard_on
python evaluate_suite.py --run-dir reports/runs/20261026T1410Z_v3 e2e --profile guard_off
python scripts/render_report.py reports/runs/20261026T1410Z_v3
```

Режимы выполняются последовательно: встроенный Qdrant допускает одного владельца индекса. Индекс `.artifacts/eval-v3-index` строится при первом запуске и проверяется при каждом следующем (число чанков и наличие всех эталонных `chunk_id`).
