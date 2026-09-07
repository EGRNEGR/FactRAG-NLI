# Этапы 8–9: локальная генерация и проверка фактов

## Обновление: HTTP llama-server

`RAG_LLM_API_BASE=http://127.0.0.1:8080/v1` теперь является полем RAGSettings;
`extra=forbid` сохраняется. Наличие API URL выбирает HTTP даже при backend=llama_cpp.
Backend=vllm требует API URL. Native Llama загружается только без API URL.
GGUF в llm_model_path разрешён и для llama-server; каталог весов также поддержан.

Клиент httpx передаёт system/user messages, temperature, top_p, max_tokens,
проверяет HTTP-статус и формат ответа, сохраняет usage/finish_reason. Таймаут:
RAG_LLM_API_TIMEOUT=120. Модель задаётся RAG_LLM_API_MODEL или определяется по
единственной записи /v1/models. Прокси и редиректы отключены, URL ограничен loopback.
Ошибки HTTP-инференса не переключают запрос на локальные DLL.

Без универсального API токенизатора HTTP-режим консервативно оценивает размер
промпта числом байтов UTF-8. Это может уменьшить число включённых фрагментов;
фактическое число токенов берётся из ответа сервера. `.env` синхронизирован с
проверенным /props: RAG_LLM_CONTEXT_WINDOW=4096.

Проверки: исходные 29 тестов генератора проходят; полный набор — 101 тест.
`ruff check .` проходит (распакованные зависимости, модели и tools исключены).
`mypy --strict settings.py generator_verifier.py tests/test_generator_verifier.py tests/test_llm_http.py`
проходит. Реальный ответ через HTTP сохранён в .artifacts/http-generation-smoke.json.
Описание прежней локальной CPU-сборки ниже относится к предыдущему этапу.

## Использование

```python
from generator_verifier import LocalLLMGenerator, NLIFactVerifier
from settings import RAGSettings

settings = RAGSettings()
generator = LocalLLMGenerator(settings)
verifier = NLIFactVerifier(settings)
try:
    raw = generator.generate(query, retrieved_chunks)
    verified = verifier.verify(raw)
    print(verified.model_dump_json(indent=2))
finally:
    generator.close()
```

`retrieved_chunks` — результаты HybridRetriever (`SearchResult`, совместимое имя
`RetrievedChunk`) либо `DocumentChunk`. Карта RawGenerationResult.contexts содержит
только реально отправленные модели фрагменты. Пропущенные по бюджету источники
указаны в metadata.omitted_sources; нумерация исходного списка сохраняется.

`GeneratorVerifier.answer` оставлен как мост к существующему pipeline.py. Фабрика
старого pipeline больше не читает устаревшие RAG_GGUF_MODEL/RAG_VLLM_URL и не
подставляет ExtractiveGenerator. Полная переработка pipeline и baseline — следующий этап.

## Настройки

- RAG_LLM_CONTEXT_WINDOW (8192), RAG_LLM_MAX_TOKENS (1024),
  RAG_LLM_TEMPERATURE (0), RAG_LLM_TOP_P (0.9).
- Старые RAG_MAX_TOKENS, RAG_TEMPERATURE, RAG_TOP_P сохраняют совместимость.
- RAG_LLM_GPU_LAYERS=-1, RAG_LLM_CPU_RETRY=true: попытка GPU, затем CPU при
  ошибке ускорителя/нативного runtime. Ошибка CPU не скрывается.
- RAG_NLI_MAX_LENGTH=512; устройство NLI задаётся RAG_MODEL_DEVICE, здесь CPU.
- В этом модуле реализован backend llama_cpp. Выбор vllm завершается явной
  ошибкой production; сетевого адаптера нет.

Нейросети загружаются с диска. Transformers использует local_files_only=True,
trust_remote_code=False, проверяет отсутствие случайно инициализированных весов.
llama.cpp открывает явный GGUF-путь; параметра local_files_only у него нет.

## Правила верификации

- Каждое предложение проверяется только против указанных `[Si]`.
  Отсутствующий, некорректный или неизвестный маркер блокирует утверждение.
- При нескольких ссылках все указанные источники должны подтверждать утверждение;
  противоречие хотя бы в одном имеет приоритет. Повтор одной ссылки не дублирует проверку.
- Полные три вероятности NLI сохраняются для каждой пары. Порядок классов берётся
  из id2label; неопределённые LABEL_0/1/2 не интерпретируются наугад.
- Пара длиннее окна NLI не усечётся: статус UNVERIFIED с причиной в evidence.
  Это предотвращает удаление исключений в конце источника, но снижает полноту проверки.
- Заблокировано строго больше 50% предложений — отказ целиком. При ровно 50%
  остаются подтверждённые предложения; is_reliable=false. Оборванная генерация
  (finish_reason=length) также приводит к отказу.
- hallucination_rate — доля заблокированных предложений (включая UNVERIFIED),
  **не** экспериментально измеренная доля галлюцинаций на корпусе.
- is_reliable=true означает прохождение фильтров этой проверки, не гарантию
  истинности нормы. Семантическая атомизация сложносочинённых фраз и математическое
  доказательство не реализованы: единицей проверки является предложение.
- Development допускает только явно помеченные fallback: отказ генератора и
  консервативную проверку буквального совпадения. Такой ответ не помечается надёжным.

## Проверки и фактический runtime

В этой итерации .venv уже использует Python 3.12.9 (ранее была 3.14).
Восстановлены совместимые бинарные пакеты librt, mypy, lxml и PyYAML.
Тестовые фикстуры изолированы от настоящего .env и весов; секреты не выводятся.

Первоначальная CUDA-сборка llama-cpp-python 0.3.35 завершалась ошибкой
WinError 0xc000001d и на GPU, и на CPU. Проверена и установлена официальная CPU-сборка
той же версии. Предыдущая сохранена в .artifacts/llama-previous-runtime.zip.
Фактическое устройство генерации теперь отражается в metadata.execution_device.
Для GPU потребуется отдельно совместимая сборка; установленный torch 2.6.0+cu124
также предупреждает, что не поддерживает sm_120 этой RTX 5060 Laptop.

Настоящая mDeBERTa на трёх синтетических примерах дала:

| Пример | Решение | Вероятность соответствующего класса |
|---|---|---:|
| Давление 10 МПа, совпадает с источником | VERIFIED | 0.859769 |
| Давление 99 МПа вместо 10 | CONTRADICTION | 0.984569 |
| Температура 90 градусов, которой нет в источнике | UNVERIFIED | 0.986835 (neutral) |

Файлы результатов: .artifacts/nli-smoke.json и .artifacts/generation-nli-smoke.json.
Это синтетические smoke-проверки настоящих весов, не Ragas и не приёмка p95.

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m mypy --strict settings.py generator_verifier.py tests/test_generator_verifier.py
.\.venv\Scripts\python -m ruff check settings.py generator_verifier.py tests/test_generator_verifier.py
```

Strict применяется к коду проекта; обход типов тяжёлых внешних библиотек отключён
в mypy.overrides (в частности, новая NumPy имеет stubs с синтаксисом Python 3.12).
Граница внешнего ML-runtime типизирована через протоколы и валидируемые результаты.

Документация API:
[llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/),
[Transformers](https://huggingface.co/docs/transformers/main_classes/model).
