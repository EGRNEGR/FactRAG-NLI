# Как мы разворачиваем наш релиз

Мы поставляем исходный дистрибутив, а не готовый air-gapped комплект с весами и wheelhouse. Мы сохраняем проверенные версии стенда в evidence/release_runtime.txt. Мы различаем их и минимальные зависимости pyproject.toml: на действующем стенде установлены более новые sentence-transformers и transformers. Мы не заявляем, что установка минимальных версий воспроизводит исторические GPU-метрики.

## Как мы подготавливаем окружение

Мы распаковываем архив в отдельную директорию и используем Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Мы устанавливаем совместимый с sm_120 CUDA-вариант PyTorch отдельно из подготовленного wheelhouse. Мы сверяем версии с evidence/release_runtime.txt; nightly 2.12.0.dev20260408+cu128 может отсутствовать в текущем онлайн-индексе. Для точного воспроизведения мы переносим сохранённые колёса и их хеши; в этот исходный архив мы их не включаем. Мы проверяем реальное вычисление, а не только доступность драйвера:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability()); x=torch.randn(64,64,device='cuda'); print((x@x).shape)"
```

## Как мы доставляем веса

Мы загружаем полные файлы конфигурации, токенизаторы и веса по карточкам источников на машине подготовки, сохраняем revision и SHA-256, затем переносим их в локальный контур:

| Наша локальная директория | Наш источник |
|---|---|
| models/bge-m3 | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| models/bge-reranker-large | [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) |
| models/mdeberta-v3-nli | [mDeBERTa MNLI/XNLI](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) |
| models/llm/Qwen2.5-14B-Instruct-Q4_K_M.gguf | [Qwen GGUF](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF) |

Мы выбираем вариант Q4_K_M. Если источник поставляет GGUF частями, мы объединяем их утилитой llama-gguf-split --merge из нашего комплекта llama.cpp и задаём итоговое имя, указанное выше. Мы сохраняем лицензии моделей и исходных документов вместе с ними; мы не предоставляем собственную лицензию на сторонние материалы.

Мы помещаем совместимые llama-server.exe и зависимые DLL в tools/llama из официального выпуска llama.cpp. Мы используем ранее проверенный бинарный комплект для данного стенда; произвольный новый выпуск требует повторного smoke-теста. Мы не включаем исполняемые файлы и веса в исходный ZIP.

## Как мы задаём локальные настройки

Мы создаём собственный .env в корне после распаковки; мы не переносим рабочий .env автора:

```dotenv
RAG_ENVIRONMENT=production
RAG_ALLOW_FALLBACK=false
RAG_LLM_BACKEND=vllm
RAG_LLM_API_BASE=http://127.0.0.1:8080/v1
RAG_LLM_MODEL_PATH=models/llm/Qwen2.5-14B-Instruct-Q4_K_M.gguf
RAG_EMBEDDING_MODEL_PATH=models/bge-m3
RAG_RERANKER_MODEL_PATH=models/bge-reranker-large
RAG_NLI_MODEL_PATH=models/mdeberta-v3-nli
RAG_DOCUMENT_ROOT=data/documents
RAG_QDRANT_STORAGE_PATH=data/qdrant
```

Мы сверяем актуальные параметры с settings.py и выполняем аудит:

```powershell
python scripts/verify_environment.py
.\scripts\start_rag.ps1
```

Мы используем профиль сервера -ngl 35 -c 4096 -b 128 -ub 64 -np 1. Мы открываем локальный интерфейс по адресу http://127.0.0.1:8501. Мы даём скрипту построить отсутствующий индекс; мы не запускаем второй embedded-клиент Qdrant параллельно Web-процессу. Для ручной индексации мы сначала завершаем Web-интерфейс.

## Как мы пересобираем материалы

```powershell
python -m pytest tests/ -v
python -m mypy --strict *.py scripts tests
python -m ruff check .
python scripts/generate_presentation.py
python scripts/build_release.py --output release_final_rag.zip
```

Мы передаём mypy явный список файлов, если оболочка не раскрывает *.py. Мы используем системный Arial на Windows или DejaVu Sans на Linux; через --font и --bold-font мы можем указать другие TTF с кириллицей.

Мы проверяем CRC архива автоматически. Мы включаем RELEASE_MANIFEST.json с SHA-256 каждого файла. Мы исключаем .env, secrets, credentials, кэши, модели, логи, data/qdrant и data/documents/uploads. Мы сохраняем эти рабочие данные на исходной машине без изменений. Мы не считаем манифест цифровой подписью издателя.
