# FactRAG-NLI 2.0 на втором ПК с RTX 3090 (Linux): установка и запуск лучшей конфигурации

**Лучшая конфигурация** (исследование `reports/optimization_2/`, итог - `FINAL_REPORT.md`):
Qwen3.8-27B + bge-reranker-v2-m3, гибридный режим `qwen38-27b-hybrid` и слой решений
`VERIFIER=judge`. Модель отвечает без рассуждения; если ответ не опубликован (отказ модели или
отклонение проверкой), генерирует его ещё раз с рассуждением; если и тогда отказ, а источник
явно подходит, делает две попытки с температурой 0.7. Каждое утверждение ответа проверяют две
NLI-модели (mDeBERTa и bge-m3-zeroshot-v2.0), решение принимает калиброванная логистическая
модель. По повтору полного прогона 0508Z (1307 вопросов): 89.8% верных ответов (было 89.0%),
ложные отказы 7.8% (было 9.2%), ложные утверждения пропускаются в 1.3-11 раз реже. Всё это
включено по умолчанию.

## 1. Что нужно

* Linux (проверено на Ubuntu 24.04), RTX 3090 24 ГБ, драйвер NVIDIA 570 или новее
  (`nvidia-smi` показывает карту);
* Python 3.11-3.12 с venv, curl, unzip; около 45 ГБ на диске;
* интернет: модели скачиваются с Hugging Face (около 24 ГБ).

```bash
sudo apt install -y python3.12 python3.12-venv curl unzip
nvidia-smi
```

## 2. Скачать проект

```bash
cd ~/Downloads
curl -4 -fL -o FactRAG-NLI-rtx3090-linux.zip https://github.com/EGRNEGR/FactRAG-NLI/archive/refs/heads/rtx3090-linux.zip
unzip -q FactRAG-NLI-rtx3090-linux.zip
cd FactRAG-NLI-rtx3090-linux
```

(или `git clone -b rtx3090-linux https://github.com/EGRNEGR/FactRAG-NLI.git FactRAG-NLI-rtx3090-linux`).
Все команды ниже выполняются из папки `FactRAG-NLI-rtx3090-linux`.

## 3. Установка (30-60 минут, в основном загрузка моделей)

```bash
LLM_PROFILES=qwen38-27b-hybrid bash scripts/linux/setup.sh
```

`setup.sh` создаёт `.venv` с PyTorch, ставит llama.cpp b12002 в `tools/llama`, скачивает BGE-M3,
оба реранкера, mDeBERTa, bge-m3-zeroshot-v2.0 и Qwen3.8-27B (с проверкой SHA-256), создаёт
`.env` из `.env.rtx3090.example`, собирает датасеты и запускает тесты (в конце: `passed`, без
`failed`). Если загрузка оборвалась, запустите ту же команду ещё раз: она продолжит с места
обрыва. Без `LLM_PROFILES=...` дополнительно скачается Qwen2.5-14B (8.4 ГБ) для полного плана.

**Уже установленный проект** обновляется так (модели, `.venv`, `.env` и результаты не
затрагиваются; повторный `setup.sh` докачает только новую модель bge-m3-zeroshot, ~1.1 ГБ):

```bash
bash scripts/linux/stop_benchmark.sh      # если идёт прогон
bash scripts/linux/update.sh
LLM_PROFILES=qwen38-27b-hybrid bash scripts/linux/setup.sh
```

## 4. Проверка (около 20 минут)

```bash
PLAN=quick bash scripts/linux/run_benchmark.sh
```

10 вопросов на датасет. В конце лога - `finished ... with exit code 0`, в
`reports/runs/<RUN_ID>/COMPARISON.md` в столбце Errors - 0, строка
`qwen38-27b-hybrid+bge-reranker-v2-m3+judge`.

## 5. Работа с лучшей конфигурацией

```bash
bash scripts/linux/llama_server.sh start        # status | stop | restart
source scripts/linux/env.rtx3090.sh
.venv/bin/python -X utf8 demo.py --ensure-index --query "Какой код состояния означает ..."
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

Ответ всегда со ссылками [S1], [S2] на источники; в логах у каждого утверждения есть степень
обоснованности (`supported`, `partially_supported`, `contradicted`, `ambiguous`,
`insufficient_evidence`) и вероятность поддержки.

## 6. Замер качества

```bash
PLAN=main bash scripts/linux/run_benchmark.sh     # лучшая конфигурация, 1307 вопросов, ~3.5 ч
PLAN=compare bash scripts/linux/run_benchmark.sh  # она же с judge и с rules, ~7 ч
bash scripts/linux/run_benchmark.sh               # полный план с абляциями, ~10.5 ч (нужна Qwen2.5)
```

`VERIFIER` выбирает, как проверяются утверждения ответа:

* `judge` (по умолчанию) - слой решений: каждое утверждение оценивают mDeBERTa и
  bge-m3-zeroshot-v2.0, решение принимает логистическая модель; ложные утверждения
  пропускаются в 1.3-11 раз реже, нужно ~1.2 ГБ видеопамяти на вторую модель;
* `rules` - верификатор v2 прогона 0508Z: только mDeBERTa и пороговые правила; по повтору
  0508Z верных ответов столько же (90.0% против 89.8%), но подменённый ответ или отрицание
  он пропускает заметно чаще (51.5% против 32.6%, 32.6% против 13.6%).

`VERIFIER=rules PLAN=main` - один прогон лучшей конфигурации с верификатором v2.
`PLAN=compare` - обе строки в одном `COMPARISON.md`: `qwen38-27b-hybrid+bge-reranker-v2-m3+judge`
и `qwen38-27b-hybrid+bge-reranker-v2-m3` (rules); Qwen2.5 не нужна.

Каждые 45 минут видеокарта отдыхает 5 минут; прогон идёт в фоне, терминал и AnyDesk можно
закрыть (Ctrl+C только прекращает показ лога).

```
bash scripts/linux/run_benchmark.sh --resume    # продолжить после остановки или перезагрузки
bash scripts/linux/stop_benchmark.sh            # остановить
cat .artifacts/duty_cycle/status.json           # состояние: работа/отдых, температура, VRAM
```

Результат - `reports/runs/<RUN_ID>/COMPARISON.md`. Ожидаемое по повтору 0508Z для строки
`qwen38-27b-hybrid+bge-reranker-v2-m3+judge`: около 89.8% верных (v3 ~86.8%, XQuAD ~90.4%),
ложные отказы ~7.8%, галлюцинации ~1.1%. Архив прогона для разбора:

```bash
tar -czf ~/Desktop/run_$(sed -n 's/^RUN_ID=//p' .artifacts/benchmark/last_run).tgz -C reports/runs "$(sed -n 's/^RUN_ID=//p' .artifacts/benchmark/last_run)"
```

## 7. Неполадки

* `CUDA out of memory` при старте llama-server или в начале e2e: закройте браузер и другие
  программы на видеокарте (`nvidia-smi` покажет, кто занимает память); при запасе меньше 1 ГБ
  временно `VERIFIER=rules PLAN=main bash scripts/linux/run_benchmark.sh` (верификатор v2 без
  второй модели) или `LLAMA_CTX=6144`.
* `CUDA requested but unavailable`: PyTorch не видит GPU (драйвер, `CUDA_VISIBLE_DEVICES`);
  прогон намеренно не переходит на CPU. Проверка: `.venv/bin/python -c "import torch;
  print(torch.cuda.is_available())"`.
* Остальное - `docs/RTX3090_LINUX.md`, раздел 14.
