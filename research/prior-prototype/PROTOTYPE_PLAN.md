# KV Research — Trace-Level Diagnostics

Диагностика failure modes KV-квантизации в compact reasoning моделях.

## Структура проекта

```
kv_research/
├── configs/
│   └── config.py          # все настройки в одном месте
├── src/
│   ├── trace_utils.py     # parse_trace(), split_steps()
│   ├── model_loader.py    # загрузка FP16 / KIVI / KVQuant
│   ├── storage.py         # build_record(), append_record(), HF sync
│   └── datasets_loader.py # AIME-24, MATH-500
├── data/                  # .jsonl файлы с трейсами (создаётся автоматически)
├── scripts/
│   └── setup.sh           # установка на Vast.ai одной командой
├── run_day1.py            # День 1: FP16 baseline
├── requirements.txt
└── README.md
```

## Запуск на Vast.ai

### 1. Поднять инстанс
- GPU: RTX 3090 24GB, interruptible ($0.07/ч)
- Template: PyTorch 2.3 + CUDA 12.1
- Reliability: > 0.95, Internet: > 500 Mbps

### 2. Подключиться по SSH и настроить проект
```bash
# Клонировать проект
git clone https://github.com/YOUR_USERNAME/kv_research.git
cd kv_research

# Установить всё одной командой (~10–30 мин из-за CUDA kernels)
bash scripts/setup.sh

# Задать HF токен
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Запустить в tmux (защита от разрыва SSH)
```bash
tmux new -s day1

# Тест парсера без GPU
python run_day1.py --test-parser

# Пилот — 5 задач (~10 мин)
python run_day1.py --tasks 5

# Полный прогон AIME-24 — 30 задач (~2–3 ч)
python run_day1.py --tasks 30

# С MATH-500
python run_day1.py --tasks 30 --math 100

# Отсоединиться от tmux (прогон продолжается)
# Ctrl+B, затем D
```

### 4. Проверить результаты
```bash
# Подключиться обратно
tmux attach -t day1

# Посмотреть файл
head -n 1 data/traces_fp16.jsonl | python3 -m json.tool | head -30

# Статистика
python3 -c "
from src.storage import print_stats
print_stats('data/traces_fp16.jsonl')
"
```

## Дни проекта

| День | Скрипт | Что делает |
|------|--------|------------|
| 1 | `run_day1.py` | FP16 baseline трейсы |
| 2 | `run_day2.py` | KIVI-2/4 + KVQuant-2 пилот + batch runner |
| 3 | Vast.ai запуск | Полный батч 1.5B + 7B |
| 4 | `run_fdp.py` | FDP detection на всём датасете |
| 5 | `run_judge.py` | LLM-judge классификация + Cohen's κ |
| 6 | `run_stats.py` | Статистика + визуализации |

## HF Hub синхронизация (опционально)

```python
# configs/config.py
HF_TOKEN = "hf_xxx"
HF_REPO  = "username/kv-research-traces"
SYNC_EVERY_N = 10  # пушить каждые 10 задач
```

## Стоп-критерий Дня 1

После `run_day1.py --tasks 5` должен быть файл `data/traces_fp16.jsonl` с 5 записями.
Каждая запись содержит:
- `steps[]` — список из 5–30 шагов рассуждения
- `predicted_answer` — ответ модели
- `parse_status` — `ok` для большинства

Если `n_steps` у всех записей = 1, значит сплиттер не работает — смотри `trace_utils.py`.
#   v a s t - a i - t e s t
