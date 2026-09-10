"""
Центральный конфиг проекта.
Все пути и параметры меняй здесь — больше нигде.
"""

import os

# ─── HuggingFace ───────────────────────────────────────────────
HF_TOKEN  = os.environ.get("HF_TOKEN", "")   # задай через: export HF_TOKEN=hf_xxx
HF_REPO   = "mikemikemike1111/kv"                                # опционально: "username/kv-research-traces"

# ─── Модели ────────────────────────────────────────────────────
MODELS = {
    "1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "7B":   "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}
PRIMARY_MODEL = "1.5B"

# ─── Методы квантизации ────────────────────────────────────────
METHODS = ["fp16", "kivi2", "kivi4", "kvquant2"]

# ─── Генерация ─────────────────────────────────────────────────
TEMPERATURE    = 0.6
SEED           = 42
MAX_NEW_TOKENS = 8192

# ─── Пути ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
SRC_DIR    = os.path.join(BASE_DIR, "src")

os.makedirs(DATA_DIR, exist_ok=True)

# ─── Сплиттер шагов ────────────────────────────────────────────
STEP_MIN_WORDS = 15    # шаги короче — мерджить с соседним
STEP_MAX_WORDS = 150   # шаги длиннее — бить по предложениям

# ─── HF sync ───────────────────────────────────────────────────
SYNC_EVERY_N   = 10    # пушить на HF Hub каждые N задач (0 = отключить)
