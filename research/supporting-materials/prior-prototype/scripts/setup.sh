#!/bin/bash
# ─────────────────────────────────────────────────────────────
# setup.sh — первичная настройка окружения на Vast.ai / RunPod
# Запуск: bash scripts/setup.sh
# ─────────────────────────────────────────────────────────────
set -e

echo "=== [1/5] Системные пакеты ==="
apt-get update -qq
apt-get install -y -qq git wget build-essential ninja-build python3-dev tmux htop

echo "=== [2/5] pip + базовые пакеты ==="
pip install -q --upgrade pip setuptools wheel packaging

echo "=== [3/5] ML-стек ==="
pip install -q -r requirements.txt

echo "=== [4/5] KIVI ==="
if [ ! -d "KIVI" ]; then
    git clone https://github.com/jy-yuan/KIVI.git
fi
pip install -e ./KIVI --no-deps || echo "⚠ KIVI установка не удалась — продолжаем без него"

echo "=== [5/5] KVQuant ==="
if [ ! -d "KVQuant" ]; then
    git clone https://github.com/SqueezeAILab/KVQuant.git --depth=1
fi
pip install -q -e KVQuant/quant/ || echo "⚠ KVQuant установка не удалась — продолжаем без него"

echo ""
echo "✓ Установка завершена"
echo ""
echo "Следующий шаг:"
echo "  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
echo "  python run_day1.py --test-parser   # проверка без GPU"
echo "  python run_day1.py --tasks 5       # пилот на 5 задачах"
echo "  python run_day1.py --tasks 30      # полный прогон AIME-24"
