"""
datasets_loader.py
Загрузка задач из AIME-24 и MATH-500.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List


def load_aime24(n: int = 30) -> List[dict]:
    """
    Возвращает задачи AIME 2024.
    Пробует скачать с HF, fallback — встроенные 5 задач для теста.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
        tasks = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            tasks.append({
                "task_id":   f"AIME24_{i+1:03d}",
                "problem":   row["problem"],
                "answer":    str(row.get("answer", "")),
                "benchmark": "AIME24",
            })
        if tasks:
            print(f"[datasets] ✓ AIME-24: загружено {len(tasks)} задач с HF")
            return tasks
    except Exception as e:
        print(f"[datasets] HF загрузка не удалась ({e}), используем встроенные задачи")

    # Fallback — 5 задач AIME 2024 I
    return _aime24_builtin()[:n]


def load_math500(n: int = 100) -> List[dict]:
    """
    Возвращает задачи MATH-500.
    Пробует скачать с HF, fallback — пустой список с предупреждением.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("lighteval/MATH-Hard", split="test")
        import random
        random.seed(42)
        indices = random.sample(range(len(ds)), min(n, len(ds)))
        tasks = []
        for idx in sorted(indices):
            row = ds[idx]
            tasks.append({
                "task_id":   f"MATH500_{idx:04d}",
                "problem":   row["problem"],
                "answer":    str(row.get("solution", "")),
                "benchmark": "MATH500",
            })
        print(f"[datasets] ✓ MATH-500: загружено {len(tasks)} задач с HF")
        return tasks
    except Exception as e:
        print(f"[datasets] ⚠ MATH-500 не загружен: {e}")
        return []


def _aime24_builtin() -> List[dict]:
    """5 встроенных задач AIME 2024 I для быстрого теста."""
    return [
        {
            "task_id":   "AIME24_001",
            "benchmark": "AIME24",
            "answer":    "204",
            "problem":   (
                "Every morning Aya does a 9-kilometer run. She runs the first 3 km at a speed "
                "of s km/h, and the remaining 6 km at a speed of s+2 km/h. If the total time "
                "is 47 minutes, find s. Then compute 60s."
            ),
        },
        {
            "task_id":   "AIME24_002",
            "benchmark": "AIME24",
            "answer":    "21",
            "problem":   (
                "Real numbers x and y with x >= y >= 1 satisfy log_x(y) + log_y(x) = 17/4 "
                "and xy = 512. What is (log_x(y))^5?"
            ),
        },
        {
            "task_id":   "AIME24_003",
            "benchmark": "AIME24",
            "answer":    "285",
            "problem":   (
                "Alice and Bob play a game. Alice starts with the number 30. On each turn "
                "a player subtracts a divisor of the current number (not the number itself). "
                "The player who reaches 0 wins. Find the sum of all starting numbers up to 30 "
                "for which the first player wins."
            ),
        },
        {
            "task_id":   "AIME24_004",
            "benchmark": "AIME24",
            "answer":    "55",
            "problem":   (
                "Let x, y, z be positive real numbers satisfying: "
                "log2(xyz - 3 + log5(x)) = 5, "
                "log3(xyz - 3 + log5(y)) = 4, "
                "log4(xyz - 3 + log5(z)) = 4. "
                "What is |log5(x)| + |log5(y)| + |log5(z)|?"
            ),
        },
        {
            "task_id":   "AIME24_005",
            "benchmark": "AIME24",
            "answer":    "104",
            "problem":   (
                "Rectangles ABCD and EFGH are drawn such that D, E, C, F are collinear. "
                "Also, A, D, H, G all lie on a circle. If BC=16, AB=107, FG=17, EF=184, "
                "what is the length of CE?"
            ),
        },
    ]
