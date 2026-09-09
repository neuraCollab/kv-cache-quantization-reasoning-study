"""
run_day1.py
─────────────────────────────────────────────────────────────────
День 1: Генерация FP16 трейсов + тест парсера.

Запуск:
    python run_day1.py                        # 5 задач, быстрый тест
    python run_day1.py --tasks 30             # все AIME-24
    python run_day1.py --tasks 30 --math 100  # AIME + MATH-500
    python run_day1.py --test-parser          # только тест парсера, без GPU

Что делает:
    1. Тест parse_trace() и split_steps() без GPU
    2. Загружает модель 1.5B в FP16
    3. Генерирует трейсы на N задачах AIME-24
    4. Парсит, разбивает на шаги, сохраняет в data/traces_fp16.jsonl
    5. Выводит статистику и чеклист
─────────────────────────────────────────────────────────────────
"""

import sys
import os
import argparse
import time

# Чтобы импорты работали из любой директории
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import DATA_DIR, SYNC_EVERY_N, HF_REPO
from src.trace_utils import parse_trace, split_steps
from src.storage import (
    build_record, append_record, load_jsonl,
    get_done_ids, sync_to_hub, print_stats,
)
from src.datasets_loader import load_aime24, load_math500


# ─── Тест парсера (без GPU) ─────────────────────────────────────

def test_parser() -> bool:
    """Прогоняет parse_trace() и split_steps() на синтетических данных."""
    print("\n" + "═"*50)
    print("ТЕСТ ПАРСЕРА (без GPU)")
    print("═"*50)

    # --- parse_trace ---
    cases = [
        ("normal",
         "<think>\nFirst I find the roots.\n\nUsing quadratic: x=2 or x=3.\n</think>\nThe answer is \\boxed{2}",
         "ok", "2"),
        ("unclosed",
         "<think>\nLet me think...\n\nSo the answer is \\boxed{42}",
         "unclosed_tag", "42"),
        ("no_tag",
         "I compute step by step.\n\nFirst: 2+2=4.\n\nFinal: \\boxed{4}",
         "no_tag", "4"),
    ]

    all_ok = True
    print("\nparse_trace():")
    for name, text, exp_status, exp_answer in cases:
        result = parse_trace(text)
        ok = (result["parse_status"] == exp_status and
              result["final_answer"] == exp_answer)
        mark = "✓" if ok else "✗"
        print(f"  [{mark}] {name}: status={result['parse_status']}, answer={result['final_answer']}")
        if not ok:
            all_ok = False
            print(f"       ожидалось: status={exp_status}, answer={exp_answer}")

    # --- split_steps ---
    print("\nsplit_steps():")
    sample = (
        "We need to find the sum of first 10 primes.\n\n"
        "First, I'll list them: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29.\n\n"
        "Now let me add them carefully: 2+3=5, 5+5=10, 10+7=17, 17+11=28.\n\n"
        "Continuing: 28+13=41, 41+17=58, 58+19=77, 77+23=100, 100+29=129.\n\n"
        "The sum is 129. This confirms our calculation is correct."
    )
    steps = split_steps(sample)
    bad = [s for s in steps if len(s.split()) < 10 or len(s.split()) > 200]
    mark = "✓" if not bad else "✗"
    print(f"  [{mark}] {len(steps)} шагов, вне нормы: {len(bad)}")
    for i, s in enumerate(steps):
        print(f"       Шаг {i+1} ({len(s.split())} сл.): {s[:70]}...")

    if all_ok and not bad:
        print("\n✓ Тест парсера пройден\n")
    else:
        print("\n✗ Тест парсера НЕ пройден — проверь trace_utils.py\n")

    return all_ok and not bad


# ─── Основной прогон ────────────────────────────────────────────

def run(n_aime: int = 5, n_math: int = 0) -> None:
    # Импортируем torch только когда нужен GPU
    import torch
    from src.model_loader import load_model_fp16, generate_trace, unload_model

    output_file = os.path.join(DATA_DIR, "traces_fp16.jsonl")
    done_ids    = get_done_ids(output_file)
    print(f"\nВыходной файл: {output_file}")
    print(f"Уже записано:  {len(done_ids)} трейсов")

    # Загружаем задачи
    tasks = load_aime24(n_aime)
    if n_math > 0:
        tasks += load_math500(n_math)
    print(f"Задач всего:   {len(tasks)}")

    # Убираем уже обработанные (resume)
    todo = [t for t in tasks if f"{t['task_id']}_fp16" not in done_ids]
    print(f"Осталось:      {len(todo)} задач\n")

    if not todo:
        print("Все задачи уже обработаны. Запусти с --tasks большим числом или удали файл.")
        print_stats(output_file)
        return

    # Загружаем модель
    MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model, tokenizer = load_model_fp16(MODEL_ID)

    # Прогон
    print("\n" + "═"*50)
    print(f"ГЕНЕРАЦИЯ FP16 — {len(todo)} задач")
    print("═"*50)

    ok_count  = 0
    err_count = 0
    t_start   = time.time()

    for i, task in enumerate(todo, 1):
        t0 = time.time()
        print(f"\n[{i:3d}/{len(todo)}] {task['task_id']}", end="", flush=True)

        try:
            gen = generate_trace(
                problem=task["problem"],
                model=model,
                tokenizer=tokenizer,
                method="fp16",
            )
            record = build_record(task, gen, MODEL_ID)
            append_record(record, output_file)

            elapsed = time.time() - t0
            status  = "✓" if record["is_correct"] else ("✗" if record["is_correct"] is False else "?")
            print(f" {status} | {gen['n_tokens']:4d} токенов | {record['n_steps']:2d} шагов | "
                  f"ответ: {record['predicted_answer']} | {elapsed:.1f}с")
            ok_count += 1

        except torch.cuda.OutOfMemoryError:
            print(" OOM — пропускаем")
            torch.cuda.empty_cache()
            err_count += 1

        except Exception as e:
            print(f" ОШИБКА: {e}")
            err_count += 1

        # HF sync каждые N задач
        if SYNC_EVERY_N > 0 and i % SYNC_EVERY_N == 0 and HF_REPO:
            sync_to_hub(output_file)

    # Финальная синхронизация
    if HF_REPO:
        sync_to_hub(output_file)

    unload_model(model)

    # Итог
    total_time = time.time() - t_start
    print("\n" + "═"*50)
    print("ИТОГ")
    print("═"*50)
    print(f"  Успешно:  {ok_count}")
    print(f"  Ошибок:   {err_count}")
    print(f"  Время:    {total_time/60:.1f} мин")
    print_stats(output_file)
    _print_checklist(output_file)


def _print_checklist(output_file: str) -> None:
    records = load_jsonl(output_file)
    n       = len(records)
    has_steps   = all("steps" in r and r["steps"] for r in records)
    has_answer  = all("predicted_answer" in r for r in records)
    parse_ok    = sum(1 for r in records if r.get("parse_status") == "ok")

    print("\n─── Чеклист конца Дня 1 ───────────────────────────")
    _check(n > 0,     f"traces_fp16.jsonl существует и содержит {n} записей")
    _check(has_steps, "Все записи содержат непустое поле steps[]")
    _check(has_answer,"Все записи содержат predicted_answer")
    _check(parse_ok == n, f"Статус парсера: {parse_ok}/{n} = ok (остальные: unclosed/no_tag)")
    print("")
    print("  Следующий шаг → День 2:")
    print("    python run_day2.py --pilot   # KIVI-2 пилот на 1 задаче")
    print("──────────────────────────────────────────────────\n")


def _check(cond: bool, msg: str) -> None:
    print(f"  {'✓' if cond else '✗'} {msg}")


# ─── CLI ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="День 1: FP16 baseline трейсы")
    parser.add_argument("--tasks",       type=int, default=5,
                        help="Число задач AIME-24 (default: 5 для теста, 30 для полного)")
    parser.add_argument("--math",        type=int, default=0,
                        help="Число задач MATH-500 (default: 0)")
    parser.add_argument("--test-parser", action="store_true",
                        help="Только тест парсера, без генерации (без GPU)")
    args = parser.parse_args()

    # Всегда сначала тест парсера
    parser_ok = test_parser()
    if not parser_ok:
        print("Тест парсера провален. Исправь trace_utils.py перед запуском генерации.")
        sys.exit(1)

    if args.test_parser:
        sys.exit(0)

    run(n_aime=args.tasks, n_math=args.math)


if __name__ == "__main__":
    main()
