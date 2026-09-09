"""
storage.py
Сохранение трейсов в .jsonl и синхронизация с HF Hub.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from typing import Optional

from configs.config import HF_TOKEN, HF_REPO, SYNC_EVERY_N
from src.trace_utils import parse_trace, split_steps


def build_record(
    task: dict,
    gen_result: dict,
    model_name: str,
) -> dict:
    """
    Собирает финальную JSON-запись из сырого результата генерации.

    Структура:
        task_id, benchmark, model, method, seed, timestamp
        steps[], n_steps, n_tokens, vram_delta_gb
        predicted_answer, correct_answer, is_correct
        parse_status, raw_output
    """
    parsed = parse_trace(gen_result["raw_output"])
    steps  = split_steps(parsed["think_block"])

    predicted  = parsed["final_answer"]
    correct    = task.get("answer")
    is_correct = None
    if correct and correct != "None" and predicted:
        is_correct = (str(predicted).strip() == str(correct).strip())

    return {
        "task_id":          task["task_id"],
        "benchmark":        task.get("benchmark", ""),
        "model":            model_name,
        "method":           gen_result["method"],
        "seed":             gen_result["seed"],
        "timestamp":        datetime.now().isoformat(),

        "steps":            steps,
        "n_steps":          len(steps),
        "n_tokens":         gen_result["n_tokens"],
        "vram_delta_gb":    gen_result["vram_delta_gb"],

        "predicted_answer": predicted,
        "correct_answer":   correct,
        "is_correct":       is_correct,

        "parse_status":     parsed["parse_status"],
        "raw_output":       gen_result["raw_output"],
    }


def append_record(record: dict, filepath: str) -> None:
    """Дописывает одну запись в .jsonl (append mode, потокобезопасно для одного процесса)."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(filepath: str) -> list:
    """Читает .jsonl и возвращает список записей."""
    if not os.path.exists(filepath):
        return []
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[storage] Пропускаем битую строку: {e}")
    return records


def get_done_ids(filepath: str) -> set:
    """Возвращает множество task_id которые уже есть в файле (для resume)."""
    records = load_jsonl(filepath)
    return {r["task_id"] + "_" + r["method"] for r in records}


def sync_to_hub(local_path: str, repo_id: Optional[str] = None) -> bool:
    """
    Пушит файл на HF Hub Dataset.
    Возвращает True если успешно, False если HF недоступен.
    """
    repo = repo_id or HF_REPO
    if not repo or not HF_TOKEN:
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=os.path.basename(local_path),
            repo_id=repo,
            repo_type="dataset",
        )
        print(f"[storage] ✓ Синхронизировано на HF Hub: {repo}")
        return True
    except Exception as e:
        print(f"[storage] ⚠ HF sync не удался: {e}")
        return False


def print_stats(filepath: str) -> None:
    """Выводит краткую статистику по текущему файлу."""
    records = load_jsonl(filepath)
    if not records:
        print("[stats] Файл пуст.")
        return

    total    = len(records)
    by_meth  = {}
    correct  = 0
    total_ok = 0

    for r in records:
        m = r.get("method", "?")
        by_meth[m] = by_meth.get(m, 0) + 1
        if r.get("is_correct") is True:
            correct += 1
        if r.get("is_correct") is not None:
            total_ok += 1

    print(f"\n{'─'*40}")
    print(f"  Записей всего:    {total}")
    print(f"  По методам:       {by_meth}")
    if total_ok:
        print(f"  Accuracy (fp16):  {correct}/{total_ok} = {correct/total_ok*100:.1f}%")
    print(f"{'─'*40}\n")
