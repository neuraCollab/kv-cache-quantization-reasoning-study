"""
trace_utils.py
Парсинг сырого вывода модели и разбивка на атомарные шаги.
"""

import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from configs.config import STEP_MIN_WORDS, STEP_MAX_WORDS


def parse_trace(raw_output: str) -> dict:
    """
    Извлекает <think> блок и финальный ответ из сырого вывода модели.

    Обрабатывает 3 кейса:
      1. ok          — нормальный <think>...</think>
      2. unclosed_tag — <think> без закрывающего тега
      3. no_tag       — тега нет вообще

    Returns:
        dict: {think_block, final_answer, parse_status}
    """
    raw = raw_output.strip()

    # Кейс 1: нормальный
    match = re.search(r'<think>(.*?)</think>(.*)', raw, re.DOTALL)
    if match:
        think_block  = match.group(1).strip()
        after_think  = match.group(2).strip()
        return {
            "think_block":  think_block,
            "final_answer": _extract_answer(after_think) or _extract_answer(think_block),
            "parse_status": "ok",
        }

    # Кейс 2: незакрытый тег
    match_open = re.search(r'<think>(.*)', raw, re.DOTALL)
    if match_open:
        think_block = match_open.group(1).strip()
        return {
            "think_block":  think_block,
            "final_answer": _extract_answer(think_block),
            "parse_status": "unclosed_tag",
        }

    # Кейс 3: нет тега
    return {
        "think_block":  raw,
        "final_answer": _extract_answer(raw),
        "parse_status": "no_tag",
    }


def _extract_answer(text: str) -> Optional[str]:
    """Ищет \\boxed{...}, затем последнюю непустую строку."""
    if not text:
        return None
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines[-1] if lines else None


def split_steps(think_block: str) -> list:
    """
    Разбивает think_block на атомарные шаги рассуждения.

    Логика:
      1. Разбивка по \\n\\n
      2. Fallback: по маркерам 'Step N:' / 'Шаг N:'
      3. Постобработка: шаги < STEP_MIN_WORDS — мерджить,
                        шаги > STEP_MAX_WORDS — бить по предложениям

    Returns:
        List[str] — список нормализованных шагов
    """
    if not think_block or not think_block.strip():
        return []

    # Разбивка по двойному переносу
    raw = re.split(r'\n\s*\n', think_block)
    raw = [s.strip() for s in raw if s.strip()]

    # Fallback: маркеры шагов
    if len(raw) <= 1:
        by_marker = re.split(r'(?:Step|Шаг)\s+\d+[:.)\-]', think_block)
        by_marker = [s.strip() for s in by_marker if s.strip()]
        if len(by_marker) > 1:
            raw = by_marker

    steps  = []
    buffer = ""

    for chunk in raw:
        wc = len(chunk.split())

        if wc > STEP_MAX_WORDS:
            # Слишком длинный → сначала дампим буфер, потом бьём по предложениям
            if buffer:
                steps.append(buffer)
                buffer = ""
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            acc = ""
            for sent in sentences:
                candidate = (acc + " " + sent).strip() if acc else sent
                if len(candidate.split()) > STEP_MAX_WORDS and acc:
                    steps.append(acc)
                    acc = sent
                else:
                    acc = candidate
            if acc:
                steps.append(acc)

        elif wc < STEP_MIN_WORDS:
            # Слишком короткий → добавляем в буфер
            buffer = (buffer + " " + chunk).strip() if buffer else chunk

        else:
            # Нормальный размер
            if buffer:
                merged = (buffer + " " + chunk).strip()
                if len(merged.split()) <= STEP_MAX_WORDS:
                    steps.append(merged)
                else:
                    steps.append(buffer)
                    steps.append(chunk)
                buffer = ""
            else:
                steps.append(chunk)

    # Остаток буфера
    if buffer:
        if steps:
            steps[-1] = (steps[-1] + " " + buffer).strip()
        else:
            steps.append(buffer)

    return steps if steps else [think_block.strip()]
