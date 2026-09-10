# Результаты

Только результаты, для которых есть проверяемая команда воспроизведения в
этом репозитории. Каждый помечен, что именно нужно для запуска — ничего не
требует GPU, кроме явно помеченного.

## 1. Семейство модели определяет устойчивость к квантованию — не размер и не агрессивность метода

DeepSeek-R1-Distill (1.5B и 7B) коллапсирует до 0% accuracy уже на самом
мягком FP8-E4M3; Qwen3-1.7B — модель сравнимого размера — не теряет
точность на том же квантовании (53.7% vs 52.5% baseline, разница в
пределах шума).

**Воспроизвести (CPU, данные уже в репозитории, ~1 сек):**

```bash
python scripts/04_analyze.py \
    --judgments_dir research/kv-cache-reasoning-divergence-study/data/judgments \
    --out_dir outputs
python scripts/05_paper_analysis.py \
    --traces_dir research/kv-cache-reasoning-divergence-study/data/traces \
    --fdps_dir research/kv-cache-reasoning-divergence-study/data/fdps \
    --judgments_dir research/kv-cache-reasoning-divergence-study/data/judgments \
    --out_dir outputs
```

Проверено в этой сессии: `04_analyze.py` даёт файл, побайтово идентичный
`data/report.md`; `05_paper_analysis.py` отрабатывает без ошибок и
пересчитывает все таблицы/графики в `paper/`+`tables/` (см.
`tests/test_paper_analysis.py`, `tests/test_paper_report.py` — регрессионные
тесты именно против этих чисел).

**Пересобрать данные с нуля** (GPU + `ANTHROPIC_API_KEY`, не выполнялось в
этой среде — нет GPU):

```bash
python scripts/01_generate_traces.py --model qwen3-1.7b --config fp8_e4m3
python scripts/02_find_fdps.py --model qwen3-1.7b
python scripts/03_judge_fdps.py
```

## 2. Концентрация шума в outlier-каналах K при FP8-квантовании

Небольшая доля каналов K несёт непропорционально много шума квантования —
механизм: per-tensor FP8-scale считается по максимуму всего тензора, и один
канал-выброс задаёт шаг квантования для всех остальных.

**Воспроизвести (GPU, не выполнялось в этой среде):**

```bash
python scripts/06_kv_capture.py --model qwen3-1.7b --quant fp8_e4m3
cat outputs/kv_capture/qwen3-1.7b/fp8_e4m3/summary.json  # top10_channel_fraction_median
```

Скрипт и метрика (`top_n_outlier_channels`, `outlier_channel_scores`)
проверены на реальной (хоть и игрушечной) модели через
`pytest -m network` — сама механика подсчёта корректна; на игрушечной
модели с head_dim=2 top10_channel_fraction тривиально равен 1.0, содержательное
число появится только на реальной 1.7B-модели.

## 3. Защитный рецепт (top-N каналов в bf16) работает в тесте на одном forward-проходе, но не в реальной генерации

Метрика K-ошибки восстанавливается в teacher-forced измерении, но в
настоящей авторегрессивной генерации эффект падает до уровня шума —
потому что decode-шаг видит только один новый K-токен, и идентичность
outlier-каналов нестабильна между шагами.

**Воспроизвести (GPU, не выполнялось в этой среде):**

```bash
python scripts/08_defense_recipe.py --model qwen3-1.7b --quant fp8_e4m3 --mode both --top_n 10
cat outputs/kv_capture/qwen3-1.7b/fp8_e4m3/defense_tf.json   # TF: undefended_k_error vs defended_k_error
cat outputs/kv_capture/qwen3-1.7b/fp8_e4m3/defense_ar.json   # AR: first_divergence_step, mean_logit_kl
```

Обе функции (`teacher_forced_defense_effect`, `ar_defense_validation`,
последняя — реальный `model.generate()` с кастомным `Cache` и
`output_scores=True`) проверены на реальной модели через `pytest -m network`.
**Важно:** это переисполнение того же эксперимента по восстановленной из
отчёта методологии, а не запуск оригинального (утраченного) скрипта — точные
исходные числа (-34% / -2.4%) этой командой не гарантированно
воспроизводятся, гарантирована только сама методология сравнения.

## 4. Устойчивость outlier-каналов K к семплированию (не артефакт одного прогона)

**Воспроизвести (GPU, не выполнялось в этой среде):**

```bash
python scripts/09_multiseed_jaccard.py --model qwen3-1.7b --seeds 1 2 3 --temperature 0.6
cat outputs/kv_capture/qwen3-1.7b/multiseed_jaccard.json   # median_jaccard
```

## 5. Текстовые признаки baseline-трейса предсказывают точку расхождения для DeepSeek, но не для Qwen3

На 640 реальных строках (без каких-либо K-матриц — только repetition rate,
длина, finish_reason): Spearman до 0.70 на deepseek-1.5b×fp8_e5m2
(p=0.0099 на полной выборке, до учёта поправки на множественные сравнения);
на Qwen3-1.7b сигнала нет вообще.

**Воспроизвести (CPU, данные уже в репозитории, ~1 сек):**

```bash
python scripts/10_fdp_text_predictor.py \
    --fdps_dir research/kv-cache-reasoning-divergence-study/data/fdps \
    --traces_dir research/kv-cache-reasoning-divergence-study/data/traces \
    --out_dir outputs
cat outputs/fdp_text_predictor.md
```

Требует `scikit-learn` (добавлен в `requirements.txt`, нужен только этому
скрипту). Числа выше получены именно этой командой в этой сессии — см.
[`addenda/fdp_text_predictor.json`](addenda/fdp_text_predictor.json) и
честную интерпретацию в [`addenda/README.md`](addenda/README.md).

---

Что не включено в этот файл, потому что физически невоспроизводимо
командой этого репозитория (нет кода / нет данных), см. явный список в
чате, а не здесь.
