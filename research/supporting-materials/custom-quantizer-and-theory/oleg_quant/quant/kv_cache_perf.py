import torch
import time
import json
import os
import gc
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Конфигурация
MODEL_ID = "unsloth/llama-3-8b" 
RESULTS_FILE = "diploma_full_stats.json"
CTX_LEN_TESTS = [512, 2048, 4096] 

def get_vram():
    return torch.cuda.memory_allocated() / 1024**2

def calculate_perplexity(model, tokenizer, dataset_name="wikitext", split="test", num_samples=20):
    """Оценка точности через Perplexity с защитой от ООМ"""
    print(f"    -> Оценка PPL на {dataset_name} (может занять пару минут)...")
    data = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(data["text"][:num_samples])
    
    # Добавляем pad_token, если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    encodings = tokenizer(text, return_tensors="pt")
    
    max_length = 512 
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    
    # Защита: если текста меньше 512 токенов
    if seq_len < max_length:
        print("    -> Предупреждение: мало текста для PPL.")
        return 0.0

    for begin_loc in tqdm(range(0, seq_len, stride), desc="PPL Calculation"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - begin_loc
        
        # Пропускаем последний кусок, если он слишком мал
        if trg_len < max_length // 2: 
            continue

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss.detach()) # ВАЖНО: detach() чтобы не копить граф

        del input_ids, target_ids, outputs
    
    if not nlls:
        return 0.0
        
    ppl = torch.exp(torch.stack(nlls).mean())
    return round(ppl.item(), 4)

def benchmark_kv_cache(model, tokenizer, lengths):
    print("    -> Тест KV-Cache (реалистичный)...")
    kv_stats = {}

    for length in lengths:
        gc.collect()
        torch.cuda.empty_cache()

        base_mem = get_vram()

        input_ids = torch.ones((1, length), dtype=torch.long).to("cuda")

        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values

            # теперь добавляем 1 токен (как в реальной генерации)
            next_token = torch.ones((1, 1), dtype=torch.long).to("cuda")
            _ = model(next_token, use_cache=True, past_key_values=past_key_values)

        mem_with_cache = get_vram()
        kv_stats[f"kv_{length}_mb"] = round(mem_with_cache - base_mem, 2)

        del input_ids, outputs, past_key_values

    return kv_stats

def run_full_benchmark(name, loader_func):
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
            
    if name in results:
        print(f">>> {name} пропущен (уже в JSON).")
        return

    print(f"\n🚀 ТЕСТ МЕТОДА: {name}")
    
    model = None
    tokenizer = None
    
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model, tokenizer = loader_func()
        
        # Фикс для паддинга
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 1. Базовые замеры
        mem_weights = get_vram()
        print(f"    -> Веса загружены: {mem_weights:.1f} MB")
        
        # 2. Скорость (Throughput)
        prompt = "Technical analysis of quantization shows"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
        
        print("    -> Прогрев...")
        _ = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.pad_token_id)
        
        print("    -> Замер скорости...")
        start = time.time()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        tps = round(50 / (time.time() - start), 2)

        # Очистка перед тяжелыми тестами
        del inputs
        gc.collect()
        torch.cuda.empty_cache()

        # 3. Точность (Perplexity)
        ppl = calculate_perplexity(model, tokenizer)

        # 4. KV-Cache
        kv_data = benchmark_kv_cache(model, tokenizer, CTX_LEN_TESTS)

        results[name] = {
            "weights_mem_mb": round(mem_weights, 2),
            "speed_tps": tps,
            "perplexity": ppl,
            **kv_data
        }
        
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"    ✅ УСПЕХ: PPL={ppl}, Speed={tps} t/s")

    except Exception as e:
        print(f"    ❌ ОШИБКА в {name}: {e}")

    finally:
        # Жесткая очистка памяти
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

# --- Конфигурации ---
def load_fp16():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

def load_nf4():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    cfg = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=cfg, device_map="auto")
    return model, tokenizer

def load_int8():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    cfg = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=cfg, device_map="auto")
    return model, tokenizer

# Запуск
methods = {
    "FP16": load_fp16,
    "INT8 (BitsAndBytes)": load_int8,
    "NF4 (ZeroQuant Style)": load_nf4,
}

for name, loader in methods.items():
    run_full_benchmark(name, loader)

# Итоговая таблица
print("\n" + "="*80)
print(f"{'Метод':<25} | {'VRAM (Веса)':<12} | {'PPL':<8} | {'Скорость':<10} | {'KV 4k (MB)':<10}")
print("-" * 80)
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
        for m, d in data.items():
            print(f"{m:<25} | {d.get('weights_mem_mb', 0):<12.0f} | {d.get('perplexity', 0):<8.2f} | {d.get('speed_tps', 0):<10.1f} | {d.get('kv_4096_mb', 0):<10.1f}")