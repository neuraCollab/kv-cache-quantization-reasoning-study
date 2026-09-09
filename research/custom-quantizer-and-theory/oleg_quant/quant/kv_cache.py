import torch
import time
import json
import os
import gc
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Конфигурация
MODEL_ID = "unsloth/llama-3-8b" # Открытая версия (не требует HF_TOKEN)
RESULTS_FILE = "quant_results_fixed.json"

def get_gpu_memory():
    return torch.cuda.memory_allocated() / 1024**2

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)

def benchmark(name, model_loader_func):
    results = load_results()
    if name in results:
        print(f"--- {name} уже есть в JSON, пропускаем. ---")
        return results[name]

    print(f"\n>>> Тестируем метод: {name}")
    
    # Жесткая очистка памяти перед стартом
    gc.collect()
    torch.cuda.empty_cache()
    
    model = None
    tokenizer = None
    
    try:
        start_time_load = time.time()
        model, tokenizer = model_loader_func()
        load_time = time.time() - start_time_load
        
        mem_used = get_gpu_memory()
        
        # Тест генерации
        input_text = "The mathematical foundation of neural network quantization involves"
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        
        print("    [1/2] Прогрев GPU...")
        _ = model.generate(**inputs, max_new_tokens=5)
        
        print("    [2/2] Замер скорости...")
        start_gen = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        gen_time = time.time() - start_gen
        
        tokens_sec = 50 / gen_time

        res = {
            "memory_mb": round(mem_used, 2),
            "tokens_per_sec": round(tokens_sec, 2),
            "load_time_sec": round(load_time, 2)
        }
        
        results[name] = res
        save_results(results)
        print(f"    УСПЕХ: {res['memory_mb']} MB | {res['tokens_per_sec']} t/s")
        return res

    except Exception as e:
        print(f"    [ОШИБКА] при тестировании {name}: {e}")
        return None

    finally:
        # ЭТОТ БЛОК ВЫПОЛНИТСЯ ВСЕГДА. 
        # Гарантирует, что память будет очищена даже при ошибке!
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("    Память очищена.")

# --- Функции загрузки моделей ---

def load_fp16():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

def load_llm_int8():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=config, device_map="auto")
    return model, tokenizer

def load_4bit_nf4():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=config, device_map="auto")
    return model, tokenizer

# --- Запуск цикла тестов ---
# Я убрал SmoothQuant заглушки, чтобы график строился по реальным данным, которые мы можем получить сейчас
methods = {
    "FP16 (Base)": load_fp16,
    "INT8 (LLM.int8)": load_llm_int8,
    "ZeroQuant (NF4 4bit)": load_4bit_nf4,
}

for name, loader in methods.items():
    benchmark(name, loader)

# --- Генерация таблицы и графиков ---
final_results = load_results()

print("\n" + "="*50)
print(f"{'Метод':<25} | {'VRAM (MB)':<10} | {'Ток/сек':<10}")
print("-" * 50)
for name, data in final_results.items():
    print(f"{name:<25} | {data['memory_mb']:<10} | {data['tokens_per_sec']:<10}")
print("="*50)

# Построение графика
if len(final_results) > 1:
    names = list(final_results.keys())
    vrams = [final_results[n]['memory_mb'] for n in names]
    speeds = [final_results[n]['tokens_per_sec'] for n in names]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.twinx()
    ax1.bar(names, vrams, color='teal', alpha=0.7, label='VRAM (MB)')
    ax2.plot(names, speeds, color='darkred', marker='o', linewidth=2, markersize=8, label='Speed (tokens/s)')

    ax1.set_ylabel('VRAM Usage (MB)', color='teal', fontsize=12)
    ax2.set_ylabel('Generation Speed (tokens/s)', color='darkred', fontsize=12)
    
    plt.title('Сравнение методов квантования (RTX 4070 Ti)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.savefig('quant_chart.png', bbox_inches='tight')
    print("\n[✔] График успешно сохранен как 'quant_chart.png'")
else:
    print("\n[!] Недостаточно данных для построения графика. Должно быть минимум 2 успешных теста.")