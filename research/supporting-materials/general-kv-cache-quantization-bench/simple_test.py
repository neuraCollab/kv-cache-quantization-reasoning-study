import torch
import time
import json
import os
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Используем модель, которая не требует авторизации
MODEL_ID = "unsloth/llama-3-8b" 
RESULTS_FILE = "diploma_final_stats.json"

def get_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_results(data):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def run_test(name, bnb_config):
    data = get_results()
    if name in data:
        print(f"--- {name} уже протестирован. ---")
        return

    print(f"\n>>> Тестирование метода: {name}")
    torch.cuda.empty_cache()
    
    try:
        # 1. Замер памяти до загрузки
        torch.cuda.reset_peak_memory_stats()
        
        # 2. Загрузка модели
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # 3. Замер VRAM (Пиковое значение)
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2
        
        # 4. Тест скорости (на 100 токенах для точности)
        prompt = "The implementation of quantization in deep learning requires"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Warmup
        _ = model.generate(**inputs, max_new_tokens=10)
        
        start_time = time.time()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=100, do_sample=False)
        end_time = time.time()
        
        tps = 100 / (end_time - start_time)
        
        # 5. Сохранение
        data[name] = {"vram": round(vram_mb, 2), "tps": round(tps, 2)}
        save_results(data)
        
        print(f"Результат {name}: {vram_mb:.1f} MB | {tps:.2f} t/s")
        
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Ошибка в {name}: {e}")

# Определение стратегий (согласно твоему списку)
test_suites = {
    "FP16 (Baseline)": None,
    "LLM.int8()": BitsAndBytesConfig(load_in_8bit=True),
    "ZeroQuant (NF4)": BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4"
    ),
    "NF4 + Double Quant": BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True
    )
}

# Запуск тестов
for name, config in test_suites.items():
    run_test(name, config)

# --- Генерация таблицы и графиков для диплома ---
final_data = get_results()
if final_data:
    names = list(final_data.keys())
    vram_vals = [final_data[n]['vram'] for n in names]
    tps_vals = [final_data[n]['tps'] for n in names]

    plt.figure(figsize=(12, 6))
    
    # График VRAM
    plt.subplot(1, 2, 1)
    plt.bar(names, vram_vals, color='teal')
    plt.title('VRAM Usage (Lower is better)')
    plt.xticks(rotation=45)
    plt.ylabel('MB')

    # График TPS
    plt.subplot(1, 2, 2)
    plt.plot(names, tps_vals, marker='o', color='red', linestyle='dashed')
    plt.title('Throughput (Higher is better)')
    plt.xticks(rotation=45)
    plt.ylabel('Tokens/sec')

    plt.tight_layout()
    plt.savefig('diploma_charts.png')
    print("\n[!] Графики сохранены в 'diploma_charts.png'")