import torch
import time
import json
import os
import gc
import evaluate
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Конфигурация
MODEL_ID = "unsloth/llama-3-8b"
RESULTS_FILE = "diploma_mega_stats.json"

# Тестовые данные для генерации и метрик (Reference Text)
# Для диплома лучше взять датасет xsum или cnn_dailymail
TEST_PROMPTS = [
    "Summarize the concept of neural network quantization in one sentence.",
    "Translate the following to French: The speed of light is constant."
]
REFERENCES = [
    ["Neural network quantization reduces the precision of weights and activations to save memory and increase speed."],
    ["La vitesse de la lumière est constante."]
]

def get_vram():
    return torch.cuda.memory_allocated() / 1024**2

def measure_latency_and_generate(model, tokenizer, prompt):
    """Замер TTFT (Time To First Token) и TPOT (Time Per Output Token)"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # 1. Замер TTFT (генерируем ровно 1 токен)
    start_time = time.time()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    ttft = time.time() - start_time
    
    # 2. Замер генерации и TPOT (генерируем до 50 токенов)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    total_time = time.time() - start_time
    
    generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    tpot = total_time / generated_tokens if generated_tokens > 0 else 0
    
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return generated_text, round(ttft, 4), round(tpot, 4)

def calculate_nlp_metrics(predictions, references):
    """Оценка BLEU, ROUGE, METEOR (Выполняется на CPU)"""
    print("    -> Подсчет NLP метрик...")
    bleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    
    b_score = bleu.compute(predictions=predictions, references=references)
    r_score = rouge.compute(predictions=predictions, references=references)
    m_score = meteor.compute(predictions=predictions, references=references)
    
    return {
        "BLEU": round(b_score["score"], 2),
        "ROUGE-L": round(r_score["rougeL"], 2),
        "METEOR": round(m_score["meteor"], 2)
    }

def llm_as_a_judge(prediction, reference):
    """Оценка качества через API (например, OpenAI или локальную vLLM)"""
    # В дипломе можно использовать бесплатный tier OpenAI или API Groq для скорости
    # Ниже представлен концепт промпта-судьи.
    judge_prompt = f"""You are an impartial judge evaluating AI models.
    Reference Answer: {reference}
    Model Prediction: {prediction}
    Score the prediction from 1 to 10 based on semantic similarity and accuracy.
    Output ONLY the integer score."""
    
    # Заглушка: в реальности здесь будет запрос к `openai.ChatCompletion.create`
    # return int(api_response)
    return 8 # Dummy score для примера

def run_full_benchmark(name, loader_func):
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
            
    if name in results:
        print(f">>> {name} пропущен (уже в JSON).")
        return

    print(f"\n🚀 ТЕСТ МЕТОДА: {name}")
    model = None; tokenizer = None
    
    try:
        gc.collect()
        torch.cuda.empty_cache()
        
        model, tokenizer = loader_func()
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        mem_weights = get_vram()
        
        # --- БЛОК 1: СКОРОСТЬ И ГЕНЕРАЦИЯ ---
        print("    -> Замер Latency и генерация...")
        predictions = []
        ttft_list, tpot_list = [], []
        
        for prompt in TEST_PROMPTS:
            gen_text, ttft, tpot = measure_latency_and_generate(model, tokenizer, prompt)
            predictions.append(gen_text)
            ttft_list.append(ttft)
            tpot_list.append(tpot)
            
        avg_ttft = sum(ttft_list) / len(ttft_list)
        avg_tpot = sum(tpot_list) / len(tpot_list)
        throughput = 1 / avg_tpot if avg_tpot > 0 else 0

        # УДАЛЯЕМ МОДЕЛЬ ПЕРЕД ТЯЖЕЛЫМИ МЕТРИКАМИ (Защита от OOM)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # --- БЛОК 2: ТОЧНОСТЬ И СЕМАНТИКА ---
        nlp_scores = calculate_nlp_metrics(predictions, REFERENCES)
        
        print("    -> Подсчет BERTScore...")
        bertscore = evaluate.load("bertscore")
        # Выполняется безопасно, так как Llama уже выгружена
        b_res = bertscore.compute(predictions=predictions, references=REFERENCES, lang="en")
        avg_bert = sum(b_res["f1"]) / len(b_res["f1"])

        # --- БЛОК 3: LLM-as-a-Judge ---
        print("    -> Оценка LLM-as-a-Judge...")
        judge_scores = [llm_as_a_judge(p, r[0]) for p, r in zip(predictions, REFERENCES)]
        avg_judge = sum(judge_scores) / len(judge_scores)

        results[name] = {
            "weights_mem_mb": round(mem_weights, 2),
            "latency_ttft_sec": round(avg_ttft, 4),
            "latency_tpot_sec": round(avg_tpot, 4),
            "throughput_tps": round(throughput, 2),
            "nlp_metrics": nlp_scores,
            "bertscore_f1": round(avg_bert, 3),
            "llm_judge_score": round(avg_judge, 2)
        }
        
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"    ✅ УСПЕХ: TPS={throughput:.1f}, TTFT={avg_ttft:.2f}s, Judge Score={avg_judge}")

    except Exception as e:
        print(f"    ❌ ОШИБКА в {name}: {e}")
    finally:
        if model is not None: del model
        if tokenizer is not None: del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

# --- Конфигурации ---
def load_fp16():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

def load_nf4():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=cfg, device_map="auto")
    return model, tokenizer

def load_smoothquant():
    """
    Эмуляция загрузки SmoothQuant (W8A8). 
    В реальном сценарии на Windows без Neural Compressor это сложно запустить на лету.
    Обычно загружают заранее сглаженную модель в INT8.
    """
    # Заглушка: для диплома нужно либо использовать Intel OpenVINO, либо предобученный чекпоинт
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    cfg = BitsAndBytesConfig(load_in_8bit=True) 
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=cfg, device_map="auto")
    return model, tokenizer

methods = {
    "FP16": load_fp16,
    "NF4 (4-bit)": load_nf4,
    "SmoothQuant (W8A8)": load_smoothquant
}

for name, loader in methods.items():
    run_full_benchmark(name, loader)