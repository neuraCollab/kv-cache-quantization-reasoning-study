"""
model_loader.py
Загрузка модели (FP16 / KIVI / KVQuant) и генерация трейсов.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from configs.config import HF_TOKEN, TEMPERATURE, SEED, MAX_NEW_TOKENS


def load_model_fp16(model_id: str) -> tuple:
    """Загружает модель в FP16 (референсный режим)."""
    print(f"[model_loader] Загружаем {model_id} в FP16...")
    _print_vram("до загрузки")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True,
        token=HF_TOKEN or None
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN or None
    )
    model.eval()

    _print_vram("после загрузки FP16")
    return model, tokenizer


def load_model_kivi(model_id: str, bits: int = 2) -> tuple:
    """
    Загружает модель с KIVI KV-квантизацией.
    bits: 2 или 4
    """
    try:
        from kivi import KIVIConfig, patch_model_kivi  # noqa
    except ImportError:
        raise ImportError(
            "KIVI не установлен. Выполни:\n"
            "  git clone https://github.com/jy-liu-hkust/KIVI\n"
            "  pip install -e KIVI/"
        )

    print(f"[model_loader] Загружаем {model_id} с KIVI-{bits}bit...")
    _print_vram("до загрузки")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True,
        token=HF_TOKEN or None
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN or None
    )

    # Применяем KIVI патч
    kivi_config = KIVIConfig(
        bits=bits,
        group_size=32,
        residual_length=128,
    )
    patch_model_kivi(model, kivi_config)
    model.eval()

    _print_vram(f"после загрузки KIVI-{bits}bit")
    return model, tokenizer


def load_model_kvquant(model_id: str, bits: int = 2) -> tuple:
    """
    Загружает модель с KVQuant KV-квантизацией.
    bits: 2 или 4
    """
    try:
        import kvquant  # noqa
    except ImportError:
        raise ImportError(
            "KVQuant не установлен. Выполни:\n"
            "  git clone https://github.com/SqueezeAILab/KVQuant\n"
            "  pip install -e KVQuant/quant/"
        )

    print(f"[model_loader] Загружаем {model_id} с KVQuant-{bits}bit...")
    _print_vram("до загрузки")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True,
        token=HF_TOKEN or None
    )

    # KVQuant грузит через свой API
    from kvquant.modelutils import get_model
    model = get_model(model_id, bits=bits)
    model.eval()

    _print_vram(f"после загрузки KVQuant-{bits}bit")
    return model, tokenizer


def load_model_int8(model_id: str) -> tuple:
    """Загружает модель в INT8 (bitsandbytes) — для 7B на 24GB VRAM."""
    print(f"[model_loader] Загружаем {model_id} в INT8...")
    _print_vram("до загрузки")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True,
        token=HF_TOKEN or None
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN or None
    )
    model.eval()

    _print_vram("после загрузки INT8")
    return model, tokenizer


def generate_trace(
    problem: str,
    model,
    tokenizer,
    method: str = "fp16",
    temperature: float = TEMPERATURE,
    seed: int = SEED,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    """
    Генерирует один трейс рассуждения.

    Returns:
        dict: {raw_output, method, seed, n_tokens, vram_delta_gb}
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt = (
        "<|begin_of_sentence|>User: "
        f"{problem.strip()}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        "<|end_of_sentence|>\nAssistant: <think>\n"
    )

    inputs   = tokenizer(prompt, return_tensors="pt").to(model.device)
    in_len   = inputs["input_ids"].shape[1]
    vram_bef = torch.cuda.memory_allocated() / 1e9

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id,
        )

    vram_aft    = torch.cuda.memory_allocated() / 1e9
    new_tokens  = out[0][in_len:]
    raw_output  = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "raw_output":    raw_output,
        "method":        method,
        "seed":          seed,
        "n_tokens":      int(len(new_tokens)),
        "vram_delta_gb": round(vram_aft - vram_bef, 3),
    }


def unload_model(model) -> None:
    """Выгружает модель из VRAM."""
    del model
    torch.cuda.empty_cache()
    _print_vram("после выгрузки")


def _print_vram(label: str = ""):
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM {label}: {used:.1f}/{total:.1f} GB")
