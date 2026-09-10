print("1. Импорт torch...")
import torch
print("   Torch загружен!")

print("2. Импорт transformers...")
from transformers import AutoModelForCausalLM, AutoTokenizer
print("   Transformers загружен!")

print("3. Импорт bitsandbytes...")
import bitsandbytes as bnb
print("   BitsAndBytes загружен!")

print("4. Импорт datasets и evaluate...")
import evaluate
from datasets import load_dataset
print("   Evaluate загружен! ВСЕ ИМПОРТЫ УСПЕШНЫ.")