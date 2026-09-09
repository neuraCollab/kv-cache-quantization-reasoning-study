import torch
import matplotlib.pyplot as plt

def pseudo_quantize(tensor, bits=4):
    """
    Базовая функция квантования (Round-to-Nearest).
    Переводит float в int, а затем обратно во float на сетке квантования.
    """
    qmin = -(2**(bits-1))
    qmax = 2**(bits-1) - 1
    
    scale = tensor.abs().max() / qmax
    if scale.abs() < 1e-12:
        return tensor
        
    q_tensor = torch.round(tensor / scale).clamp(qmin, qmax)
    return q_tensor * scale

def run_experiment():
    # 1. Параметры эксперимента
    features = 128
    bits = 4
    outlier_idx = 42 # Индекс канала, где будет выброс
    
    # 2. Генерация весов (W) и активаций (X)
    torch.manual_seed(2026) # Для воспроизводимости
    W = torch.randn(features) 
    X = torch.randn(features) 
    
    # ==========================================
    # TODO 1: Добавь экстремальный выброс в X
    # Умножь значение X по индексу outlier_idx на 100
    # ==========================================
    X[outlier_idx] *= 100.0

    # --- БАЗОВЫЙ ПОДСЧЕТ (БЕЗ МАСШТАБИРОВАНИЯ) ---
    W_q_baseline = pseudo_quantize(W, bits=bits)
    
    # Истинный выход и выход с обычным квантованием
    Y_true = W * X
    Y_baseline = W_q_baseline * X
    
    # Считаем абсолютную ошибку для каждого канала
    error_baseline = torch.abs(Y_true - Y_baseline)


    # --- УМНОЕ МАСШТАБИРОВАНИЕ (AWQ / SmoothQuant) ---
    # ==========================================
    # TODO 2: Вычисли коэффициент масштабирования s
    # Для AWQ используем s = |X|^alpha, где alpha = 0.5
    # ==========================================
    alpha = 0.5
    s_awq = X.abs().pow(alpha).clamp(min=1e-6)
    
    # ==========================================
    # TODO 3: Примени масштабирование ПРАВИЛЬНО
    # Веса ДЕЛЯТСЯ на s, активации УМНОЖАЮТСЯ на s
    # Это позволяет сохранить произведение: (W/s) * (X*s) = W*X
    # ==========================================
    W_scaled = W / s_awq  # ← Веса уменьшаются для каналов с большими активациями
    X_scaled = X * s_awq  # ← Активации увеличиваются
    
    # Квантуем отмасштабированные веса
    W_q_scaled = pseudo_quantize(W_scaled, bits=bits)
    
    # Считаем новый выход (Y = W_q_scaled * X_scaled)
    Y_smart = W_q_scaled * X_scaled
    
    # Ошибка умного метода
    error_smart = torch.abs(Y_true - Y_smart)


    # --- ВИЗУАЛИЗАЦИЯ ---
    # Раскомментируй блок ниже, когда заполнишь TODO

    plt.figure(figsize=(10, 5))
    
    # Строим графики ошибок
    plt.plot(error_baseline.numpy(), label='Ошибка RTN (Без масштабирования)', alpha=0.8, color='red')
    plt.plot(error_smart.numpy(), label='Ошибка AWQ (С масштабированием)', alpha=0.8, color='green')
    
    # Отмечаем канал с выбросом
    plt.axvline(x=outlier_idx, color='black', linestyle='--', label='Канал с выбросом')
    
    # Логарифмическая шкала критически важна, чтобы увидеть разницу
    plt.yscale('log') 
    
    plt.title('Сравнение ошибки квантования на уровне каналов')
    plt.xlabel('Индекс канала')
    plt.ylabel('Абсолютная ошибка (Log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Средняя ошибка (Baseline): {error_baseline.mean().item():.4f}")
    print(f"Средняя ошибка (Smart):    {error_smart.mean().item():.4f}")
    print(f"\nУлучшение: {(error_baseline.mean() / error_smart.mean()).item():.2f}x")


if __name__ == "__main__":
    run_experiment()
