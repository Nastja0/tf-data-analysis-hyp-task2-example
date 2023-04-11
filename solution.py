import pandas as pd
import numpy as np
from hyppo.ksample import MMD


chat_id = 522929689 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    # Измените код этой функции
    # Это будет вашим решением
    # Не меняйте название функции и её аргументы
    p_value = 0.008
    return MMD(compute_kernel="laplacian", gamma=1).test(x, y)[1] <=  p_value# Ваш ответ, True или False
