import numpy as np
from typing import Tuple, List
from scipy.signal import medfilt

def normalize_angle(angle: float) -> float:
    """角度を-180から180の範囲に正規化する"""
    return (angle + 180) % 360 - 180

def calculate_angle(a: List[float], b: List[float], c: List[float]) -> float:
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def smooth_angle_data(angles: List[float], window_size: int = 5) -> List[float]:
    # window_size が偶数の場合、1を加えて奇数にする
    if window_size % 2 == 0:
        window_size += 1
    
    # window_size がデータ長を超えないようにする
    window_size = min(window_size, len(angles))
    
    # window_size が 1 未満にならないようにする
    window_size = max(1, window_size)
    
    if window_size == 1:
        return angles
    else:
        return list(medfilt(angles, kernel_size=window_size))
    
def remove_outliers(data: List[float], threshold: float = 2.0) -> List[float]:
    median = np.median(data)
    deviation = np.abs(data - median)
    median_deviation = np.median(deviation)
    s = deviation / median_deviation if median_deviation else 0
    return [d if s_ < threshold else median for d, s_ in zip(data, s)]
