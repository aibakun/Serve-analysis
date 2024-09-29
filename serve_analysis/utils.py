import numpy as np
from typing import List
from scipy.signal import medfilt, savgol_filter

def calculate_angle(a: List[float], b: List[float], c: List[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def smooth_angle_data(angles: List[float], window_size: int = 11) -> List[float]:
    # Savitzky-Golayフィルタを適用
    if len(angles) < window_size:
        return angles
    return list(savgol_filter(angles, window_size, 3))

def remove_outliers(data: List[float], threshold: float = 2.5) -> List[float]:
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad
    return [d if abs(score) < threshold else median for d, score in zip(data, modified_z_scores)]
