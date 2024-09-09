import numpy as np
from typing import Tuple

def normalize_angle(angle: float) -> float:
    """角度を-180から180の範囲に正規化する"""
    return (angle + 180) % 360 - 180

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """3点間の角度を計算する"""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)
