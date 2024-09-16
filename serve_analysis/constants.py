from typing import Dict, Tuple

TENNIS_KEYPOINTS: Dict[int, str] = {
    5: 'left_shoulder', 6: 'right_shoulder',
    7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist',
    11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee',
    15: 'left_ankle', 16: 'right_ankle'
}

PHASE_COLORS: Dict[str, Tuple[float, float, float]] = {
    'Preparation': (1.0, 0.0, 0.0),    # 赤
    'Backswing': (0.0, 1.0, 0.0),      # 緑
    'Loading': (0.0, 0.0, 1.0),        # 青
    'Forward Swing': (1.0, 1.0, 0.0),  # 黄
    'Impact': (1.0, 0.0, 1.0),         # マゼンタ
    'Follow Through': (0.0, 1.0, 1.0)  # シアン
}