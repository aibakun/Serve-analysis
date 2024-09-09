from typing import Dict, Tuple

TENNIS_KEYPOINTS: Dict[int, str] = {
    5: 'left_shoulder', 6: 'right_shoulder',
    7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist',
    11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee',
    15: 'left_ankle', 16: 'right_ankle'
}

PHASE_COLORS: Dict[str, Tuple[int, int, int]] = {
    'Preparation': (255, 0, 0),    # 赤
    'Backswing': (0, 255, 0),      # 緑
    'Loading': (0, 0, 255),        # 青
    'Forward Swing': (255, 255, 0),# 黄
    'Impact': (255, 0, 255),       # マゼンタ
    'Follow Through': (0, 255, 255)# シアン
}
