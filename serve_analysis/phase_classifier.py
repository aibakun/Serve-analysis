import numpy as np
from typing import List, Dict
from scipy.signal import medfilt, find_peaks
from .utils import calculate_angle, smooth_angle_data, remove_outliers

def analyze_serve_phases(keypoints_history: List[Dict[str, List[float]]]) -> List[str]:
    # 角度データの計算と前処理
    elbow_angles = [calculate_angle(kp['right_shoulder'], kp['right_elbow'], kp['right_wrist']) for kp in keypoints_history]
    knee_angles = [calculate_angle(kp['right_hip'], kp['right_knee'], kp['right_ankle']) for kp in keypoints_history]
    wrist_heights = [kp['right_wrist'][1] for kp in keypoints_history]

    elbow_angles = remove_outliers(elbow_angles)
    knee_angles = remove_outliers(knee_angles)
    wrist_heights = remove_outliers(wrist_heights)

    elbow_angles = smooth_angle_data(elbow_angles)
    knee_angles = smooth_angle_data(knee_angles)
    wrist_heights = smooth_angle_data(wrist_heights)

    total_frames = len(keypoints_history)
    if total_frames < 5:
        return ['Preparation'] * total_frames

    # 動的なフェーズ検出
    elbow_velocity = np.diff(elbow_angles)
    knee_velocity = np.diff(knee_angles)
    wrist_velocity = np.diff(wrist_heights)

    # Preparation終了の検出（手首の下降速度が最大）
    preparation_end = min(np.argmin(wrist_velocity[:total_frames//2]), total_frames - 1)

    # Backswing終了の検出（手首が最も低い位置）
    backswing_end = preparation_end + np.argmin(wrist_heights[preparation_end:])

    # Loading終了の検出（膝の角度が最小）
    loading_end = backswing_end + np.argmin(knee_angles[backswing_end:])

    # Forward Swing開始の検出（肘の角速度が正に最大）
    forward_swing_start = loading_end + np.argmax(elbow_velocity[loading_end:])

    # Impact検出の改善（手首の高さが最大）
    impact_range = wrist_heights[forward_swing_start:]
    impact_frame = forward_swing_start + np.argmax(impact_range)

    # Impact の範囲を制限（前後2フレーム）
    impact_start = max(impact_frame - 2, forward_swing_start)
    impact_end = min(impact_frame + 2, total_frames - 1)

    # Follow Through の終了（インパクト後、手首の高さが最初の極小値になる点）
    follow_through_end = impact_end + 1
    while follow_through_end < total_frames - 1 and wrist_heights[follow_through_end] > wrist_heights[follow_through_end + 1]:
        follow_through_end += 1

    # フェーズの割り当て
    phase_boundaries = [0, preparation_end, backswing_end, loading_end, forward_swing_start, impact_start, impact_end, follow_through_end, total_frames]
    phase_names = ['Preparation', 'Backswing', 'Loading', 'Forward Swing', 'Impact', 'Follow Through']

    phases = []
    for i in range(len(phase_boundaries) - 1):
        phases.extend([phase_names[min(i, len(phase_names)-1)]] * (phase_boundaries[i+1] - phase_boundaries[i]))

    # フェーズのスムージング
    phases = smooth_phases(phases)

    return phases

def smooth_phases(phases: List[str], window_size: int = 5) -> List[str]:
    if not phases:
        return []
    
    phase_to_num = {phase: i for i, phase in enumerate(['Preparation', 'Backswing', 'Loading', 'Forward Swing', 'Impact', 'Follow Through'])}
    num_phases = [phase_to_num[phase] for phase in phases]
    
    # メディアンフィルタを適用
    smoothed_num_phases = medfilt(num_phases, kernel_size=min(window_size, len(num_phases)))
    
    num_to_phase = {i: phase for phase, i in phase_to_num.items()}
    return [num_to_phase[int(num)] for num in smoothed_num_phases]