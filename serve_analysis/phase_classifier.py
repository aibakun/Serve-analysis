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

    # Preparation終了の検出
    preparation_end = min(np.argmax(np.abs(wrist_velocity[:total_frames//2])), total_frames - 1)

    # Backswing終了の検出（手首が最も低い位置）
    backswing_end = preparation_end + np.argmin(wrist_heights[preparation_end:])

    # Loading終了の検出（膝の角度が最大）
    loading_end = backswing_end + np.argmax(knee_angles[backswing_end:])

    # Forward Swing開始の検出（肘の角度が最小）
    forward_swing_start = loading_end + np.argmin(elbow_angles[loading_end:])

    # Impact検出の改善（手首の高さが最大かつ肘の角速度が最大）
    remaining_frames = total_frames - forward_swing_start
    if remaining_frames > 1:
        wrist_height_peaks, _ = find_peaks(wrist_heights[forward_swing_start:])
        elbow_velocity_peaks, _ = find_peaks(elbow_velocity[forward_swing_start:])
        
        if len(wrist_height_peaks) > 0 and len(elbow_velocity_peaks) > 0:
            impact_candidates = set(wrist_height_peaks).intersection(set(elbow_velocity_peaks))
            if impact_candidates:
                impact_frame = forward_swing_start + min(impact_candidates)
            else:
                impact_frame = forward_swing_start + min(wrist_height_peaks[0], elbow_velocity_peaks[0])
        else:
            impact_frame = forward_swing_start + np.argmax(wrist_heights[forward_swing_start:])
    else:
        impact_frame = forward_swing_start

    # フェーズの割り当て
    phase_boundaries = [0, preparation_end, backswing_end, loading_end, forward_swing_start, impact_frame, total_frames]
    phase_names = ['Preparation', 'Backswing', 'Loading', 'Forward Swing', 'Impact', 'Follow Through']

    phases = []
    for i in range(len(phase_boundaries) - 1):
        phases.extend([phase_names[i]] * (phase_boundaries[i+1] - phase_boundaries[i]))

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
