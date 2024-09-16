import numpy as np
from typing import List, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import os
from .utils import calculate_angle
from scipy.signal import medfilt

def smooth_phases(phases: List[str], window_size: int = 5) -> List[str]:
    if not phases:
        return []
    phase_to_num = {phase: i for i, phase in enumerate(['Preparation', 'Backswing', 'Loading', 'Forward Swing', 'Impact', 'Follow Through'])}
    num_phases = [phase_to_num[phase] for phase in phases]
    smoothed_num_phases = medfilt(num_phases, kernel_size=min(window_size, len(num_phases)))
    num_to_phase = {i: phase for phase, i in phase_to_num.items()}
    return [num_to_phase[num] for num in smoothed_num_phases]

def analyze_serve_phases(keypoints_history: List[Dict[str, List[float]]]) -> List[str]:
    initial_phases = initial_phase_classification(keypoints_history)
    if not initial_phases:
        return ['Unknown'] * len(keypoints_history)
    
    window_size = max(3, len(initial_phases) // 20)
    if window_size % 2 == 0:
        window_size += 1  # 偶数の場合は1を加えて奇数にする
    
    smoothed_phases = smooth_phases(initial_phases, window_size=window_size)
    return smoothed_phases

def extract_features(keypoints: Dict[str, List[float]]) -> List[float]:
    features = []
    
    if all(k in keypoints for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
        shoulder = np.array(keypoints['right_shoulder'][:2])
        elbow = np.array(keypoints['right_elbow'][:2])
        wrist = np.array(keypoints['right_wrist'][:2])
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        features.append(elbow_angle)
    else:
        features.append(0)
    
    if all(k in keypoints for k in ['right_hip', 'right_knee', 'right_ankle']):
        hip = np.array(keypoints['right_hip'][:2])
        knee = np.array(keypoints['right_knee'][:2])
        ankle = np.array(keypoints['right_ankle'][:2])
        knee_angle = calculate_angle(hip, knee, ankle)
        features.append(knee_angle)
    else:
        features.append(0)
    
    return features

def initial_phase_classification(keypoints_history: List[Dict[str, List[float]]]) -> List[str]:
    phases = []
    elbow_angles = []
    knee_angles = []
    wrist_heights = []
    
    for keypoints in keypoints_history:
        if all(k in keypoints for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            elbow_angle = calculate_angle(keypoints['right_shoulder'], keypoints['right_elbow'], keypoints['right_wrist'])
            elbow_angles.append(elbow_angle)
        else:
            elbow_angles.append(0)
        
        if all(k in keypoints for k in ['right_hip', 'right_knee', 'right_ankle']):
            knee_angle = calculate_angle(keypoints['right_hip'], keypoints['right_knee'], keypoints['right_ankle'])
            knee_angles.append(knee_angle)
        else:
            knee_angles.append(0)
        
        if 'right_wrist' in keypoints:
            wrist_heights.append(keypoints['right_wrist'][1])
        else:
            wrist_heights.append(0)
    
    # スムージング（奇数のカーネルサイズを保証）
    window_size = max(3, len(elbow_angles) // 20)
    if window_size % 2 == 0:
        window_size += 1
    
    elbow_angles = medfilt(elbow_angles, kernel_size=window_size)
    knee_angles = medfilt(knee_angles, kernel_size=window_size)
    wrist_heights = medfilt(wrist_heights, kernel_size=window_size)
    
    total_frames = len(keypoints_history)
    if total_frames < 5:
        return ['Preparation'] * total_frames

    # 動的なフェーズ検出
    elbow_velocity = np.diff(elbow_angles)
    knee_velocity = np.diff(knee_angles)
    wrist_velocity = np.diff(wrist_heights)

    preparation_end = min(np.argmax(np.abs(wrist_velocity[:total_frames//2])), total_frames - 1)
    backswing_end = min(preparation_end + np.argmin(wrist_heights[preparation_end:]), total_frames - 1)
    loading_end = min(backswing_end + np.argmax(knee_angles[backswing_end:]), total_frames - 1)
    forward_swing_start = min(loading_end + np.argmin(elbow_angles[loading_end:]), total_frames - 1)
    
    # Impact検出の改善
    remaining_frames = total_frames - forward_swing_start
    if remaining_frames > 1:
        forward_swing_velocity = elbow_velocity[forward_swing_start:forward_swing_start + remaining_frames]
        impact_frame = forward_swing_start + np.argmax(forward_swing_velocity)
    else:
        impact_frame = forward_swing_start

    # フェーズの割り当て
    phase_boundaries = [0, preparation_end, backswing_end, loading_end, forward_swing_start, impact_frame, total_frames]
    phase_names = ['Preparation', 'Backswing', 'Loading', 'Forward Swing', 'Impact', 'Follow Through']

    for i in range(len(phase_boundaries) - 1):
        phases.extend([phase_names[i]] * (phase_boundaries[i+1] - phase_boundaries[i]))

    # デバッグ情報
    print(f"Total frames: {total_frames}")
    print(f"Window size: {window_size}")
    print(f"Preparation end: {preparation_end}")
    print(f"Backswing end: {backswing_end}")
    print(f"Loading end: {loading_end}")
    print(f"Forward swing start: {forward_swing_start}")
    print(f"Impact frame: {impact_frame}")

    return phases

def train_phase_classifier(keypoints_history: List[Dict[str, List[float]]], phases: List[str]) -> RandomForestClassifier:
    X = [extract_features(keypoints) for keypoints in keypoints_history]
    y = phases
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Phase classifier accuracy: {accuracy:.2f}")
    
    dump(clf, 'phase_classifier.joblib')
    return clf