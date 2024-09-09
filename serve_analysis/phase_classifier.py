import numpy as np
from typing import List, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
from scipy.signal import find_peaks

from .utils import calculate_angle
from .constants import TENNIS_KEYPOINTS

def extract_features(keypoints: Dict[str, List[float]], prev_keypoints: Dict[str, List[float]] = None) -> List[float]:
    """キーポイントから特徴量を抽出する"""
    features = []
    
    # 静的特徴
    for key in TENNIS_KEYPOINTS.values():
        if key in keypoints:
            features.extend(keypoints[key][:2])  # x, y座標のみを使用
        else:
            features.extend([0, 0])  # キーポイントが見つからない場合は0を追加
    
    # 角度特徴
    angle_pairs = [
        ('right_shoulder', 'right_elbow', 'right_wrist'),
        ('right_elbow', 'right_shoulder', 'right_hip'),
        ('right_shoulder', 'right_hip', 'right_knee'),
        ('right_hip', 'right_knee', 'right_ankle'),
        ('left_shoulder', 'left_elbow', 'left_wrist'),
        ('left_elbow', 'left_shoulder', 'left_hip'),
        ('left_shoulder', 'left_hip', 'left_knee'),
        ('left_hip', 'left_knee', 'left_ankle')
    ]
    
    for a, b, c in angle_pairs:
        if all(k in keypoints for k in [a, b, c]):
            angle = calculate_angle(
                np.array(keypoints[a][:2]),
                np.array(keypoints[b][:2]),
                np.array(keypoints[c][:2])
            )
            features.append(angle)
        else:
            features.append(0)  # 角度が計算できない場合は0を追加
    
    # 動的特徴（速度）
    if prev_keypoints is not None:
        for key in TENNIS_KEYPOINTS.values():
            if key in keypoints and key in prev_keypoints:
                velocity = np.linalg.norm(np.array(keypoints[key][:2]) - np.array(prev_keypoints[key][:2]))
                features.append(velocity)
            else:
                features.append(0)
    else:
        features.extend([0] * len(TENNIS_KEYPOINTS))  # 最初のフレームの場合、速度を0とする
    
    return features

def analyze_serve_phases(keypoints_history: List[Dict[str, List[float]]]) -> List[str]:
    """サーブのフェーズを分析する"""
    try:
        clf = load('phase_classifier.joblib')
        print("Loaded existing phase classifier.")
    except:
        print("Phase classifier not found. Training a new one.")
        initial_phases = initial_phase_classification(keypoints_history)
        clf = train_phase_classifier(keypoints_history, initial_phases)
    
    X = []
    for i in range(len(keypoints_history)):
        prev_keypoints = keypoints_history[i-1] if i > 0 else None
        features = extract_features(keypoints_history[i], prev_keypoints)
        X.append(features)
    
    # 特徴量の数をチェックし、必要に応じて再トレーニング
    if len(X[0]) != clf.n_features_in_:
        print(f"Feature mismatch. Retraining classifier. Expected {clf.n_features_in_}, got {len(X[0])}.")
        initial_phases = initial_phase_classification(keypoints_history)
        clf = train_phase_classifier(keypoints_history, initial_phases)
    
    phases = clf.predict(X)
    return list(phases)

def initial_phase_classification(keypoints_history: List[Dict[str, List[float]]]) -> List[str]:
    """初期フェーズ分類（ルールベース）"""
    phases = []
    velocities = [np.linalg.norm(np.array(keypoints_history[i+1]['right_wrist'][:2]) - np.array(keypoints_history[i]['right_wrist'][:2])) for i in range(len(keypoints_history)-1)]
    peaks, _ = find_peaks(velocities, height=np.mean(velocities), distance=10)
    racket_heights = [kp['right_wrist'][1] for kp in keypoints_history]
    shoulder_rotations = [np.abs(kp['right_shoulder'][0] - kp['left_shoulder'][0]) for kp in keypoints_history]
    
    preparation_end = np.argmax(shoulder_rotations[:len(shoulder_rotations)//2])
    backswing_end = np.argmin(racket_heights)
    forward_swing_start = peaks[-2] if len(peaks) >= 2 else len(velocities) // 2
    impact_frame = peaks[-1] if peaks.size > 0 else len(velocities) - 1
    
    for i in range(len(keypoints_history)):
        if i < preparation_end:
            phases.append('Preparation')
        elif i < backswing_end:
            phases.append('Backswing')
        elif i < forward_swing_start:
            phases.append('Loading')
        elif i < impact_frame:
            phases.append('Forward Swing')
        elif i == impact_frame:
            phases.append('Impact')
        else:
            phases.append('Follow Through')
    
    return phases

def train_phase_classifier(keypoints_history: List[Dict[str, List[float]]], phases: List[str]) -> RandomForestClassifier:
    """フェーズ分類器を訓練する"""
    X = []
    y = phases
    
    for i in range(len(keypoints_history)):
        prev_keypoints = keypoints_history[i-1] if i > 0 else None
        features = extract_features(keypoints_history[i], prev_keypoints)
        X.append(features)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Phase classifier accuracy: {accuracy:.2f}")
    
    dump(clf, 'phase_classifier.joblib')
    return clf