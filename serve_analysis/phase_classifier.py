import numpy as np
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
from scipy.signal import find_peaks

from .utils import calculate_angle

def extract_features(keypoints: np.ndarray, prev_keypoints: np.ndarray = None) -> List[float]:
    """キーポイントから特徴量を抽出する"""
    features = []
    
    # 静的特徴
    for i in range(5, 17):  # 肩から足首までのキーポイント
        features.extend(keypoints[i])
    
    # 角度特徴
    right_elbow_angle = calculate_angle(keypoints[6], keypoints[8], keypoints[10])
    right_shoulder_angle = calculate_angle(keypoints[8], keypoints[6], keypoints[12])
    right_hip_angle = calculate_angle(keypoints[6], keypoints[12], keypoints[14])
    right_knee_angle = calculate_angle(keypoints[12], keypoints[14], keypoints[16])
    features.extend([right_elbow_angle, right_shoulder_angle, right_hip_angle, right_knee_angle])
    
    # 動的特徴（速度）
    if prev_keypoints is not None:
        for i in range(5, 17):
            velocity = np.linalg.norm(keypoints[i] - prev_keypoints[i])
            features.append(velocity)
    else:
        features.extend([0] * 12)  # 最初のフレームの場合、速度を0とする
    
    return features

def train_phase_classifier(keypoints_history: List[np.ndarray], phases: List[str]) -> RandomForestClassifier:
    """フェーズ分類器を訓練する"""
    X = []
    y = []
    for i in range(len(keypoints_history)):
        prev_keypoints = keypoints_history[i-1] if i > 0 else None
        features = extract_features(keypoints_history[i], prev_keypoints)
        X.append(features)
        y.append(phases[i])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Phase classifier accuracy: {accuracy:.2f}")
    
    dump(clf, 'phase_classifier.joblib')
    return clf

def analyze_serve_phases(keypoints_history: List[np.ndarray]) -> List[str]:
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
    
    phases = clf.predict(X)
    return list(phases)

def initial_phase_classification(keypoints_history: List[np.ndarray]) -> List[str]:
    """初期フェーズ分類（ルールベース）"""
    phases = []
    velocities = [np.linalg.norm(keypoints_history[i+1][10] - keypoints_history[i][10]) for i in range(len(keypoints_history)-1)]
    peaks, _ = find_peaks(velocities, height=np.mean(velocities), distance=10)
    racket_heights = [kp[10][1] for kp in keypoints_history]
    shoulder_rotations = [np.abs(kp[6][0] - kp[5][0]) for kp in keypoints_history]
    
    preparation_end = np.argmax(shoulder_rotations[:len(shoulder_rotations)//2])
    backswing_end = np.argmin(racket_heights)
    forward_swing_start = peaks[-2] if len(peaks) >= 2 else len(velocities) // 2
    impact_frame = peaks[-1]
    
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
