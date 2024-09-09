import numpy as np
from typing import Dict, List, Tuple
from .utils import calculate_angle
import logging

def calculate_serve_metrics(keypoints_history: List[Dict[str, List[float]]], player_height: float, fps: int, scale_factor: float) -> Dict[str, float]:
    metrics = {
        "max_knee_flexion": 0,
        "max_elbow_flexion": 0,
        "max_hip_shoulder_separation": 0,
        "serve_speed": 0
    }

    # プレイヤーの身長を使用してスケールを調整
    height_pixels = np.mean([np.linalg.norm(np.array(kp['left_ankle'][:2]) - np.array(kp['left_shoulder'][:2])) for kp in keypoints_history if 'left_ankle' in kp and 'left_shoulder' in kp])
    scale = player_height / height_pixels
    logging.info(f"Scale factor: {scale}")

    hip_shoulder_angles = []
    for keypoints in keypoints_history:
        # 最大膝屈曲角度
        if all(k in keypoints for k in ['right_hip', 'right_knee', 'right_ankle']):
            hip = np.array(keypoints['right_hip'][:2])
            knee = np.array(keypoints['right_knee'][:2])
            ankle = np.array(keypoints['right_ankle'][:2])
            if not np.all(hip == 0) and not np.all(knee == 0) and not np.all(ankle == 0):
                knee_angle = calculate_angle(hip, knee, ankle)
                metrics["max_knee_flexion"] = max(metrics["max_knee_flexion"], 180 - knee_angle)
        
        # 最大肘屈曲角度
        if all(k in keypoints for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            shoulder = np.array(keypoints['right_shoulder'][:2])
            elbow = np.array(keypoints['right_elbow'][:2])
            wrist = np.array(keypoints['right_wrist'][:2])
            if not np.all(shoulder == 0) and not np.all(elbow == 0) and not np.all(wrist == 0):
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                metrics["max_elbow_flexion"] = max(metrics["max_elbow_flexion"], 180 - elbow_angle)
        
        # 腰-肩分離角度
        if all(k in keypoints for k in ['right_hip', 'left_hip', 'right_shoulder', 'left_shoulder']):
            right_hip = np.array(keypoints['right_hip'][:2])
            left_hip = np.array(keypoints['left_hip'][:2])
            right_shoulder = np.array(keypoints['right_shoulder'][:2])
            left_shoulder = np.array(keypoints['left_shoulder'][:2])
            if not np.all(right_hip == 0) and not np.all(left_hip == 0) and not np.all(right_shoulder == 0) and not np.all(left_shoulder == 0):
                hip_center = (right_hip + left_hip) / 2
                shoulder_center = (right_shoulder + left_shoulder) / 2
                hip_shoulder_vector = shoulder_center - hip_center
                vertical_vector = np.array([0, -1])  # 上向きのベクトル
                hip_shoulder_angle = np.abs(np.degrees(np.arccos(np.clip(np.dot(hip_shoulder_vector, vertical_vector) / (np.linalg.norm(hip_shoulder_vector) * np.linalg.norm(vertical_vector)), -1.0, 1.0))))
                hip_shoulder_angles.append(hip_shoulder_angle)

    # 最大腰-肩分離角度の計算
    if hip_shoulder_angles:
        filtered_angles = [angle for angle in hip_shoulder_angles if angle <= 90]  # 90度以上の角度を除外
        if filtered_angles:
            metrics["max_hip_shoulder_separation"] = np.max(filtered_angles)
        else:
            metrics["max_hip_shoulder_separation"] = np.min(hip_shoulder_angles)  # すべての角度が90度以上の場合、最小値を使用
        logging.info(f"Hip-shoulder separation angles: min={np.min(hip_shoulder_angles):.2f}, max={np.max(hip_shoulder_angles):.2f}, mean={np.mean(hip_shoulder_angles):.2f}, filtered_max={metrics['max_hip_shoulder_separation']:.2f}")

    # サーブスピードの計算
    wrist_positions = [np.array(kp['right_wrist'][:2]) for kp in keypoints_history if 'right_wrist' in kp and not np.all(kp['right_wrist'][:2] == 0)]
    velocities = []
    for i in range(len(wrist_positions) - 1):
        distance = np.linalg.norm(wrist_positions[i+1] - wrist_positions[i]) * scale
        velocity = distance * fps  # m/s
        velocities.append(velocity)

    if velocities:
        # 移動平均フィルタを適用
        window_size = min(5, len(velocities))
        smoothed_velocities = np.convolve(velocities, np.ones(window_size)/window_size, mode='valid')
        max_velocity = np.max(smoothed_velocities)
        serve_speed = max_velocity * 3.6  # m/s から km/h に変換
        metrics["serve_speed"] = serve_speed
        logging.info(f"Serve speed: raw_max={np.max(velocities)*3.6:.2f} km/h, smoothed_max={serve_speed:.2f} km/h")

    return metrics

def validate_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    validated_metrics = {}
    
    validation_rules = {
        'max_knee_flexion': (0, 90),
        'max_elbow_flexion': (0, 180),
        'max_hip_shoulder_separation': (0, 60),
        'serve_speed': (0, 250)
    }
    
    for key, value in metrics.items():
        if value is None or np.isnan(value):
            logging.warning(f"警告: {key} の値が不正です")
            validated_metrics[key] = 0
        elif key in validation_rules:
            min_val, max_val = validation_rules[key]
            if value < min_val or value > max_val:
                logging.warning(f"警告: {key} の値が異常です: {value}")
                validated_metrics[key] = np.clip(value, min_val, max_val)
            else:
                validated_metrics[key] = value
        else:
            validated_metrics[key] = value
    
    return validated_metrics
