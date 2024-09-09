import numpy as np
from typing import Dict, List, Tuple
from .utils import calculate_angle

def calculate_serve_metrics(keypoints_history: List[Dict[str, List[float]]], player_height: float, fps: int, scale_factor: float) -> Dict[str, float]:
    metrics = {
        "max_knee_flexion": 0,
        "max_elbow_flexion": 0,
        "max_hip_shoulder_separation": 0,
        "serve_speed": 0
    }

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
        
        # 最大腰-肩分離角度
        if all(k in keypoints for k in ['right_hip', 'left_hip', 'right_shoulder', 'left_shoulder']):
            right_hip = np.array(keypoints['right_hip'][:2])
            left_hip = np.array(keypoints['left_hip'][:2])
            right_shoulder = np.array(keypoints['right_shoulder'][:2])
            left_shoulder = np.array(keypoints['left_shoulder'][:2])
            if not np.all(right_hip == 0) and not np.all(left_hip == 0) and not np.all(right_shoulder == 0) and not np.all(left_shoulder == 0):
                hip_line = right_hip - left_hip
                shoulder_line = right_shoulder - left_shoulder
                hip_shoulder_angle = np.abs(np.degrees(np.arctan2(np.cross(hip_line, shoulder_line), np.dot(hip_line, shoulder_line))))
                metrics["max_hip_shoulder_separation"] = max(metrics["max_hip_shoulder_separation"], hip_shoulder_angle)
    
    # サーブスピードの計算
    wrist_positions = [np.array(kp['right_wrist'][:2]) for kp in keypoints_history if 'right_wrist' in kp and not np.all(kp['right_wrist'][:2] == 0)]
    velocities = []
    for i in range(len(wrist_positions) - 1):
        distance = np.linalg.norm(wrist_positions[i+1] - wrist_positions[i])
        velocity = distance * fps * scale_factor  # m/s
        velocities.append(velocity)

    if velocities:
        top_velocities = sorted(velocities)[-int(len(velocities)*0.1):]
        max_velocity = np.mean(top_velocities)
        serve_speed = max_velocity * 3.6 * 1.2  # m/s から km/h に変換し、ラケットの加速を考慮して1.2倍
        metrics["serve_speed"] = serve_speed

    return metrics

def calculate_serve_speed(keypoints_history: List[np.ndarray], fps: int, scale_factor: float) -> float:
    """サーブスピードを計算する"""
    wrist_positions = [kp[10] for kp in keypoints_history if isinstance(kp, np.ndarray) and kp.dtype != np.str_ and len(kp) > 10 and not np.all(kp[10] == 0)]
    velocities = []
    for i in range(len(wrist_positions) - 1):
        if isinstance(wrist_positions[i], np.ndarray) and isinstance(wrist_positions[i+1], np.ndarray):
            distance = np.linalg.norm(wrist_positions[i+1] - wrist_positions[i])
            velocity = distance * fps * scale_factor  # m/s
            velocities.append(velocity)
    
    if velocities:
        # 上位10%の速度の平均を使用
        top_velocities = sorted(velocities)[-int(len(velocities)*0.1):]
        max_velocity = np.mean(top_velocities)
        serve_speed = max_velocity * 3.6 * 1.2  # m/s から km/h に変換し、ラケットの加速を考慮して1.2倍
        return serve_speed
    return 0.0

def validate_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    validated_metrics = {}
    
    for key, value in metrics.items():
        if value is None:
            print(f"Warning: {key} is None")
            validated_metrics[key] = 0
        elif key == 'max_knee_flexion' and (value < 30 or value > 130):
            print(f"Warning: Unusual max knee flexion detected: {value} degrees")
            validated_metrics[key] = 0
        elif key == 'max_elbow_flexion' and (value < 60 or value > 180):
            print(f"Warning: Unusual max elbow flexion detected: {value} degrees")
            validated_metrics[key] = 0
        elif key == 'max_hip_shoulder_separation' and (value < 5 or value > 90):
            print(f"Warning: Unusual max hip-shoulder separation detected: {value} degrees")
            validated_metrics[key] = 0
        elif key == 'serve_speed' and (value < 50 or value > 250):
            print(f"Warning: Unusual serve speed detected: {value} km/h")
            validated_metrics[key] = 0
        else:
            validated_metrics[key] = value
    
    return validated_metrics