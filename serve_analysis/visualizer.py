import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
from .constants import PHASE_COLORS
from .utils import calculate_angle
from typing import List

def visualize_serve_trajectory(keypoints_history: List[np.ndarray], output_path: str):
    plt.figure(figsize=(12, 8))
    num_frames = len(keypoints_history)
    joints = [6, 8, 10, 12, 14, 16]  # 右肩、右肘、右手首、右腰、右膝、右足首
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    labels = ['Right Shoulder', 'Right Elbow', 'Right Wrist', 'Right Hip', 'Right Knee', 'Right Ankle']
    
    for i, joint in enumerate(joints):
        x = [kp[joint][0] for kp in keypoints_history if isinstance(kp, np.ndarray) and len(kp) > joint and kp.dtype != np.str_]
        y = [kp[joint][1] for kp in keypoints_history if isinstance(kp, np.ndarray) and len(kp) > joint and kp.dtype != np.str_]
        
        for j in range(len(x) - 1):
            alpha = (j + 1) / num_frames
            plt.plot(x[j:j+2], y[j:j+2], color=colors[i], alpha=alpha, linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Serve Trajectory')
    plt.legend(labels)
    plt.gca().invert_yaxis()  # Y軸を反転
    plt.savefig(output_path)
    plt.close()

def visualize_joint_angles(keypoints_history: List[np.ndarray], serve_phases: List[str]):
    angles = {
        'Right Elbow': [],
        'Right Knee': []
    }
    
    for keypoints in keypoints_history:
        if isinstance(keypoints, np.ndarray) and keypoints.dtype != np.str_:
            if len(keypoints) > 8 and len(keypoints) > 10 and len(keypoints) > 6:  # インデックスが範囲内か確認
                shoulder = keypoints[6]
                elbow = keypoints[8]
                wrist = keypoints[10]
                if not np.all(shoulder == 0) and not np.all(elbow == 0) and not np.all(wrist == 0):
                    elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    angles['Right Elbow'].append(elbow_angle)
            
            if len(keypoints) > 12 and len(keypoints) > 14 and len(keypoints) > 16:  # インデックスが範囲内か確認
                hip = keypoints[12]
                knee = keypoints[14]
                ankle = keypoints[16]
                if not np.all(hip == 0) and not np.all(knee == 0) and not np.all(ankle == 0):
                    knee_angle = calculate_angle(hip, knee, ankle)
                    angles['Right Knee'].append(knee_angle)
    
    plt.figure(figsize=(12, 8))
    for joint, angle_list in angles.items():
        plt.plot(angle_list, label=joint)
    
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title('Joint Angles Over Time')
    plt.legend()
    plt.savefig('joint_angles.png')
    plt.close()
