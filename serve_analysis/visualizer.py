import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import logging
from .utils import calculate_angle

plt.rcParams['font.sans-serif'] = ['Arial']  # デフォルトのフォントを設定

def visualize_serve_trajectory(keypoints_history: List[Dict[str, List[float]]], output_path: str):
    try:
        plt.figure(figsize=(12, 8))
        num_frames = len(keypoints_history)
        joints = ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip', 'right_knee', 'right_ankle']
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
        labels = ['Right Shoulder', 'Right Elbow', 'Right Wrist', 'Right Hip', 'Right Knee', 'Right Ankle']
        
        for i, joint in enumerate(joints):
            x = [kp[joint][0] for kp in keypoints_history if joint in kp]
            y = [kp[joint][1] for kp in keypoints_history if joint in kp]
            
            for j in range(len(x) - 1):
                alpha = (j + 1) / num_frames
                plt.plot(x[j:j+2], y[j:j+2], color=colors[i], alpha=alpha, linewidth=2)
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Serve Trajectory')
        plt.legend(labels)
        plt.gca().invert_yaxis()  # Y軸を反転
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Serve trajectory image saved: {output_path}")
    except Exception as e:
        logging.error(f"Error occurred while visualizing serve trajectory: {str(e)}")

def visualize_joint_angles(keypoints_history: List[Dict[str, List[float]]], output_path: str):
    try:
        angles = {
            'Right Elbow': [],
            'Right Knee': []
        }
        
        for keypoints in keypoints_history:
            if all(k in keypoints for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
                shoulder = np.array(keypoints['right_shoulder'][:2])
                elbow = np.array(keypoints['right_elbow'][:2])
                wrist = np.array(keypoints['right_wrist'][:2])
                if not np.all(shoulder == 0) and not np.all(elbow == 0) and not np.all(wrist == 0):
                    elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    angles['Right Elbow'].append(elbow_angle)
            
            if all(k in keypoints for k in ['right_hip', 'right_knee', 'right_ankle']):
                hip = np.array(keypoints['right_hip'][:2])
                knee = np.array(keypoints['right_knee'][:2])
                ankle = np.array(keypoints['right_ankle'][:2])
                if not np.all(hip == 0) and not np.all(knee == 0) and not np.all(ankle == 0):
                    knee_angle = calculate_angle(hip, knee, ankle)
                    angles['Right Knee'].append(knee_angle)
        
        plt.figure(figsize=(12, 8))
        for joint, angle_list in angles.items():
            plt.plot(angle_list, label=joint)
        
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title('Joint Angle Changes')
        plt.legend()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Joint angles image saved: {output_path}")
    except Exception as e:
        logging.error(f"Error occurred while visualizing joint angles: {str(e)}")