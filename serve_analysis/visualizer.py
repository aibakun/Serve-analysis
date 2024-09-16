import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import logging
from .utils import calculate_angle, smooth_angle_data
from .constants import PHASE_COLORS

plt.rcParams['font.sans-serif'] = ['Arial']

def visualize_joint_angles(elbow_angles: List[float], knee_angles: List[float], phases: List[str], output_path: str):
    plt.figure(figsize=(15, 10))
    
    # 関節角度のプロット
    plt.plot(elbow_angles, label='Elbow Angle')
    plt.plot(knee_angles, label='Knee Angle')
    
    # フェーズの背景色を追加
    phase_starts = [0] + [i for i in range(1, len(phases)) if phases[i] != phases[i-1]]
    for start, end in zip(phase_starts, phase_starts[1:] + [len(phases)]):
        plt.axvspan(start, end, facecolor=PHASE_COLORS.get(phases[start], 'gray'), alpha=0.3)
    
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Angle (degrees)', fontsize=12)
    plt.title('Joint Angle Changes During Serve', fontsize=14)
    plt.legend(fontsize=10)
    
    # フェーズラベルを追加
    unique_phases = []
    for i, phase in enumerate(phases):
        if phase not in unique_phases:
            unique_phases.append(phase)
            plt.text(i, plt.ylim()[1], phase, rotation=90, verticalalignment='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_phase_angles(elbow_angles: List[float], knee_angles: List[float], phases: List[str], output_path: str):
    phase_angles = {phase: {'Elbow': [], 'Knee': []} for phase in set(phases)}
    
    for i, phase in enumerate(phases):
        phase_angles[phase]['Elbow'].append(elbow_angles[i])
        phase_angles[phase]['Knee'].append(knee_angles[i])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))
    
    for phase, angles in phase_angles.items():
        ax1.boxplot(angles['Elbow'], positions=[list(phase_angles.keys()).index(phase)], labels=[phase])
        ax2.boxplot(angles['Knee'], positions=[list(phase_angles.keys()).index(phase)], labels=[phase])
    
    ax1.set_title('Elbow Angle by Serve Phase', fontsize=14)
    ax1.set_ylabel('Angle (degrees)', fontsize=12)
    ax2.set_title('Knee Angle by Serve Phase', fontsize=14)
    ax2.set_ylabel('Angle (degrees)', fontsize=12)
    ax2.set_xlabel('Serve Phase', fontsize=12)
    
    ax1.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_phase_statistics(elbow_angles: List[float], knee_angles: List[float], phases: List[str]) -> Dict:
    phase_stats = {phase: {'Elbow': [], 'Knee': []} for phase in set(phases)}
    
    for i, phase in enumerate(phases):
        phase_stats[phase]['Elbow'].append(elbow_angles[i])
        phase_stats[phase]['Knee'].append(knee_angles[i])
    
    results = {}
    for phase, angles in phase_stats.items():
        results[phase] = {
            'Elbow': {
                'Min': min(angles['Elbow']),
                'Max': max(angles['Elbow']),
                'Mean': np.mean(angles['Elbow']),
                'Std': np.std(angles['Elbow'])
            },
            'Knee': {
                'Min': min(angles['Knee']),
                'Max': max(angles['Knee']),
                'Mean': np.mean(angles['Knee']),
                'Std': np.std(angles['Knee'])
            }
        }
    
    return results
