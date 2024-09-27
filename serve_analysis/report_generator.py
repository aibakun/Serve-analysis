from typing import Dict, List
import numpy as np
from .utils import calculate_angle, smooth_angle_data

def generate_focused_html_report(phases: List[str], elbow_angles: List[float], knee_angles: List[float], image_paths: List[str]):
    phase_order = ['Preparation', 'Backswing', 'Loading', 'Forward Swing', 'Impact', 'Follow Through']
    phase_stats = {phase: {'Elbow': [], 'Knee': []} for phase in phase_order}
    
    for i, phase in enumerate(phases):
        phase_stats[phase]['Elbow'].append(elbow_angles[i])
        phase_stats[phase]['Knee'].append(knee_angles[i])
    
    html_content = """
    <html>
    <head>
        <title>Tennis Serve Analysis Report: Elbow and Knee Angles</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            img { max-width: 100%; height: auto; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Tennis Serve Analysis Report: Elbow and Knee Angles</h1>

        <h2>Joint Angle Statistics by Phase</h2>
        <table>
            <tr><th>Phase</th><th>Joint</th><th>Min Angle</th><th>Max Angle</th><th>Mean Angle</th><th>Std Dev</th></tr>
    """
    
    for phase in phase_order:
        for joint in ['Elbow', 'Knee']:
            if phase_stats[phase][joint]:
                min_angle = min(phase_stats[phase][joint])
                max_angle = max(phase_stats[phase][joint])
                mean_angle = sum(phase_stats[phase][joint]) / len(phase_stats[phase][joint])
                std_dev = np.std(phase_stats[phase][joint])
                html_content += f"<tr><td>{phase}</td><td>{joint}</td><td>{min_angle:.2f}</td><td>{max_angle:.2f}</td><td>{mean_angle:.2f}</td><td>{std_dev:.2f}</td></tr>"
    
    html_content += """
        </table>

        <h2>Visualizations</h2>
    """
    
    for image_path in image_paths:
        html_content += f'<img src="{image_path}" alt="Serve Analysis Visualization">'
    
    html_content += """
    </body>
    </html>
    """
    
    with open('focused_serve_analysis_report.html', 'w') as f:
        f.write(html_content)
