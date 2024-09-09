from typing import Dict, List
import numpy as np

def generate_focused_report(metrics: Dict[str, float]) -> str:
    report = "Focused Tennis Serve Analysis Report\n"
    report += "====================================\n\n"
    
    report += "Key Serve Metrics:\n"
    for metric, value in metrics.items():
        if metric == 'max_knee_flexion':
            report += f"- Maximum Knee Flexion: {value:.2f} degrees\n"
        elif metric == 'max_elbow_flexion':
            report += f"- Maximum Elbow Flexion: {value:.2f} degrees\n"
        elif metric == 'max_hip_shoulder_separation':
            report += f"- Maximum Hip-Shoulder Separation: {value*100:.2f} cm\n"
        elif metric == 'serve_speed':
            report += f"- Serve Speed: {value:.2f} km/h\n"
    
    return report

def generate_focused_html_report(metrics: Dict[str, float]):
    html_content = """
    <html>
    <head>
        <title>Focused Tennis Serve Analysis Report</title>
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
        <h1>Focused Tennis Serve Analysis Report</h1>

        <h2>Key Serve Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th><th>Unit</th></tr>
    """

    for metric, value in metrics.items():
        if metric == 'max_knee_flexion':
            html_content += f"<tr><td>Maximum Knee Flexion</td><td>{value:.2f}</td><td>degrees</td></tr>"
        elif metric == 'max_elbow_flexion':
            html_content += f"<tr><td>Maximum Elbow Flexion</td><td>{value:.2f}</td><td>degrees</td></tr>"
        elif metric == 'max_hip_shoulder_separation':
            html_content += f"<tr><td>Maximum Hip-Shoulder Separation</td><td>{value*100:.2f}</td><td>cm</td></tr>"
        elif metric == 'serve_speed':
            html_content += f"<tr><td>Serve Speed</td><td>{value:.2f}</td><td>km/h</td></tr>"

    html_content += """
        </table>

        <h2>Visualizations</h2>
        <img src="focused_serve_metrics.png" alt="Focused Serve Metrics">
    </body>
    </html>
    """

    with open('focused_serve_analysis_report.html', 'w') as f:
        f.write(html_content)

def generate_recommendations(metrics: Dict[str, float]) -> str:
    recommendations = ""

    if metrics['max_knee_flexion'] is not None:
        if metrics['max_knee_flexion'] < 90:
            recommendations += "- Your knee bend could be improved. Try to incorporate more leg drive into your serve for added power.\n"
        elif metrics['max_knee_flexion'] > 120:
            recommendations += "- Your knee bend is significant. Ensure you're maintaining balance and exploding upwards effectively.\n"
        else:
            recommendations += "- Your knee flexion is good. Continue to work on leg strength and explosiveness.\n"

    if metrics['max_elbow_flexion'] is not None:
        if metrics['max_elbow_flexion'] < 80:
            recommendations += "- Your elbow flexion could be improved. Work on creating a more compact motion for better control and spin.\n"
        elif metrics['max_elbow_flexion'] > 110:
            recommendations += "- Your elbow flexion is significant. Focus on a quicker extension for more power.\n"
        else:
            recommendations += "- Your elbow flexion is good. Continue to work on the timing of your arm extension.\n"

    if metrics['max_hip_shoulder_separation'] is not None:
        if metrics['max_hip_shoulder_separation'] < 20:
            recommendations += "- Your hip-shoulder separation could be improved. Work on creating more torque in your serve motion.\n"
        elif metrics['max_hip_shoulder_separation'] > 50:
            recommendations += "- You have excellent hip-shoulder separation. Focus on timing to maximize the power generated.\n"
        else:
            recommendations += "- Your hip-shoulder separation is good. Continue to work on core strength and flexibility.\n"

    if metrics['serve_speed'] is not None:
        if metrics['serve_speed'] < 160:
            recommendations += "- Your serve speed is relatively low. Focus on improving overall technique and power generation.\n"
        elif metrics['serve_speed'] > 200:
            recommendations += "- Your serve speed is excellent. Work on consistency and placement to maximize its effectiveness.\n"
        else:
            recommendations += "- Your serve speed is good. Continue to work on technique and power to increase it further.\n"

    return recommendations