import os
import logging
from typing import Dict
import yaml
import tensorflow as tf
import tensorflow_hub as hub

from serve_analysis.video_processor import analyze_and_visualize_serve
from serve_analysis.phase_classifier import analyze_serve_phases
from serve_analysis.visualizer import visualize_joint_angles
from serve_analysis.report_generator import generate_focused_html_report
from serve_analysis.utils import remove_outliers, smooth_angle_data, calculate_angle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config() -> Dict:
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def setup_tfhub():
    cache_dir = "/tmp/tfhub_cache"
    os.environ["TFHUB_CACHE_DIR"] = cache_dir
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    logging.info(f"TensorFlow Hubのキャッシュディレクトリを設定しました: {cache_dir}")

def main():
    try:
        config = load_config()
        setup_tfhub()
        
        video_path = config['video_path']
        output_video_path = "serve_analysis_with_overlay.mp4"
        
        keypoints_history, phases = analyze_and_visualize_serve(video_path, output_video_path)
        
        elbow_angles = [calculate_angle(kp['right_shoulder'], kp['right_elbow'], kp['right_wrist']) for kp in keypoints_history]
        knee_angles = [calculate_angle(kp['right_hip'], kp['right_knee'], kp['right_ankle']) for kp in keypoints_history]
        
        elbow_angles = remove_outliers(elbow_angles)
        knee_angles = remove_outliers(knee_angles)
        
        elbow_angles = smooth_angle_data(elbow_angles)
        knee_angles = smooth_angle_data(knee_angles)
        
        visualize_joint_angles(elbow_angles, knee_angles, phases, "joint_angles.png")
        
        generate_focused_html_report(phases, elbow_angles, knee_angles, ["joint_angles.png"])
        
        logging.info("Processing completed. Results are saved in 'focused_serve_analysis_report.html' and individual graphs.")
    except Exception as e:
        logging.error(f"予期せぬエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    