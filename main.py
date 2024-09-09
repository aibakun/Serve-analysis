import os
import logging
from typing import Dict
import yaml
import tensorflow as tf
import tensorflow_hub as hub

from serve_analysis.video_processor import process_video, get_video_info
from serve_analysis.metrics_calculator import validate_metrics
from serve_analysis.visualizer import visualize_serve_trajectory, visualize_joint_angles
from serve_analysis.report_generator import generate_focused_html_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config() -> Dict:
    """設定ファイルを読み込む"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def setup_tfhub():
    """TensorFlow Hubの設定"""
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
        fps, scale_factor = get_video_info(video_path)
        
        player_height = config['player_height']
        
        metrics, phases, keypoints_history = process_video(video_path, player_height)
        validated_metrics = validate_metrics(metrics)
        
        serve_trajectory_path = "serve_trajectory.png"
        visualize_serve_trajectory(keypoints_history, serve_trajectory_path)
        
        joint_angles_path = "joint_angles.png"
        visualize_joint_angles(keypoints_history, joint_angles_path)
        
        generate_focused_html_report(validated_metrics, serve_trajectory_path, joint_angles_path)
        
        logging.info("Processing completed. Results are saved in 'focused_serve_analysis_report.html' and individual graphs.")
    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    main()