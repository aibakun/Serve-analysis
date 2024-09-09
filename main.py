import os
import logging
from typing import Dict, List, Tuple
import yaml
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from serve_analysis.video_processor import process_video, get_video_info
from serve_analysis.metrics_calculator import validate_metrics, calculate_serve_metrics, calculate_serve_speed
from serve_analysis.visualizer import (
    visualize_serve_trajectory,
    visualize_joint_angles
)
from serve_analysis.report_generator import generate_focused_html_report

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config() -> Dict:
    """設定ファイルを読み込む"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def setup_tfhub():
    """TensorFlow Hubの設定"""
    # キャッシュディレクトリの設定
    cache_dir = "/tmp/tfhub_cache"
    os.environ["TFHUB_CACHE_DIR"] = cache_dir
    
    # キャッシュディレクトリが存在しない場合は作成
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    logging.info(f"TensorFlow Hubのキャッシュディレクトリを設定しました: {cache_dir}")

def main():
    # 設定ファイルの読み込み
    config = load_config()
    
    # TensorFlow Hubの設定
    setup_tfhub()
    
    # ビデオ情報の取得
    video_path = config['video_path']
    video_info = get_video_info(video_path)
    
    # プレイヤーの身長を取得
    player_height = config['player_height']
    
    # ビデオの処理
    keypoints_history, serve_phases, keypoints_3d = process_video(video_path, player_height)
    
    # メトリクスの計算
    fps = video_info[0]  # タプルの最初の要素を取得
    scale_factor = config.get('scale_factor', 1.0)  # 設定ファイルからスケールファクターを取得、デフォルトは1.0
    metrics = calculate_serve_metrics(keypoints_history, player_height, fps, scale_factor)
    validated_metrics = validate_metrics(metrics)
    
    # 可視化
    output_path = "serve_trajectory.png"
    visualize_serve_trajectory(keypoints_history, output_path)
    visualize_joint_angles(keypoints_history, serve_phases)
    
    # レポートの生成
    generate_focused_html_report(validated_metrics)
    
    logging.info("処理が完了しました。結果は 'focused_serve_analysis_report.html' に保存され、各種グラフも個別に保存されました。")

if __name__ == "__main__":
    main()
