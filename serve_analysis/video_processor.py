import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import Tuple, List, Dict
import logging
from scipy.signal import savgol_filter

from .constants import TENNIS_KEYPOINTS
from .metrics_calculator import calculate_serve_metrics
from .phase_classifier import analyze_serve_phases

def load_movenet():
    """MoveNetモデルをロードする"""
    model_handle = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
    module = hub.load(model_handle)
    return module.signatures['serving_default']

def get_video_info(video_path: str) -> Tuple[float, float]:
    """ビデオの情報を取得する"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    scale_factor = calculate_scale_factor(cap)
    cap.release()
    return fps, scale_factor

def calculate_scale_factor(cap: cv2.VideoCapture) -> float:
    """ビデオのスケールファクターを計算する"""
    _, frame = cap.read()
    if frame is not None:
        return frame.shape[0] / 256  # 256はMoveNetの入力サイズ
    return 1.0  # デフォルト値

def process_video(video_path: str, player_height: float) -> Tuple[Dict[str, float], List[str], List[Dict[str, List[float]]]]:
    """ビデオを処理し、メトリクス、フェーズ、キーポイント履歴を返す"""
    movenet = load_movenet()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"ビデオファイルを開けませんでした: {video_path}")

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keypoints_history = []

    fps, scale_factor = get_video_info(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = tf.image.resize_with_pad(tf.expand_dims(input_image, axis=0), 256, 256)
        input_image = tf.cast(input_image, dtype=tf.int32)

        results = movenet(input=input_image)
        keypoints = results['output_0'].numpy().squeeze()

        keypoints[:, 0] *= frame.shape[1] / 256
        keypoints[:, 1] *= frame.shape[0] / 256

        keypoints_dict = {TENNIS_KEYPOINTS[i]: keypoints[i].tolist() for i in TENNIS_KEYPOINTS}
        keypoints_history.append(keypoints_dict)

        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            logging.info(f"処理済みフレーム: {frame_count}/{total_frames} ({progress:.2f}%)")

    cap.release()

    # Savitzky-Golayフィルタを適用してノイズを軽減
    if len(keypoints_history) > 7:
        window_length = min(7, len(keypoints_history) - 1)
        if window_length % 2 == 0:
            window_length -= 1
        
        smoothed_keypoints_dict = {}
        for key in TENNIS_KEYPOINTS.values():
            keypoint_data = np.array([frame[key] for frame in keypoints_history])
            smoothed_data = savgol_filter(keypoint_data, window_length=window_length, polyorder=3, axis=0)
            smoothed_keypoints_dict[key] = smoothed_data.tolist()

        logging.info(f"Smoothed keypoints shape: {len(smoothed_keypoints_dict)}")
        
        # 辞書のリストに変換
        smoothed_keypoints_history = [
            {key: smoothed_keypoints_dict[key][i] for key in TENNIS_KEYPOINTS.values()}
            for i in range(len(smoothed_keypoints_dict[list(TENNIS_KEYPOINTS.values())[0]]))
        ]
    else:
        smoothed_keypoints_history = keypoints_history

    logging.info(f"Smoothed keypoints shape: {len(smoothed_keypoints_dict)}")
    logging.info(f"TENNIS_KEYPOINTS length: {len(TENNIS_KEYPOINTS)}")
    logging.info(f"Smoothed keypoints history length: {len(smoothed_keypoints_history)}")
    
    # サンプルのキーポイントデータを出力
    if smoothed_keypoints_history:
        logging.info(f"Sample keypoint data: {smoothed_keypoints_history[0]}")

    # サーブを分析
    phases = analyze_serve_phases(smoothed_keypoints_history)
    metrics = calculate_serve_metrics(smoothed_keypoints_history, player_height, fps, scale_factor)

    return metrics, phases, smoothed_keypoints_history
