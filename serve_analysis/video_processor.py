import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import savgol_filter
from typing import Tuple, List, Dict
import logging

from .constants import TENNIS_KEYPOINTS
from .phase_classifier import analyze_serve_phases
from .metrics_calculator import calculate_serve_metrics

def load_movenet():
    """MoveNetモデルをロードする"""
    model_handle = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
    module = hub.load(model_handle)
    return module.signatures['serving_default']

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # scale_factor の計算ロジックを追加
    scale_factor = calculate_scale_factor(cap)
    cap.release()
    return fps, scale_factor

def calculate_scale_factor(cap):
    # 実際のスケール係数の計算ロジックを実装
    # 例: フレームの高さに基づいて計算
    _, frame = cap.read()
    if frame is not None:
        return frame.shape[0] / 256  # 256はMoveNetの入力サイズ
    return 1.0  # デフォルト値

def process_video(video_path: str, player_height: float) -> Tuple[Dict[str, float], List[str], np.ndarray]:
    """ビデオを処理し、メトリクス、フェーズ、キーポイント履歴を返す"""
    movenet = load_movenet()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"ビデオファイルを開けませんでした: {video_path}")

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keypoints_history = []

    # フレームレートとスケールファクターを取得
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

        # キーポイントを元のフレームサイズにスケーリング
        keypoints[:, 0] *= frame.shape[1] / 256
        keypoints[:, 1] *= frame.shape[0] / 256

        keypoints_history.append(keypoints)

        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            logging.info(f"Processed {frame_count}/{total_frames} frames ({progress:.2f}%)")

    cap.release()

    # Savitzky-Golayフィルタを適用してノイズを軽減
    if len(keypoints_history) > 7:
        window_length = min(7, len(keypoints_history) - 1)
        if window_length % 2 == 0:
            window_length -= 1
        smoothed_keypoints = savgol_filter(np.array(keypoints_history), window_length=window_length, polyorder=3, axis=0)
    else:
        smoothed_keypoints = np.array(keypoints_history)

    # サーブを分析
    phases = analyze_serve_phases(smoothed_keypoints)
    metrics = calculate_serve_metrics(smoothed_keypoints, player_height, fps, scale_factor)

    return metrics, phases, smoothed_keypoints