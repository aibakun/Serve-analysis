import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import Tuple, List, Dict
import logging
from .constants import TENNIS_KEYPOINTS
from .phase_classifier import analyze_serve_phases

def load_movenet():
    model_handle = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
    module = hub.load(model_handle)
    return module.signatures['serving_default']

def get_video_info(video_path: str) -> Tuple[float, float]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    scale_factor = calculate_scale_factor(cap)
    cap.release()
    return fps, scale_factor

def calculate_scale_factor(cap: cv2.VideoCapture) -> float:
    _, frame = cap.read()
    if frame is not None:
        return frame.shape[0] / 256  # 256はMoveNetの入力サイズ
    return 1.0  # デフォルト値

def process_video(video_path: str) -> List[Dict[str, List[float]]]:
    movenet = load_movenet()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"ビデオファイルを開けませんでした: {video_path}")

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keypoints_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 画像の前処理
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = tf.image.resize_with_pad(tf.expand_dims(input_image, axis=0), 256, 256)
        input_image = tf.cast(input_image, dtype=tf.int32)

        # キーポイントの検出
        results = movenet(input_image)
        keypoints = results['output_0'].numpy().squeeze()

        # キーポイントの後処理
        keypoints[:, 0] *= frame.shape[1] / 256
        keypoints[:, 1] *= frame.shape[0] / 256

        keypoints_dict = {TENNIS_KEYPOINTS[i]: keypoints[i].tolist() for i in TENNIS_KEYPOINTS}
        keypoints_history.append(keypoints_dict)

        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            logging.info(f"処理済みフレーム: {frame_count}/{total_frames} ({progress:.2f}%)")

    cap.release()

    return keypoints_history
