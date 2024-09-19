import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import List, Dict
import logging
from .constants import TENNIS_KEYPOINTS, PHASE_COLORS
from serve_analysis.phase_classifier import analyze_serve_phases

def load_movenet(model_name="movenet_thunder"):
    if model_name == "movenet_thunder":
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    elif model_name == "movenet_lightning":
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    else:
        raise ValueError("Unsupported model name")
    return module.signatures['serving_default']

def process_image(image, model):
    input_image = tf.cast(image, dtype=tf.int32)
    input_image = tf.image.resize_with_pad(input_image, 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)
    input_image = tf.expand_dims(input_image, axis=0)
    
    results = model(input_image)
    keypoints = results['output_0'].numpy().squeeze()
    return keypoints

def convert_keypoints_to_dict(keypoints: np.ndarray) -> Dict[str, List[float]]:
    keypoint_dict = {}
    for i, keypoint_name in enumerate(TENNIS_KEYPOINTS.values()):
        keypoint_dict[keypoint_name] = keypoints[i].tolist()
    return keypoint_dict

def process_video(video_path: str, model_name: str = "movenet_thunder") -> List[Dict[str, List[float]]]:
    model = load_movenet(model_name)

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

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = process_image(frame_rgb, model)
        keypoints_dict = convert_keypoints_to_dict(keypoints)
        keypoints_history.append(keypoints_dict)

        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            logging.info(f"処理済みフレーム: {frame_count}/{total_frames} ({progress:.2f}%)")

    cap.release()

    return keypoints_history

def add_phase_overlay(frame: np.ndarray, phase: str, frame_number: int, total_frames: int) -> np.ndarray:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    
    # フェーズ情報を表示
    cv2.putText(overlay, f"Phase: {phase}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # プログレスバーを表示
    progress = int(width * frame_number / total_frames)
    cv2.rectangle(overlay, (0, height - 20), (progress, height), PHASE_COLORS.get(phase, (0, 0, 0)), -1)
    cv2.putText(overlay, f"{frame_number}/{total_frames}", (10, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

def process_video_with_overlay(video_path: str, phases: List[str], output_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_number in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        phase = phases[frame_number] if frame_number < len(phases) else "Unknown"
        frame_with_overlay = add_phase_overlay(frame, phase, frame_number, total_frames)
        
        out.write(frame_with_overlay)
    
    cap.release()
    out.release()

def analyze_and_visualize_serve(video_path: str, output_path: str, model_name: str = "movenet_thunder"):
    keypoints_history = process_video(video_path, model_name)
    phases = analyze_serve_phases(keypoints_history)
    process_video_with_overlay(video_path, phases, output_path)
    return keypoints_history, phases