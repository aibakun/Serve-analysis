import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import List, Dict, Tuple
import logging
from scipy.ndimage import gaussian_filter1d
from .constants import PHASE_COLORS
from serve_analysis.phase_classifier import analyze_serve_phases

# Define the TENNIS_KEYPOINTS
TENNIS_KEYPOINTS = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

# Define the KEYPOINT_DICT
KEYPOINT_DICT = {v: k for k, v in TENNIS_KEYPOINTS.items()}

def load_movenet(model_name="movenet_thunder"):
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    return module.signatures['serving_default']

def process_image(image, model):
    input_image = tf.cast(image, dtype=tf.int32)
    input_image = tf.image.resize_with_pad(input_image, 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)
    input_image = tf.expand_dims(input_image, axis=0)
    
    outputs = model(input_image)
    keypoints = outputs['output_0'].numpy().squeeze()
    return keypoints

def filter_keypoints(keypoints: np.ndarray, confidence_threshold=0.1) -> Dict[str, List[float]]:
    filtered_keypoints = {}
    for i, keypoint_name in enumerate(TENNIS_KEYPOINTS.values()):
        if keypoints[i, 2] > confidence_threshold:
            filtered_keypoints[keypoint_name] = [keypoints[i, 1], keypoints[i, 0]]
    return filtered_keypoints

def smooth_keypoints(keypoints_history: List[Dict[str, List[float]]], window_size=5) -> List[Dict[str, List[float]]]:
    smoothed_history = []
    for i in range(len(keypoints_history)):
        smoothed_frame = {}
        for joint in TENNIS_KEYPOINTS.values():
            joint_data = [frame.get(joint, [0, 0]) for frame in keypoints_history[max(0, i-window_size):i+1]]
            if joint_data:
                smoothed_joint = gaussian_filter1d(joint_data, sigma=1, axis=0)
                smoothed_frame[joint] = smoothed_joint[-1].tolist()
        smoothed_history.append(smoothed_frame)
    return smoothed_history

def interpolate_missing_keypoints(keypoints_history: List[Dict[str, List[float]]]) -> List[Dict[str, List[float]]]:
    interpolated_history = keypoints_history.copy()
    for i in range(1, len(keypoints_history) - 1):
        for joint in TENNIS_KEYPOINTS.values():
            if joint not in interpolated_history[i]:
                prev_frame = next((f for f in reversed(interpolated_history[:i]) if joint in f), None)
                next_frame = next((f for f in interpolated_history[i+1:] if joint in f), None)
                if prev_frame and next_frame:
                    prev_index = interpolated_history.index(prev_frame)
                    next_index = interpolated_history.index(next_frame)
                    t = (i - prev_index) / (next_index - prev_index)
                    interpolated_history[i][joint] = [
                        prev_frame[joint][0] + t * (next_frame[joint][0] - prev_frame[joint][0]),
                        prev_frame[joint][1] + t * (next_frame[joint][1] - prev_frame[joint][1])
                    ]
    return interpolated_history

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
        filtered_keypoints = filter_keypoints(keypoints)
        keypoints_history.append(filtered_keypoints)

        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            logging.info(f"処理済みフレーム: {frame_count}/{total_frames} ({progress:.2f}%)")

    cap.release()

    smoothed_keypoints = smooth_keypoints(keypoints_history)
    interpolated_keypoints = interpolate_missing_keypoints(smoothed_keypoints)

    return interpolated_keypoints

def draw_skeleton(frame: np.ndarray, keypoints: Dict[str, List[float]], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3):
    connections = [
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'),
    ]
    
    height, width = frame.shape[:2]
    for start, end in connections:
        if start in keypoints and end in keypoints:
            start_point = (int(keypoints[start][0] * width), int(keypoints[start][1] * height))
            end_point = (int(keypoints[end][0] * width), int(keypoints[end][1] * height))
            cv2.line(frame, start_point, end_point, color, thickness)
    
    for joint in keypoints:
        if joint in ['left_knee', 'right_knee', 'left_ankle', 'right_ankle']:
            point = (int(keypoints[joint][0] * width), int(keypoints[joint][1] * height))
            cv2.circle(frame, point, 5, color, -1)

    return frame

def process_video_with_overlay(video_path: str, keypoints_history: List[Dict[str, List[float]]], phases: List[str], output_path: str):
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
        
        if frame_number < len(keypoints_history):
            frame_with_overlay = draw_skeleton(frame_with_overlay, keypoints_history[frame_number])
        
        out.write(frame_with_overlay)
    
    cap.release()
    out.release()

def add_phase_overlay(frame: np.ndarray, phase: str, frame_number: int, total_frames: int) -> np.ndarray:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    
    # フェーズのテキストを大きく表示し、白背景に黒文字で表示
    text = f"Phase: {phase}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 4
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x, text_y = 10, 60
    text_w, text_h = text_size

    # 白背景の矩形を描画
    cv2.rectangle(overlay, (text_x - 5, text_y - text_h - 5), (text_x + text_w + 5, text_y + 5), (255, 255, 255), -1)
    # 黒文字でテキストを描画
    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    
    progress = int(width * frame_number / total_frames)
    cv2.rectangle(overlay, (0, height - 20), (progress, height), PHASE_COLORS.get(phase, (0, 0, 0)), -1)
    cv2.putText(overlay, f"{frame_number}/{total_frames}", (10, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

def analyze_and_visualize_serve(video_path: str, output_path: str, model_name: str = "movenet_thunder"):
    keypoints_history = process_video(video_path, model_name)
    phases = analyze_serve_phases(keypoints_history)
    process_video_with_overlay(video_path, keypoints_history, phases, output_path)
    return keypoints_history, phases

# Confidence score to determine whether a keypoint prediction is reliable.
MIN_CROP_KEYPOINT_SCORE = 0.2

def init_crop_region(image_height, image_width):
    """Defines the default crop region.

    The function provides the initial crop region (pads the full image from both
    sides to make it a square image) when the algorithm cannot reliably determine
    the crop region from the previous frame.
    """
    if image_width > image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2 - image_width / 2) / image_height
        x_min = 0.0
    else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2 - image_height / 2) / image_width

    return {
        'y_min': y_min,
        'x_min': x_min,
        'y_max': y_min + box_height,
        'x_max': x_min + box_width,
        'height': box_height,
        'width': box_width
    }

def torso_visible(keypoints):
    """Checks whether there are enough torso keypoints.

    This function checks whether the model is confident at predicting one of the
    shoulders/hips which is required to determine a good crop region.
    """
    return ((keypoints[0, 0, KEYPOINT_DICT['left_hip'], 2] > MIN_CROP_KEYPOINT_SCORE or
             keypoints[0, 0, KEYPOINT_DICT['right_hip'], 2] > MIN_CROP_KEYPOINT_SCORE) and
            (keypoints[0, 0, KEYPOINT_DICT['left_shoulder'], 2] > MIN_CROP_KEYPOINT_SCORE or
             keypoints[0, 0, KEYPOINT_DICT['right_shoulder'], 2] > MIN_CROP_KEYPOINT_SCORE))

def determine_torso_and_body_range(keypoints, target_keypoints, center_y, center_x):
    """Calculates the maximum distance from each keypoints to the center location.

    The function returns the maximum distances from the two sets of keypoints:
    full 17 keypoints and 4 torso keypoints. The returned information will be
    used to determine the crop size. See determineCropRegion for more detail.
    """
    torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    max_torso_yrange = 0.0
    max_torso_xrange = 0.0
    for joint in torso_joints:
        dist_y = abs(center_y - target_keypoints[joint][0])
        dist_x = abs(center_x - target_keypoints[joint][1])
        if dist_y > max_torso_yrange:
            max_torso_yrange = dist_y
        if dist_x > max_torso_xrange:
            max_torso_xrange = dist_x

    max_body_yrange = 0.0
    max_body_xrange = 0.0
    for joint in KEYPOINT_DICT.keys():
        if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
            continue
        dist_y = abs(center_y - target_keypoints[joint][0])
        dist_x = abs(center_x - target_keypoints[joint][1])
        if dist_y > max_body_yrange:
            max_body_yrange = dist_y
        if dist_x > max_body_xrange:
            max_body_xrange = dist_x

    return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

def determine_crop_region(keypoints, image_height, image_width):
    """Determines the region to crop the image for the model to run inference on.

    The algorithm uses the detected joints from the previous frame to estimate
    the square region that encloses the full body of the target person and
    centers at the midpoint of two hip joints. The crop size is determined by
    the distances between each joints and the center point.
    When the model is not confident with the four torso joint predictions, the
    function returns a default crop which is the full image padded to square.
    """
    target_keypoints = {}
    for joint in KEYPOINT_DICT.keys():
        target_keypoints[joint] = [
            keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
            keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width
        ]

    if torso_visible(keypoints):
        center_y = (target_keypoints['left_hip'][0] + target_keypoints['right_hip'][0]) / 2
        center_x = (target_keypoints['left_hip'][1] + target_keypoints['right_hip'][1]) / 2

        (max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange) = determine_torso_and_body_range(
            keypoints, target_keypoints, center_y, center_x)

        crop_length_half = np.amax(
            [max_torso_xrange * 1.9, max_torso_yrange * 1.9, max_body_yrange * 1.2, max_body_xrange * 1.2])

        tmp = np.array([center_x, image_width - center_x, center_y, image_height - center_y])
        crop_length_half = np.amin([crop_length_half, np.amax(tmp)])

        crop_corner = [center_y - crop_length_half, center_x - crop_length_half]

        if crop_length_half > max(image_width, image_height) / 2:
            return init_crop_region(image_height, image_width)
        else:
            crop_length = crop_length_half * 2
            return {
                'y_min': crop_corner[0] / image_height,
                'x_min': crop_corner[1] / image_width,
                'y_max': (crop_corner[0] + crop_length) / image_height,
                'x_max': (crop_corner[1] + crop_length) / image_width,
                'height': (crop_corner[0] + crop_length) / image_height - crop_corner[0] / image_height,
                'width': (crop_corner[1] + crop_length) / image_width - crop_corner[1] / image_width
            }
    else:
        return init_crop_region(image_height, image_width)

def crop_and_resize(image, crop_region, crop_size):
    """Crops and resize the image to prepare for the model input."""
    boxes = [[crop_region['y_min'], crop_region['x_min'], crop_region['y_max'], crop_region['x_max']]]
    output_image = tf.image.crop_and_resize(image, box_indices=[0], boxes=boxes, crop_size=crop_size)
    return output_image

def run_inference(movenet, image, crop_region, crop_size):
    """Runs model inference on the cropped region.

    The function runs the model inference on the cropped region and updates the
    model output to the original image coordinate system.
    """
    image_height, image_width, _ = image.shape
    input_image = crop_and_resize(tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)
    # Run model inference.
    keypoints_with_scores = movenet(input_image)
    # Update the coordinates.
    for idx in range(17):
        keypoints_with_scores[0, 0, idx, 0] = (
            crop_region['y_min'] * image_height +
            crop_region['height'] * image_height *
            keypoints_with_scores[0, 0, idx, 0]) / image_height
        keypoints_with_scores[0, 0, idx, 1] = (
            crop_region['x_min'] * image_width +
            crop_region['width'] * image_width *
            keypoints_with_scores[0, 0, idx, 1]) / image_width
    return keypoints_with_scores
