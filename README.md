# テニスサーブ分析システム

## プロジェクト概要
このプロジェクトは、テニスのサーブ動作を分析し、選手のパフォーマンスを定量的に評価するシステムです。ビデオ入力を使用して、サーブ動作中の主要な身体の動きを追跡し、重要なメトリクスを計算します。

## モジュール詳細

### 1. main.py
メインスクリプトで、全体の処理フローを制御します。
- 設定ファイル（config.yaml）の読み込み
- TensorFlow Hubのセットアップ
- ビデオ処理の実行
- メトリクスの計算と検証
- 結果の可視化
- レポートの生成

### 2. video_processor.py
ビデオ処理を担当するモジュールです。
- MoveNetモデルを使用したキーポイント抽出
- ビデオフレームの処理
- キーポイントデータのスムージング（Savitzky-Golayフィルタ使用）

### 3. metrics_calculator.py
サーブのメトリクスを計算するモジュールです。
- 最大膝屈曲角度の計算
- 最大肘屈曲角度の計算
- 最大腰-肩分離角度の計算
- サーブ速度の計算
- メトリクスの検証

### 4. visualizer.py
結果を可視化するモジュールです。
- サーブ軌道の可視化
- 関節角度の時系列プロット

### 5. report_generator.py
分析レポートを生成するモジュールです。
- HTML形式のレポート生成
- メトリクスの表示
- 推奨事項の生成

### 6. utils.py
ユーティリティ関数を含むモジュールです。
- 角度計算関数
- その他のヘルパー関数

## 主要なメトリクス
1. 最大膝屈曲角度
2. 最大肘屈曲角度
3. 最大腰-肩分離角度
4. サーブ速度

## 使用技術
- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- TensorFlow Hub (MoveNet モデル)

## 主要なメトリクスと計算方法

### 1. 最大膝屈曲角度
右腰、右膝、右足首のキーポイントを使用して計算します。

```python
if all(k in keypoints for k in ['right_hip', 'right_knee', 'right_ankle']):
    hip = np.array(keypoints['right_hip'][:2])
    knee = np.array(keypoints['right_knee'][:2])
    ankle = np.array(keypoints['right_ankle'][:2])
    if not np.all(hip == 0) and not np.all(knee == 0) and not np.all(ankle == 0):
        knee_angle = calculate_angle(hip, knee, ankle)
        metrics["max_knee_flexion"] = max(metrics["max_knee_flexion"], 180 - knee_angle)
```

### 2. 最大肘屈曲角度
右肩、右肘、右手首のキーポイントを使用して計算します。

```python
if all(k in keypoints for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
    shoulder = np.array(keypoints['right_shoulder'][:2])
    elbow = np.array(keypoints['right_elbow'][:2])
    wrist = np.array(keypoints['right_wrist'][:2])
    if not np.all(shoulder == 0) and not np.all(elbow == 0) and not np.all(wrist == 0):
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        metrics["max_elbow_flexion"] = max(metrics["max_elbow_flexion"], 180 - elbow_angle)
```

### 3. 最大腰-肩分離角度
左右の腰と肩のキーポイントを使用して計算します。

```python
if all(k in keypoints for k in ['right_hip', 'left_hip', 'right_shoulder', 'left_shoulder']):
    right_hip = np.array(keypoints['right_hip'][:2])
    left_hip = np.array(keypoints['left_hip'][:2])
    right_shoulder = np.array(keypoints['right_shoulder'][:2])
    left_shoulder = np.array(keypoints['left_shoulder'][:2])
    if not np.all(right_hip == 0) and not np.all(left_hip == 0) and not np.all(right_shoulder == 0) and not np.all(left_shoulder == 0):
        hip_line = right_hip - left_hip
        shoulder_line = right_shoulder - left_shoulder
        hip_shoulder_angle = np.abs(np.degrees(np.arctan2(np.cross(hip_line, shoulder_line), np.dot(hip_line, shoulder_line))))
        metrics["max_hip_shoulder_separation"] = max(metrics["max_hip_shoulder_separation"], hip_shoulder_angle)
```

### 4. サーブ速度
右手首の位置を追跡し、フレーム間の移動距離と撮影のフレームレートを使用して速度を計算します。

```python
wrist_positions = [np.array(kp['right_wrist'][:2]) for kp in keypoints_history if 'right_wrist' in kp and not np.all(kp['right_wrist'][:2] == 0)]
velocities = []
for i in range(len(wrist_positions) - 1):
    distance = np.linalg.norm(wrist_positions[i+1] - wrist_positions[i])
    velocity = distance * fps * scale_factor  # m/s
    velocities.append(velocity)

if velocities:
    top_velocities = sorted(velocities)[-int(len(velocities)*0.1):]
    max_velocity = np.mean(top_velocities)
    serve_speed = max_velocity * 3.6 * 1.2  # m/s から km/h に変換し、ラケットの加速を考慮して1.2倍
    metrics["serve_speed"] = serve_speed
```

## 使用方法
1. 必要なライブラリをインストール: `pip install -r requirements.txt`
2. `config.yaml` ファイルで分析するビデオのパスと選手の身長を設定
3. `python main.py` を実行
4. 結果は `focused_serve_analysis_report.html` に保存され、`serve_trajectory.png` と `joint_angles.png` も生成されます

## 今後の改善点
- 各メトリクスの計算方法の見直しによる精度改善
- リアルタイム分析機能の追加
- より詳細なフェーズ分析の実装
