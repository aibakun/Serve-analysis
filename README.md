# テニスサーブ分析システム

## プロジェクト概要
このプロジェクトは、テニスのサーブ動作を分析し、選手のパフォーマンスを定量的に評価するシステムです。ビデオ入力を使用して、サーブ動作中の主要な身体の動き、特に肘と膝の角度変化を追跡し、サーブのフェーズを分類します。

## モジュール詳細

### 1. main.py
メインスクリプトで、全体の処理フローを制御します。
- 設定ファイル（config.yaml）の読み込み
- TensorFlow Hubのセットアップ
- ビデオ処理の実行
- フェーズ分類の実行
- 結果の可視化
- レポートの生成

### 2. video_processor.py
ビデオ処理を担当するモジュールです。
- MoveNetモデルを使用したキーポイント抽出
- ビデオフレームの処理
- フェーズ情報のオーバーレイ

### 3. phase_classifier.py
サーブのフェーズを分類するモジュールです。
- 動的なフェーズ境界の検出
- フェーズのスムージング

### 4. visualizer.py
結果を可視化するモジュールです。
- 関節角度の時系列プロット
- フェーズごとの関節角度の箱ひげ図

### 5. report_generator.py
分析レポートを生成するモジュールです。
- HTML形式のレポート生成
- フェーズごとの統計情報の表示
- 視覚化結果の表示

### 6. utils.py
ユーティリティ関数を含むモジュールです。
- 角度計算関数
- データスムージング関数

## 主要な分析項目
1. サーブフェーズの分類 (Preparation, Backswing, Loading, Forward Swing, Impact, Follow Through)
2. 各フェーズにおける右肘の角度変化
3. 各フェーズにおける右膝の角度変化

## 使用技術
- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- SciPy
- TensorFlow Hub (MoveNet モデル)

## 主要な分析方法

### 1. フェーズ分類
キーポイントデータから動的にフェーズ境界を検出し、初期フェーズ分類を行います。その後、メディアンフィルタを使用してフェーズをスムージングします。

### 2. 関節角度の計算
右肩、右肘、右手首のキーポイントを使用して肘の角度を計算し、右腰、右膝、右足首のキーポイントを使用して膝の角度を計算します。

## 使用方法
1. 必要なライブラリをインストール: `pip install -r requirements.txt`
2. `config.yaml` ファイルで分析するビデオのパスと選手の身長を設定
3. `python main.py` を実行
4. 結果は `focused_serve_analysis_report.html` に保存され、`joint_angles.png` と `phase_angles.png` も生成されます

## 現在の課題と改善点

### フェーズ分析の課題
1. Impact フェーズの検出: 現在の結果では Impact フェーズが検出されない場合がある。
2. Follow Through フェーズの検出: Follow Through フェーズも検出されない場合がある。
3. Loading フェーズの長さ: Loading フェーズが不自然に長く検出される場合がある。
4. Forward Swing フェーズの短さ: Forward Swing フェーズが短すぎる可能性がある。

### 角度分析の課題
1. Backswing フェーズの肘角度: Backswing 中の肘角度が一定値（137.44度）になっている。
2. 角度の変動: 一部のフェーズで角度の標準偏差が大きくなっています。これは、ノイズや外れ値の影響である可能性がある。

### 改善案
1. フェーズ検出アルゴリズムの調整:
   - Impact フェーズを適切に検出するロジックの追加
   - Follow Through フェーズの検出方法の改善
   - Loading と Forward Swing フェーズの境界をより適切に設定するロジックの実装
2. 角度計算の精度向上:
   - 外れ値の検出と除去方法の改善
   - スムージングアルゴリズムの調整
3. データの前処理の強化:
   - キーポイントデータの品質チェックと異常値の除外
4. 可視化の改善:
   - フェーズごとの角度変化をより明確に示すグラフの作成
5. 動的なパラメータ調整:
   - サーブの速度や選手の体格に応じて、フェーズ検出のパラメータを動的に調整する機能の追加
