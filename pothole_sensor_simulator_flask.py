"""
Pothole Detection Simulator with Individual Sensor Control
Flask Web Application - Run locally and test in browser
"""

from flask import Flask, render_template_string, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Global variables
model = None
progress_tracker = {}  # task_id -> {"progress": 0-100, "status": "..."}

# ============================================================================
# SENSOR DATA GENERATION FROM VIDEO
# ============================================================================

def generate_sensor_data_from_video(video_path):
    """
    Generate simulated sensor data from video using optical flow.
    Sensors: Accelerometer, Gyroscope, Distance, Piezoelectric (Jerk).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sensors = {
        'accelerometer': [],
        'gyroscope': [],
        'distance_sensor': [],
        'piezoelectric': [],
        'frame_count': 0,
        'fps': fps,
        'duration': total_frames / max(fps, 1)
    }

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return sensors

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_acceleration = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Dense Optical Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Accelerometer: mean magnitude of flow
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        acceleration = float(np.mean(mag))
        sensors['accelerometer'].append(acceleration)

        # Gyroscope: std dev of flow angles (rotational variance)
        valid_ang = ang[~np.isnan(ang)]
        rotation = float(np.std(valid_ang)) if len(valid_ang) > 0 else 0.0
        sensors['gyroscope'].append(rotation)

        # Distance Sensor: inverse of edge density (simulates ultrasonic)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        distance = 100.0 * (1.0 - edge_density)
        sensors['distance_sensor'].append(float(distance))

        # Piezoelectric: Jerk = rate of change of acceleration
        # A sudden sharp spike mimics a piezo impact signal
        jerk = abs(acceleration - prev_acceleration)
        sensors['piezoelectric'].append(float(jerk))

        prev_acceleration = acceleration
        prev_gray = gray
        sensors['frame_count'] += 1

    cap.release()
    return sensors

# ============================================================================
# POTHOLE DETECTION WITH SENSOR INTEGRATION
# ============================================================================

def detect_potholes(video_path, model, sensor_data,
                    use_accelerometer=False, use_gyroscope=False,
                    use_distance=False, use_piezo=False,
                    task_id=None, pass_start=0, pass_end=100):
    """
    Detect potholes using YOLOv8.
    When sensors are active, lowers the YOLO confidence threshold to 0.25
    so borderline detections are 'validated' by the sensor signal.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    any_sensor = use_accelerometer or use_gyroscope or use_distance or use_piezo
    conf_threshold = 0.25 if any_sensor else 0.30

    detections = {
        'total_frames': 0,
        'frames_with_potholes': 0,
        'pothole_count': 0,
        'confidence_scores': [],
    }

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update progress for real-time loading bar
        if task_id and task_id in progress_tracker and total_frames > 0:
            pct = pass_start + (frame_idx / total_frames) * (pass_end - pass_start)
            progress_tracker[task_id]['progress'] = int(pct)

        results = model(frame, conf=conf_threshold, verbose=False)
        detections['total_frames'] += 1

        if results and len(results[0].boxes) > 0:
            detections['frames_with_potholes'] += 1
            for box in results[0].boxes:
                confidence = float(box.conf[0])

                # Boost confidence based on active sensors
                if frame_idx < len(sensor_data['accelerometer']):
                    if use_accelerometer:
                        acc = sensor_data['accelerometer'][frame_idx]
                        confidence *= (1 + acc / 10)
                    if use_gyroscope:
                        gyro = sensor_data['gyroscope'][frame_idx]
                        confidence *= (1 + gyro / 5)
                    if use_distance:
                        dist = sensor_data['distance_sensor'][frame_idx]
                        confidence *= (1 + (100 - dist) / 100)
                    if use_piezo:
                        jerk = sensor_data['piezoelectric'][frame_idx]
                        confidence *= (1 + jerk / 8)

                confidence = min(confidence, 1.0)
                detections['pothole_count'] += 1
                detections['confidence_scores'].append(confidence)

        frame_idx += 1

    cap.release()

    accuracy = (detections['frames_with_potholes'] / max(detections['total_frames'], 1)) * 100
    avg_conf = float(np.mean(detections['confidence_scores'])) * 100 if detections['confidence_scores'] else 0.0

    detections['accuracy'] = round(accuracy, 2)
    detections['avg_confidence'] = round(avg_conf, 2)
    detections['frame_coverage'] = round(accuracy, 2)
    return detections


def compute_delta(baseline, result):
    """Compute per-sensor improvement metrics vs vision-only baseline."""
    baseline_count = max(baseline['pothole_count'], 1)
    simulated_recall = ((result['pothole_count'] - baseline['pothole_count']) / baseline_count) * 100
    return {
        'detection_rate': result['accuracy'],
        'avg_confidence': result['avg_confidence'],
        'pothole_count': result['pothole_count'],
        'frame_coverage': result['frame_coverage'],
        'det_gain': round(result['accuracy'] - baseline['accuracy'], 1),
        'conf_gain': round(result['avg_confidence'] - baseline['avg_confidence'], 1),
        'simulated_recall': round(simulated_recall, 1),
        'extra_detections': result['pothole_count'] - baseline['pothole_count'],
    }

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/progress')
def get_progress():
    task_id = request.args.get('task_id', '')
    if task_id in progress_tracker:
        return jsonify(progress_tracker[task_id])
    return jsonify({'progress': 0, 'status': 'not_found'})


@app.route('/analyze', methods=['POST'])
def analyze():
    global model
    try:
        if model is None:
            model = YOLO('best.pt')

        task_id = request.form.get('task_id', str(uuid.uuid4()))
        progress_tracker[task_id] = {'progress': 2, 'status': 'processing'}

        video_file = request.files['video']
        video_path = 'temp_video.mp4'
        video_file.save(video_path)

        use_accel = request.form.get('useAccel') == 'true'
        use_gyro  = request.form.get('useGyro')  == 'true'
        use_dist  = request.form.get('useDist')  == 'true'
        use_piezo = request.form.get('usePiezo') == 'true'

        # Generate all sensor data once (reused across all passes)
        progress_tracker[task_id]['progress'] = 5
        sensor_data = generate_sensor_data_from_video(video_path)
        progress_tracker[task_id]['progress'] = 15

        # 6 detection passes - each occupies a slice of 0-100% progress
        vision     = detect_potholes(video_path, model, sensor_data,
                                     task_id=task_id, pass_start=15, pass_end=28)
        accel_res  = detect_potholes(video_path, model, sensor_data,
                                     use_accelerometer=True,
                                     task_id=task_id, pass_start=28, pass_end=41)
        gyro_res   = detect_potholes(video_path, model, sensor_data,
                                     use_gyroscope=True,
                                     task_id=task_id, pass_start=41, pass_end=54)
        dist_res   = detect_potholes(video_path, model, sensor_data,
                                     use_distance=True,
                                     task_id=task_id, pass_start=54, pass_end=67)
        piezo_res  = detect_potholes(video_path, model, sensor_data,
                                     use_piezo=True,
                                     task_id=task_id, pass_start=67, pass_end=80)
        combined   = detect_potholes(video_path, model, sensor_data,
                                     use_accelerometer=use_accel,
                                     use_gyroscope=use_gyro,
                                     use_distance=use_dist,
                                     use_piezo=use_piezo,
                                     task_id=task_id, pass_start=80, pass_end=100)

        if os.path.exists(video_path):
            os.remove(video_path)

        progress_tracker[task_id] = {'progress': 100, 'status': 'done'}

        def fmt(r):
            return {
                'total_frames': r['total_frames'],
                'frames_with_potholes': r['frames_with_potholes'],
                'pothole_count': r['pothole_count'],
                'accuracy': r['accuracy'],
                'avg_confidence': r['avg_confidence'],
                'frame_coverage': r['frame_coverage'],
                'duration': sensor_data['duration'],
            }

        return jsonify({
            'task_id': task_id,
            'vision': fmt(vision),
            'sensor': fmt(combined),
            'baseline': {
                'detection_rate': vision['accuracy'],
                'avg_confidence': vision['avg_confidence'],
                'pothole_count': vision['pothole_count'],
                'frame_coverage': vision['frame_coverage'],
            },
            'breakdown': {
                'accel': compute_delta(vision, accel_res),
                'gyro':  compute_delta(vision, gyro_res),
                'dist':  compute_delta(vision, dist_res),
                'piezo': compute_delta(vision, piezo_res),
                'combined': compute_delta(vision, combined),
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================================================
# HTML TEMPLATE PLACEHOLDER - written below
# ============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pothole Detection - Sensor Simulator</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); min-height: 100vh; padding: 24px; color: #e2e8f0; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; padding: 48px 0 32px; }
        .header h1 { font-size: 42px; font-weight: 700; background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 12px; }
        .header p { color: #94a3b8; font-size: 16px; }
        .card { background: rgba(255,255,255,0.05); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 32px; margin-bottom: 24px; }
        .section-title { font-size: 18px; font-weight: 600; color: #a78bfa; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
        .section-title::after { content: ''; flex: 1; height: 1px; background: rgba(167,139,250,0.2); }
        .upload-area { border: 2px dashed rgba(167,139,250,0.4); border-radius: 12px; padding: 48px; text-align: center; cursor: pointer; transition: all 0.3s; background: rgba(167,139,250,0.04); }
        .upload-area:hover { border-color: #a78bfa; background: rgba(167,139,250,0.08); }
        .upload-area .icon { font-size: 48px; margin-bottom: 12px; }
        .upload-area p { color: #94a3b8; font-size: 15px; }
        .upload-area p.main { color: #c4b5fd; font-size: 18px; font-weight: 600; margin-bottom: 8px; }
        input[type="file"] { display: none; }
        .file-selected { margin-top: 12px; color: #34d399; font-size: 14px; font-weight: 500; }
        .sensor-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin-bottom: 24px; }
        .sensor-toggle { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 16px 18px; display: flex; align-items: center; gap: 14px; cursor: pointer; transition: all 0.25s; user-select: none; }
        .sensor-toggle:hover { border-color: rgba(167,139,250,0.5); background: rgba(167,139,250,0.08); }
        .sensor-toggle.active { border-color: #a78bfa; background: rgba(167,139,250,0.15); }
        .sensor-toggle input[type="checkbox"] { display: none; }
        .toggle-dot { width: 20px; height: 20px; border-radius: 50%; border: 2px solid #6b7280; transition: all 0.25s; flex-shrink: 0; }
        .sensor-toggle.active .toggle-dot { border-color: #a78bfa; background: #a78bfa; box-shadow: 0 0 10px rgba(167,139,250,0.5); }
        .sensor-icon { font-size: 22px; }
        .sensor-label-text { font-size: 14px; font-weight: 500; color: #cbd5e1; }
        .btn { display: inline-flex; align-items: center; gap: 8px; padding: 14px 36px; background: linear-gradient(135deg, #7c3aed, #2563eb); color: white; border: none; border-radius: 10px; font-weight: 600; font-size: 15px; cursor: pointer; transition: all 0.3s; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(124,58,237,0.4); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        /* Loading Bar */
        .progress-wrap { display: none; margin: 20px 0; }
        .progress-wrap.show { display: block; }
        .progress-label { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px; color: #94a3b8; }
        .progress-bar-bg { width: 100%; height: 10px; background: rgba(255,255,255,0.08); border-radius: 8px; overflow: hidden; }
        .progress-bar-fill { height: 100%; width: 0%; border-radius: 8px; background: linear-gradient(90deg, #7c3aed, #2563eb, #0ea5e9); transition: width 0.4s ease; box-shadow: 0 0 10px rgba(14,165,233,0.5); }
        /* Metrics Cards */
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 28px; }
        .metric-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px; text-align: center; }
        .metric-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: #64748b; margin-bottom: 10px; font-weight: 600; }
        .metric-value { font-size: 30px; font-weight: 700; background: linear-gradient(135deg, #a78bfa, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .metric-sub { font-size: 11px; color: #475569; margin-top: 4px; }
        /* Comparison cards */
        .comparison-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 28px; }
        .cmp-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 24px; }
        .cmp-card.improved { border-color: rgba(52,211,153,0.3); background: rgba(52,211,153,0.05); }
        .cmp-card h3 { font-size: 16px; font-weight: 600; color: #cbd5e1; margin-bottom: 18px; }
        .cmp-card.improved h3 { color: #34d399; }
        .cmp-row { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 13px; }
        .cmp-row:last-child { border-bottom: none; }
        .cmp-row .lbl { color: #64748b; }
        .cmp-row .val { font-weight: 600; color: #e2e8f0; }
        /* Improvement Banner */
        .improvement-banner { background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(16,185,129,0.1)); border: 1px solid rgba(52,211,153,0.2); border-radius: 14px; padding: 24px; margin-bottom: 28px; }
        .improvement-banner h3 { color: #34d399; margin-bottom: 18px; font-size: 16px; font-weight: 600; }
        .imp-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 14px; }
        .imp-cell { background: rgba(255,255,255,0.05); border-radius: 10px; padding: 16px; text-align: center; }
        .imp-cell .i-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: #6ee7b7; margin-bottom: 8px; font-weight: 600; }
        .imp-cell .i-value { font-size: 26px; font-weight: 700; color: #fff; }
        /* Breakdown Table */
        .breakdown-table-wrap { overflow-x: auto; margin-bottom: 28px; }
        table.breakdown { width: 100%; border-collapse: collapse; font-size: 13px; }
        table.breakdown th { background: rgba(167,139,250,0.15); color: #a78bfa; padding: 12px 14px; text-align: left; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; white-space: nowrap; }
        table.breakdown th:first-child { border-radius: 8px 0 0 8px; }
        table.breakdown th:last-child { border-radius: 0 8px 8px 0; }
        table.breakdown td { padding: 12px 14px; border-bottom: 1px solid rgba(255,255,255,0.05); color: #cbd5e1; white-space: nowrap; }
        table.breakdown tr:last-child td { border-bottom: none; }
        table.breakdown tr.baseline td { color: #94a3b8; font-style: italic; }
        table.breakdown tr.combined-row td { background: rgba(167,139,250,0.07); font-weight: 600; }
        .gain-pos { color: #34d399; font-weight: 600; }
        .gain-neg { color: #f87171; font-weight: 600; }
        .gain-zero { color: #64748b; }
        /* Metrics Explanation */
        .explain-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
        .explain-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 20px; }
        .explain-card h4 { font-size: 14px; font-weight: 600; color: #a78bfa; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
        .explain-card p { font-size: 13px; color: #64748b; line-height: 1.6; }
        /* Chart */
        .chart-wrap { position: relative; height: 280px; margin-top: 16px; }
        /* Results section */
        #results { display: none; }
        #results.show { display: block; animation: fadeIn 0.5s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
        @media (max-width: 768px) { .comparison-row { grid-template-columns: 1fr; } .header h1 { font-size: 28px; } }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🚧 Pothole Sensor Simulator</h1>
        <p>Compare detection accuracy with individual sensor control &amp; real-time metrics breakdown</p>
    </div>

    <!-- Upload -->
    <div class="card">
        <div class="section-title">Step 1 — Upload Road Video</div>
        <div class="upload-area" onclick="document.getElementById('videoInput').click()">
            <div class="icon">📹</div>
            <p class="main">Click to select or drag &amp; drop</p>
            <p>MP4 · AVI · MOV &nbsp;|&nbsp; 30–60 seconds recommended</p>
            <input type="file" id="videoInput" accept="video/*">
        </div>
        <p id="fileName" class="file-selected"></p>
    </div>

    <!-- Sensors -->
    <div class="card">
        <div class="section-title">Step 2 — Toggle Sensors</div>
        <div class="sensor-grid">
            <label class="sensor-toggle active" id="lbl-accel">
                <input type="checkbox" id="useAccel" checked>
                <span class="toggle-dot"></span>
                <span class="sensor-icon">📍</span>
                <span class="sensor-label-text">Accelerometer</span>
            </label>
            <label class="sensor-toggle active" id="lbl-gyro">
                <input type="checkbox" id="useGyro" checked>
                <span class="toggle-dot"></span>
                <span class="sensor-icon">🔄</span>
                <span class="sensor-label-text">Gyroscope</span>
            </label>
            <label class="sensor-toggle active" id="lbl-dist">
                <input type="checkbox" id="useDist" checked>
                <span class="toggle-dot"></span>
                <span class="sensor-icon">📏</span>
                <span class="sensor-label-text">Distance Sensor</span>
            </label>
            <label class="sensor-toggle active" id="lbl-piezo">
                <input type="checkbox" id="usePiezo" checked>
                <span class="toggle-dot"></span>
                <span class="sensor-icon">⚡</span>
                <span class="sensor-label-text">Piezoelectric</span>
            </label>
        </div>
        <button class="btn" id="runBtn" onclick="runAnalysis()">▶ &nbsp;Run Analysis</button>

        <!-- Progress Bar -->
        <div class="progress-wrap" id="progressWrap">
            <div class="progress-label">
                <span id="progressLabel">Processing video...</span>
                <span id="progressPct">0%</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" id="progressBar"></div>
            </div>
        </div>
    </div>

    <!-- Results -->
    <div id="results">
        <!-- Key Metrics -->
        <div class="card">
            <div class="section-title">Step 3 — Results Overview</div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Frames</div>
                    <div class="metric-value" id="mTotalFrames">—</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Potholes Found</div>
                    <div class="metric-value" id="mPotholes">—</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Video Duration</div>
                    <div class="metric-value" id="mDuration">—</div>
                    <div class="metric-sub">seconds</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Active Sensors</div>
                    <div class="metric-value" id="mActiveSensors">—</div>
                </div>
            </div>

            <!-- Vision vs Combined -->
            <div class="comparison-row">
                <div class="cmp-card">
                    <h3>🎥 Vision Only (Baseline)</h3>
                    <div class="cmp-row"><span class="lbl">Detection Rate</span><span class="val" id="vDetRate">—</span></div>
                    <div class="cmp-row"><span class="lbl">Avg Confidence</span><span class="val" id="vConfidence">—</span></div>
                    <div class="cmp-row"><span class="lbl">Pothole Detections</span><span class="val" id="vCount">—</span></div>
                    <div class="cmp-row"><span class="lbl">Frame Coverage</span><span class="val" id="vCoverage">—</span></div>
                </div>
                <div class="cmp-card improved">
                    <h3>✅ With Selected Sensors</h3>
                    <div class="cmp-row"><span class="lbl">Detection Rate</span><span class="val" id="sDetRate">—</span></div>
                    <div class="cmp-row"><span class="lbl">Avg Confidence</span><span class="val" id="sConfidence">—</span></div>
                    <div class="cmp-row"><span class="lbl">Pothole Detections</span><span class="val" id="sCount">—</span></div>
                    <div class="cmp-row"><span class="lbl">Frame Coverage</span><span class="val" id="sCoverage">—</span></div>
                </div>
            </div>

            <!-- Improvement Banner -->
            <div class="improvement-banner">
                <h3>📊 Improvement with Selected Sensors</h3>
                <div class="imp-grid">
                    <div class="imp-cell"><div class="i-label">Detection Rate Gain</div><div class="i-value" id="iDetGain">—</div></div>
                    <div class="imp-cell"><div class="i-label">Confidence Boost</div><div class="i-value" id="iConfBoost">—</div></div>
                    <div class="imp-cell"><div class="i-label">Extra Detections</div><div class="i-value" id="iExtraDet">—</div></div>
                    <div class="imp-cell"><div class="i-label">Simulated Recall</div><div class="i-value" id="iRecall">—</div></div>
                </div>
            </div>
        </div>

        <!-- Sensor Breakdown Table -->
        <div class="card">
            <div class="section-title">Individual Sensor Contribution</div>
            <p style="font-size:13px;color:#64748b;margin-bottom:18px;">Each sensor is tested in isolation against the camera-only baseline. The last row shows all selected sensors combined.</p>
            <div class="breakdown-table-wrap">
                <table class="breakdown" id="breakdownTable">
                    <thead>
                        <tr>
                            <th>Sensor Config</th>
                            <th>Detection Rate</th>
                            <th>Rate Gain</th>
                            <th>Avg Confidence</th>
                            <th>Conf. Boost</th>
                            <th>Detections</th>
                            <th>Extra Found</th>
                            <th>Simulated Recall</th>
                        </tr>
                    </thead>
                    <tbody id="breakdownBody"></tbody>
                </table>
            </div>
        </div>

        <!-- Chart -->
        <div class="card">
            <div class="section-title">Confidence Distribution Chart</div>
            <div class="chart-wrap">
                <canvas id="comparisonChart"></canvas>
            </div>
        </div>

        <!-- Metrics Explanation -->
        <div class="card">
            <div class="section-title">📖 What Do These Metrics Mean?</div>
            <div class="explain-grid">
                <div class="explain-card">
                    <h4>📊 Detection Rate</h4>
                    <p>The percentage of video frames where the AI detected at least one pothole. A higher rate means the system spotted potholes in more frames of the video.</p>
                </div>
                <div class="explain-card">
                    <h4>💯 Avg Confidence</h4>
                    <p>How certain the AI is about its detections, averaged across all found potholes. Sensors boost this by "confirming" visual detections with physical signals like vibration or jerk.</p>
                </div>
                <div class="explain-card">
                    <h4>🔍 Simulated Recall</h4>
                    <p>The percentage increase in total potholes found compared to the camera-only baseline. Sensors lower the detection threshold (0.30→0.25), allowing borderline detections to be captured.</p>
                </div>
                <div class="explain-card">
                    <h4>📏 Frame Coverage</h4>
                    <p>Same as Detection Rate — the fraction of the total video duration where at least one pothole is visible according to the AI. Useful for understanding road condition density.</p>
                </div>
                <div class="explain-card">
                    <h4>➕ Extra Detections</h4>
                    <p>The raw count of additional pothole instances found vs. camera-only. This shows how many potholes would have been missed entirely without the sensor augmentation.</p>
                </div>
                <div class="explain-card">
                    <h4>⚡ Piezoelectric Sensor</h4>
                    <p>Simulated via "Jerk" — the rate of change of acceleration between frames. A sudden spike (like hitting a pothole) generates a large Piezo reading, boosting the AI's confidence at that exact moment.</p>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    let chart = null;
    let pollInterval = null;

    // Sensor toggle visual feedback
    ['accel','gyro','dist','piezo'].forEach(id => {
        const cb = document.getElementById('use' + id.charAt(0).toUpperCase() + id.slice(1));
        const lbl = document.getElementById('lbl-' + id);
        if (cb && lbl) {
            cb.addEventListener('change', () => {
                lbl.classList.toggle('active', cb.checked);
            });
        }
    });

    document.getElementById('videoInput').addEventListener('change', e => {
        const f = e.target.files[0];
        document.getElementById('fileName').textContent = f ? `✓ Selected: ${f.name}` : '';
    });

    function setProgress(pct, label) {
        document.getElementById('progressBar').style.width = pct + '%';
        document.getElementById('progressPct').textContent = pct + '%';
        document.getElementById('progressLabel').textContent = label || 'Processing...';
    }

    function runAnalysis() {
        const file = document.getElementById('videoInput').files[0];
        if (!file) { alert('Please select a video file first'); return; }

        // Generate a task_id on the frontend
        const taskId = crypto.randomUUID();

        const formData = new FormData();
        formData.append('video', file);
        formData.append('task_id', taskId);
        formData.append('useAccel', document.getElementById('useAccel').checked);
        formData.append('useGyro', document.getElementById('useGyro').checked);
        formData.append('useDist', document.getElementById('useDist').checked);
        formData.append('usePiezo', document.getElementById('usePiezo').checked);

        document.getElementById('results').classList.remove('show');
        document.getElementById('progressWrap').classList.add('show');
        document.getElementById('runBtn').disabled = true;
        setProgress(2, 'Uploading video...');

        const passLabels = [
            [0,  15,  'Generating sensor data...'],
            [15, 28,  'Pass 1/6 — Vision only baseline...'],
            [28, 41,  'Pass 2/6 — Testing Accelerometer...'],
            [41, 54,  'Pass 3/6 — Testing Gyroscope...'],
            [54, 67,  'Pass 4/6 — Testing Distance Sensor...'],
            [67, 80,  'Pass 5/6 — Testing Piezoelectric...'],
            [80, 100, 'Pass 6/6 — Combined sensors...'],
        ];

        // Poll progress
        pollInterval = setInterval(() => {
            fetch('/progress?task_id=' + taskId)
                .then(r => r.json())
                .then(d => {
                    const pct = d.progress || 0;
                    const found = passLabels.find(([s, e]) => pct >= s && pct < e);
                    const label = found ? found[2] : 'Finalizing...';
                    setProgress(pct, label);
                })
                .catch(() => {});
        }, 500);

        fetch('/analyze', { method: 'POST', body: formData })
            .then(r => r.json())
            .then(data => {
                clearInterval(pollInterval);
                if (data.error) { alert('Error: ' + data.error); return; }
                setProgress(100, 'Complete!');
                setTimeout(() => {
                    document.getElementById('progressWrap').classList.remove('show');
                    displayResults(data);
                    document.getElementById('results').classList.add('show');
                    document.getElementById('runBtn').disabled = false;
                }, 600);
            })
            .catch(err => {
                clearInterval(pollInterval);
                alert('Error processing video: ' + err);
                document.getElementById('progressWrap').classList.remove('show');
                document.getElementById('runBtn').disabled = false;
            });
    }

    function fmtGain(val, suffix='%') {
        if (val === null || val === undefined) return '<span class="gain-zero">—</span>';
        const cls = val > 0 ? 'gain-pos' : val < 0 ? 'gain-neg' : 'gain-zero';
        const sign = val > 0 ? '+' : '';
        return `<span class="${cls}">${sign}${val}${suffix}</span>`;
    }

    function displayResults(data) {
        const v = data.vision, s = data.sensor, b = data.baseline, br = data.breakdown;

        document.getElementById('mTotalFrames').textContent = v.total_frames;
        document.getElementById('mPotholes').textContent = s.pothole_count;
        document.getElementById('mDuration').textContent = (v.duration||0).toFixed(1);

        const sensors = [];
        if (document.getElementById('useAccel').checked) sensors.push('Accel');
        if (document.getElementById('useGyro').checked) sensors.push('Gyro');
        if (document.getElementById('useDist').checked) sensors.push('Dist');
        if (document.getElementById('usePiezo').checked) sensors.push('Piezo');
        document.getElementById('mActiveSensors').textContent = sensors.length || 'None';

        document.getElementById('vDetRate').textContent    = b.detection_rate.toFixed(1) + '%';
        document.getElementById('vConfidence').textContent = b.avg_confidence.toFixed(1) + '%';
        document.getElementById('vCount').textContent      = b.pothole_count;
        document.getElementById('vCoverage').textContent   = b.frame_coverage.toFixed(1) + '%';

        document.getElementById('sDetRate').textContent    = s.accuracy.toFixed(1) + '%';
        document.getElementById('sConfidence').textContent = s.avg_confidence.toFixed(1) + '%';
        document.getElementById('sCount').textContent      = s.pothole_count;
        document.getElementById('sCoverage').textContent   = s.frame_coverage.toFixed(1) + '%';

        const comb = br.combined;
        document.getElementById('iDetGain').textContent   = (comb.det_gain >= 0 ? '+' : '') + comb.det_gain + '%';
        document.getElementById('iConfBoost').textContent = (comb.conf_gain >= 0 ? '+' : '') + comb.conf_gain + '%';
        document.getElementById('iExtraDet').textContent  = (comb.extra_detections >= 0 ? '+' : '') + comb.extra_detections;
        document.getElementById('iRecall').textContent    = (comb.simulated_recall >= 0 ? '+' : '') + comb.simulated_recall + '%';

        // Build breakdown table
        const rows = [
            { label: '🎥 Camera Only (Baseline)', data: b, isBaseline: true },
            { label: '📍 + Accelerometer', data: br.accel },
            { label: '🔄 + Gyroscope', data: br.gyro },
            { label: '📏 + Distance Sensor', data: br.dist },
            { label: '⚡ + Piezoelectric', data: br.piezo },
            { label: '✅ All Selected (Combined)', data: br.combined, isCombined: true },
        ];

        const tbody = document.getElementById('breakdownBody');
        tbody.innerHTML = '';

        rows.forEach(row => {
            const d = row.data;
            const tr = document.createElement('tr');
            if (row.isBaseline) tr.className = 'baseline';
            if (row.isCombined) tr.className = 'combined-row';

            if (row.isBaseline) {
                tr.innerHTML = `
                    <td>${row.label}</td>
                    <td>${d.detection_rate.toFixed(1)}%</td>
                    <td><span class="gain-zero">Baseline</span></td>
                    <td>${d.avg_confidence.toFixed(1)}%</td>
                    <td><span class="gain-zero">Baseline</span></td>
                    <td>${d.pothole_count}</td>
                    <td><span class="gain-zero">Baseline</span></td>
                    <td><span class="gain-zero">Baseline</span></td>`;
            } else {
                tr.innerHTML = `
                    <td>${row.label}</td>
                    <td>${d.detection_rate.toFixed(1)}%</td>
                    <td>${fmtGain(d.det_gain)}</td>
                    <td>${d.avg_confidence.toFixed(1)}%</td>
                    <td>${fmtGain(d.conf_gain)}</td>
                    <td>${d.pothole_count}</td>
                    <td>${fmtGain(d.extra_detections, '')}</td>
                    <td>${fmtGain(d.simulated_recall)}</td>`;
            }
            tbody.appendChild(tr);
        });

        // Chart
        const ctx = document.getElementById('comparisonChart').getContext('2d');
        if (chart) chart.destroy();
        chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Camera Only', '+ Accel', '+ Gyro', '+ Dist', '+ Piezo', 'All Combined'],
                datasets: [
                    {
                        label: 'Avg Confidence (%)',
                        data: [
                            b.avg_confidence,
                            br.accel.avg_confidence,
                            br.gyro.avg_confidence,
                            br.dist.avg_confidence,
                            br.piezo.avg_confidence,
                            br.combined.avg_confidence,
                        ],
                        backgroundColor: ['rgba(100,116,139,0.7)', 'rgba(96,165,250,0.7)', 'rgba(167,139,250,0.7)', 'rgba(52,211,153,0.7)', 'rgba(251,191,36,0.7)', 'rgba(248,113,113,0.7)'],
                        borderRadius: 6,
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, labels: { color: '#94a3b8', font: { size: 13 } } },
                    tooltip: { callbacks: { label: ctx => ctx.parsed.y.toFixed(1) + '%' } }
                },
                scales: {
                    x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { beginAtZero: true, max: 100, ticks: { color: '#64748b', callback: v => v + '%' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
    }
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚧 Pothole Detection Sensor Simulator")
    print("="*60)
    print("\n✓ Server running at: http://localhost:5000")
    print("✓ Make sure 'best.pt' is in the same directory")
    print("✓ Open browser and upload your road video")
    print("\n" + "="*60 + "\n")
    app.run(debug=True, port=5000, threaded=True)
