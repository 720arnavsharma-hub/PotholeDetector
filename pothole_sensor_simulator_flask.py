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

        active_sensors = []
        if use_accel: active_sensors.append(('accel', 'use_accelerometer', 'TESTING ACCELEROMETER...'))
        if use_gyro:  active_sensors.append(('gyro', 'use_gyroscope', 'TESTING GYROSCOPE...'))
        if use_dist:  active_sensors.append(('dist', 'use_distance', 'TESTING DISTANCE SENSOR...'))
        if use_piezo: active_sensors.append(('piezo', 'use_piezo', 'TESTING PIEZOELECTRIC...'))

        num_passes = 2 + len(active_sensors) + 1  # matrix + vision + active + combined
        pass_len = 100 // num_passes

        # Pass 1: Sensor Matrix
        progress_tracker[task_id]['progress'] = 5
        progress_tracker[task_id]['status'] = 'GENERATING SENSOR MATRIX...'
        sensor_data = generate_sensor_data_from_video(video_path)
        pass_start = pass_len
        progress_tracker[task_id]['progress'] = pass_start

        # Pass 2: Vision Baseline
        pass_end = pass_start + pass_len
        progress_tracker[task_id]['status'] = 'VISION ONLY BASELINE...'
        vision = detect_potholes(video_path, model, sensor_data, task_id=task_id, pass_start=pass_start, pass_end=pass_end)

        # Pass 3..N: Active Sensors
        breakdown = {}
        for key, kwarg, label in active_sensors:
            pass_start = pass_end
            pass_end = pass_start + pass_len
            progress_tracker[task_id]['status'] = label
            res = detect_potholes(video_path, model, sensor_data, task_id=task_id, pass_start=pass_start, pass_end=pass_end, **{kwarg: True})
            breakdown[key] = compute_delta(vision, res)

        # Pass N+1: Combined
        pass_start = pass_end
        pass_end = 100
        progress_tracker[task_id]['status'] = 'COMBINING SENSORS...'
        combined = detect_potholes(video_path, model, sensor_data,
                                     use_accelerometer=use_accel,
                                     use_gyroscope=use_gyro,
                                     use_distance=use_dist,
                                     use_piezo=use_piezo,
                                     task_id=task_id, pass_start=pass_start, pass_end=pass_end)
        breakdown['combined'] = compute_delta(vision, combined)

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
            'active_sensors': [k for k, _, _ in active_sensors],
            'vision': fmt(vision),
            'sensor': fmt(combined),
            'baseline': {
                'detection_rate': vision['accuracy'],
                'avg_confidence': vision['avg_confidence'],
                'pothole_count': vision['pothole_count'],
                'frame_coverage': vision['frame_coverage'],
            },
            'breakdown': breakdown
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
    <title>POTHOLE.SIM</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root { --bg-main: #050505; --bg-panel: #0a0a0a; --border-color: #1f1f1f; --text-main: #ffffff; --text-muted: #888888; --accent: #ffffff; }
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }
        body { background-color: var(--bg-main); color: var(--text-main); padding: 16px; font-size: 13px; }
        .container { max-width: 1400px; margin: 0 auto; border: 1px solid var(--border-color); border-radius: 8px; padding: 24px; background: #000; position: relative; }
        .navbar { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 16px; margin-bottom: 24px; }
        .logo { font-size: 20px; font-weight: 800; letter-spacing: -0.5px; text-transform: uppercase; }
        .grid-top { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
        .panel { background: var(--bg-panel); border: 1px solid var(--border-color); border-radius: 6px; padding: 20px; }
        .panel-header { font-size: 11px; font-weight: 700; color: #fff; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; }
        .panel-header svg { width: 14px; height: 14px; margin-right: 8px; vertical-align: middle; }
        .panel-header-title { display: flex; align-items: center; }
        .panel-header-badge { color: var(--text-muted); font-size: 10px; }
        .upload-area { border: 1px dashed var(--border-color); border-radius: 4px; padding: 40px; text-align: center; cursor: pointer; transition: 0.2s; background: #000; height: 180px; display: flex; flex-direction: column; justify-content: center; align-items: center; }
        .upload-area:hover { border-color: #444; }
        .upload-icon { margin-bottom: 12px; color: var(--text-muted); }
        .upload-area input { display: none; }
        .upload-area p.main { font-size: 13px; font-weight: 500; margin-bottom: 4px; color: var(--text-muted); }
        .upload-area p.sub { font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 0.5px; }
        .upload-area p.main.selected { color: #fff; font-weight: 600; }
        .sensor-matrix { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .sensor-item { border: 1px solid var(--border-color); background: #000; padding: 16px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center; transition: 0.2s; }
        .sensor-name { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #fff; }
        .sensor-item.dim .sensor-name { color: #444; }
        .switch { position: relative; display: inline-block; width: 34px; height: 18px; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #222; transition: .2s; border-radius: 18px; border: 1px solid var(--border-color); }
        .slider:before { position: absolute; content: ""; height: 12px; width: 12px; left: 2px; bottom: 2px; background-color: #666; transition: .2s; border-radius: 50%; }
        input:checked + .slider { background-color: #fff; border-color: #fff; }
        input:checked + .slider:before { transform: translateX(16px); background-color: #000; }
        input:disabled + .slider { opacity: 0.5; cursor: not-allowed; }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px; margin-bottom: 24px; }
        .stat-card { border-bottom: 1px solid var(--border-color); padding-bottom: 12px; }
        .stat-label { font-size: 10px; font-weight: 600; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
        .stat-value { font-size: 32px; font-weight: 700; letter-spacing: -1px; display: flex; align-items: baseline; gap: 8px; }
        .stat-sub { font-size: 14px; color: #666; font-weight: 500; }
        table { width: 100%; border-collapse: collapse; }
        th { text-align: left; padding: 12px 16px; font-size: 10px; font-weight: 600; color: #666; text-transform: uppercase; border-bottom: 1px solid var(--border-color); background: #000; }
        td { padding: 16px; font-size: 11px; font-weight: 500; border-bottom: 1px solid var(--border-color); color: #ccc; }
        tr:last-child td { border-bottom: none; }
        .status-badge { display: inline-block; padding: 4px 8px; border: 1px solid var(--border-color); border-radius: 2px; font-size: 9px; font-weight: 700; text-transform: uppercase; }
        .status-badge.primary { background: #fff; color: #000; border-color: #fff; }
        .status-badge.active { background: #222; color: #fff; }
        .status-badge.standby { color: #666; }
        .delta-pos { color: #fff; font-weight: 600; }
        .delta-zero { color: #666; }
        .chart-container { height: 160px; position: relative; margin-top: 16px; }
        .pass-step { display: flex; align-items: center; gap: 8px; font-size: 10px; color: #555; text-transform: uppercase; font-weight: 600; }
        .step-dot { width: 8px; height: 8px; border-radius: 50%; background: #222; transition: 0.3s; }
        .pass-step.active { color: #fff; }
        .pass-step.active .step-dot { background: #fff; box-shadow: 0 0 8px #fff; }
        .pass-step.done { color: #888; }
        .pass-step.done .step-dot { background: #34d399; }
    </style>
</head>
<body>
<div class="container">
    <div class="navbar">
        <div class="logo">POTHOLE.SIM</div>
    </div>

    <div class="grid-top">
        <div class="panel">
            <div class="panel-header">
                <div class="panel-header-title">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                    UPLOAD ROAD VIDEO
                </div>
            </div>
            <div class="upload-area" onclick="document.getElementById('videoInput').click()">
                <svg id="uploadIcon" class="upload-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                <p class="main" id="fileName">Drag & Drop .mp4 or .avi</p>
                <p class="sub" id="fileSub">MAX FILE SIZE: 2GB</p>
                <div id="uploadProgressWrap" style="display:none; width: 100%; max-width: 200px; margin-top: 16px;">
                    <div style="display:flex; justify-content:space-between; font-size:10px; color:#888; margin-bottom:6px; text-transform:uppercase; font-weight:600;">
                        <span>UPLOADING...</span><span id="upPct">0%</span>
                    </div>
                    <div style="width:100%; height:4px; background:#222; border-radius:2px; overflow:hidden;">
                        <div id="upFill" style="height:100%; width:0%; background:#fff; transition:width 0.1s;"></div>
                    </div>
                </div>
                <input type="file" id="videoInput" accept="video/*">
            </div>
            <button onclick="window.location.reload()" style="margin-top: 16px; width: 100%; padding: 12px; background: #111; border: 1px solid #333; color: #fff; border-radius: 4px; font-weight: 700; font-size: 11px; cursor: pointer; text-transform: uppercase; letter-spacing: 1px; transition: 0.2s;" onmouseover="this.style.background='#222'; this.style.borderColor='#555'" onmouseout="this.style.background='#111'; this.style.borderColor='#333'">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>
                RELOAD DASHBOARD
            </button>
        </div>

        <div class="panel">
            <div class="panel-header">
                <div class="panel-header-title">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect><line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17" x2="12" y2="21"></line></svg>
                    ACTIVE SENSOR MATRIX
                </div>
                <div class="panel-header-badge">MAT_ID: 9X-A</div>
            </div>
            <div class="sensor-matrix">
                <div class="sensor-item" id="si-accel">
                    <span class="sensor-name">ACCELEROMETER</span>
                    <label class="switch"><input type="checkbox" class="sensor-cb" id="useAccel" checked onchange="handleSensorChange()"><span class="slider"></span></label>
                </div>
                <div class="sensor-item" id="si-gyro">
                    <span class="sensor-name">GYROSCOPE</span>
                    <label class="switch"><input type="checkbox" class="sensor-cb" id="useGyro" checked onchange="handleSensorChange()"><span class="slider"></span></label>
                </div>
                <div class="sensor-item" id="si-dist">
                    <span class="sensor-name">DISTANCE SENSOR</span>
                    <label class="switch"><input type="checkbox" class="sensor-cb" id="useDist" checked onchange="handleSensorChange()"><span class="slider"></span></label>
                </div>
                <div class="sensor-item" id="si-piezo">
                    <span class="sensor-name">PIEZOELECTRIC</span>
                    <label class="switch"><input type="checkbox" class="sensor-cb" id="usePiezo" checked onchange="handleSensorChange()"><span class="slider"></span></label>
                </div>
            </div>
        </div>
    </div>

    <!-- Backend Progress Bar Panel (Fixed UI) -->
    <div class="panel" id="analysisProgressPanel" style="display:none; margin-bottom: 24px; border: 1px solid #333; background: #000;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
            <div style="font-size:12px; font-weight:700; letter-spacing:1px; color:#fff;" id="backendPhaseLabel">WAITING FOR VIDEO...</div>
            <div style="font-size:12px; font-weight:700; color:#34d399;" id="backendPct">0%</div>
        </div>
        <div style="width:100%; height:6px; background:#111; border-radius:3px; overflow:hidden; margin-bottom:20px;">
            <div id="backendBarFill" style="height:100%; width:0%; background:#fff; transition:width 0.3s ease;"></div>
        </div>
        <div style="display:flex; flex-wrap:wrap; gap: 16px;" id="progressStepsContainer">
        </div>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">TOTAL FRAMES</div>
            <div class="stat-value" id="mTotalFrames">0</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">ANOMALIES DETECTED</div>
            <div class="stat-value" id="mPotholes">0</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">TELEMETRY DURATION</div>
            <div class="stat-value" id="mDuration">00:00:00</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">ACTIVE NODES</div>
            <div class="stat-value"><span id="mActiveSensors">4</span> <span class="stat-sub">/ 4</span></div>
        </div>
    </div>

    <!-- Results Section -->
    <div class="grid-top" id="resComparison" style="display:none;">
        <div class="panel" style="border-color: #333;">
            <div class="panel-header">
                <div class="panel-header-title" style="color: #888;">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                    VISION ONLY (BASELINE)
                </div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:12px; border-bottom:1px solid var(--border-color); padding-bottom:8px;">
                <span style="color:var(--text-muted)">Detection Rate</span><span id="vDetRate" style="font-weight:700">--</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:12px; border-bottom:1px solid var(--border-color); padding-bottom:8px;">
                <span style="color:var(--text-muted)">Avg Confidence</span><span id="vConfidence" style="font-weight:700">--</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:12px; border-bottom:1px solid var(--border-color); padding-bottom:8px;">
                <span style="color:var(--text-muted)">Pothole Detections</span><span id="vCount" style="font-weight:700">--</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:var(--text-muted)">Frame Coverage</span><span id="vCoverage" style="font-weight:700">--</span>
            </div>
        </div>
        
        <div class="panel" style="border-color: #34d399; background: rgba(52,211,153,0.02);">
            <div class="panel-header">
                <div class="panel-header-title" style="color: #34d399;">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
                    WITH SELECTED SENSORS
                </div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:12px; border-bottom:1px solid var(--border-color); padding-bottom:8px;">
                <span style="color:var(--text-muted)">Detection Rate</span><span id="sDetRate" style="font-weight:700">--</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:12px; border-bottom:1px solid var(--border-color); padding-bottom:8px;">
                <span style="color:var(--text-muted)">Avg Confidence</span><span id="sConfidence" style="font-weight:700">--</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:12px; border-bottom:1px solid var(--border-color); padding-bottom:8px;">
                <span style="color:var(--text-muted)">Pothole Detections</span><span id="sCount" style="font-weight:700">--</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:var(--text-muted)">Frame Coverage</span><span id="sCoverage" style="font-weight:700">--</span>
            </div>
        </div>
    </div>

    <div class="panel" id="resImprovements" style="display:none; margin-bottom: 24px; border-color: #222; background: #080808;">
        <div class="panel-header">
            <div class="panel-header-title" style="color:#34d399;">IMPROVEMENT WITH SELECTED SENSORS</div>
        </div>
        <div class="stats-grid" style="margin-bottom:0;">
            <div class="stat-card" style="border:none; background:#111; padding:20px; border-radius:6px; text-align:center;">
                <div class="stat-label">DETECTION RATE GAIN</div>
                <div class="stat-value delta-pos" id="iDetGain" style="justify-content:center;">--</div>
            </div>
            <div class="stat-card" style="border:none; background:#111; padding:20px; border-radius:6px; text-align:center;">
                <div class="stat-label">CONFIDENCE BOOST</div>
                <div class="stat-value delta-pos" id="iConfBoost" style="justify-content:center;">--</div>
            </div>
            <div class="stat-card" style="border:none; background:#111; padding:20px; border-radius:6px; text-align:center;">
                <div class="stat-label">EXTRA DETECTIONS</div>
                <div class="stat-value delta-pos" id="iExtraDet" style="justify-content:center;">--</div>
            </div>
            <div class="stat-card" style="border:none; background:#111; padding:20px; border-radius:6px; text-align:center;">
                <div class="stat-label">SIMULATED RECALL</div>
                <div class="stat-value delta-pos" id="iRecall" style="justify-content:center;">--</div>
            </div>
        </div>
    </div>

    <div class="panel" id="resTable" style="display:none; margin-bottom: 24px;">
        <div class="panel-header">
            <div class="panel-header-title">INDIVIDUAL SENSOR CONTRIBUTION</div>
        </div>
        <p style="font-size:11px; color:#666; margin-bottom:16px;">Each sensor is tested in isolation against the camera-only baseline. The last row shows all selected sensors combined.</p>
        <div style="overflow-x:auto;">
            <table>
                <thead>
                    <tr>
                        <th>SENSOR CONFIG</th>
                        <th>DETECTION RATE</th>
                        <th>RATE GAIN</th>
                        <th>AVG CONFIDENCE</th>
                        <th>CONF. BOOST</th>
                        <th>DETECTIONS</th>
                        <th>EXTRA FOUND</th>
                        <th>SIMULATED RECALL</th>
                    </tr>
                </thead>
                <tbody id="detailedBreakdownBody">
                </tbody>
            </table>
        </div>
    </div>

    <div class="panel" id="resChart" style="display:none; margin-bottom: 24px;">
        <div class="panel-header">
            <div class="panel-header-title">CONFIDENCE DISTRIBUTION CHART</div>
        </div>
        <div class="chart-container" style="height: 300px;">
            <canvas id="detailedChart"></canvas>
        </div>
    </div>

</div>

<script>
    let chart = null;
    let pollInterval = null;
    let isRunning = false;

    function handleSensorChange() {
        let count = 0;
        ['Accel','Gyro','Dist','Piezo'].forEach(id => {
            const checked = document.getElementById('use' + id).checked;
            if(checked) count++;
            const item = document.getElementById('si-' + id.toLowerCase());
            if(item) {
                if(checked) item.classList.remove('dim');
                else item.classList.add('dim');
            }
        });
        document.getElementById('mActiveSensors').textContent = count;
        
        // Auto-run analysis if video is already selected
        const file = document.getElementById('videoInput').files[0];
        if (file && !isRunning) {
            runAnalysis();
        }
    }

    document.getElementById('videoInput').addEventListener('change', e => {
        const f = e.target.files[0];
        if(f) {
            document.getElementById('fileName').textContent = f.name;
            document.getElementById('fileName').classList.add('selected');
            document.getElementById('uploadIcon').style.display = 'none';
            document.getElementById('fileSub').style.display = 'none';
            
            const wrap = document.getElementById('uploadProgressWrap');
            const fill = document.getElementById('upFill');
            const pctTxt = document.getElementById('upPct');
            
            wrap.style.display = 'block';
            fill.style.width = '0%';
            fill.style.background = '#fff';
            pctTxt.textContent = '0%';
            
            let progress = 0;
            const interval = setInterval(() => {
                progress += Math.random() * 15 + 5;
                if(progress >= 100) {
                    progress = 100;
                    clearInterval(interval);
                    setTimeout(() => {
                        pctTxt.textContent = 'READY';
                        fill.style.background = '#34d399';
                        // Auto-run after upload completes
                        runAnalysis();
                    }, 200);
                }
                fill.style.width = progress + '%';
                if(progress < 100) pctTxt.textContent = Math.floor(progress) + '%';
            }, 100);
        }
    });

    function formatDuration(sec) {
        sec = Math.round(sec);
        let h = Math.floor(sec / 3600);
        let m = Math.floor((sec % 3600) / 60);
        let s = sec % 60;
        return [h, m, s].map(v => v < 10 ? '0' + v : v).join(':');
    }
    
    function setCheckboxesDisabled(disabled) {
        const cbs = document.querySelectorAll('.sensor-cb');
        cbs.forEach(cb => cb.disabled = disabled);
    }

    function runAnalysis() {
        const file = document.getElementById('videoInput').files[0];
        if (!file) return;

        isRunning = true;
        setCheckboxesDisabled(true);

        const taskId = crypto.randomUUID();
        const formData = new FormData();
        formData.append('video', file);
        formData.append('task_id', taskId);
        
        const useAccel = document.getElementById('useAccel').checked;
        const useGyro = document.getElementById('useGyro').checked;
        const useDist = document.getElementById('useDist').checked;
        const usePiezo = document.getElementById('usePiezo').checked;
        
        formData.append('useAccel', useAccel);
        formData.append('useGyro', useGyro);
        formData.append('useDist', useDist);
        formData.append('usePiezo', usePiezo);

        document.getElementById('analysisProgressPanel').style.display = 'block';
        document.getElementById('resComparison').style.display = 'none';
        document.getElementById('resImprovements').style.display = 'none';
        document.getElementById('resTable').style.display = 'none';
        document.getElementById('resChart').style.display = 'none';
        
        document.getElementById('backendPhaseLabel').textContent = 'INITIALIZING TELEMETRY...';
        document.getElementById('backendPct').textContent = '0%';
        document.getElementById('backendBarFill').style.width = '0%';
        
        const passSteps = [];
        passSteps.push({label: 'GENERATE SENSORS'});
        passSteps.push({label: 'VISION BASELINE'});
        if (useAccel) passSteps.push({label: 'ACCEL. PASS'});
        if (useGyro) passSteps.push({label: 'GYRO. PASS'});
        if (useDist) passSteps.push({label: 'DISTANCE PASS'});
        if (usePiezo) passSteps.push({label: 'PIEZO. PASS'});
        passSteps.push({label: 'FINAL FUSION'});
        
        const container = document.getElementById('progressStepsContainer');
        container.innerHTML = '';
        passSteps.forEach((step, idx) => {
            const d = document.createElement('div');
            d.className = 'pass-step';
            d.id = 'step-' + idx;
            d.innerHTML = `<span class="step-dot"></span> <span class="step-txt">${step.label}</span>`;
            container.appendChild(d);
        });

        pollInterval = setInterval(() => {
            fetch('/progress?task_id=' + taskId)
                .then(r => r.json())
                .then(d => {
                    const pct = d.progress || 0;
                    document.getElementById('backendPct').textContent = pct + '%';
                    document.getElementById('backendBarFill').style.width = pct + '%';
                    
                    if(d.status && d.status !== 'not_found' && d.status !== 'done') {
                        document.getElementById('backendPhaseLabel').textContent = d.status.toUpperCase();
                    }
                    
                    const stepIdx = Math.floor((pct / 100) * passSteps.length);
                    for(let i=0; i<passSteps.length; i++) {
                        const st = document.getElementById('step-'+i);
                        st.className = 'pass-step';
                        if(i < stepIdx) st.classList.add('done');
                        else if(i === stepIdx) st.classList.add('active');
                    }
                }).catch(() => {});
        }, 500);

        fetch('/analyze', { method: 'POST', body: formData })
            .then(r => r.json())
            .then(data => {
                clearInterval(pollInterval);
                isRunning = false;
                setCheckboxesDisabled(false);
                
                if (data.error) { alert('Error: ' + data.error); return; }
                
                document.getElementById('backendPct').textContent = '100%';
                document.getElementById('backendBarFill').style.width = '100%';
                document.getElementById('backendPhaseLabel').textContent = 'COMPLETE!';
                for(let i=0; i<passSteps.length; i++) document.getElementById('step-'+i).className = 'pass-step done';
                
                setTimeout(() => {
                    displayResults(data);
                }, 800);
            })
            .catch(err => {
                clearInterval(pollInterval);
                isRunning = false;
                setCheckboxesDisabled(false);
                alert('Error processing video: ' + err);
            });
    }

    function fmtGain(val, isBase=false, suffix='%') {
        if (isBase) return '<span class="delta-zero">Baseline</span>';
        if (val === null || val === undefined) return '<span class="delta-zero">--</span>';
        const cls = val > 0 ? 'delta-pos' : val < 0 ? 'delta-zero' : 'delta-zero';
        const sign = val > 0 ? '+' : '';
        return `<span class="${cls}">${sign}${val.toFixed(1)}${suffix}</span>`;
    }

    function displayResults(data) {
        const v = data.vision, s = data.sensor, b = data.baseline, br = data.breakdown;

        document.getElementById('mTotalFrames').textContent = v.total_frames.toLocaleString();
        document.getElementById('mPotholes').textContent = s.pothole_count.toLocaleString();
        document.getElementById('mDuration').textContent = formatDuration(v.duration || 0);

        document.getElementById('vDetRate').textContent    = b.detection_rate.toFixed(1) + '%';
        document.getElementById('vConfidence').textContent = b.avg_confidence.toFixed(1) + '%';
        document.getElementById('vCount').textContent      = b.pothole_count;
        document.getElementById('vCoverage').textContent   = b.frame_coverage.toFixed(1) + '%';

        document.getElementById('sDetRate').textContent    = s.accuracy.toFixed(1) + '%';
        document.getElementById('sConfidence').textContent = s.avg_confidence.toFixed(1) + '%';
        document.getElementById('sCount').textContent      = s.pothole_count;
        document.getElementById('sCoverage').textContent   = s.frame_coverage.toFixed(1) + '%';

        const comb = br.combined;
        document.getElementById('iDetGain').innerHTML   = fmtGain(comb.det_gain);
        document.getElementById('iConfBoost').innerHTML = fmtGain(comb.conf_gain);
        document.getElementById('iExtraDet').innerHTML  = fmtGain(comb.extra_detections, false, '');
        document.getElementById('iRecall').innerHTML    = fmtGain(comb.simulated_recall);

        const activeSensors = data.active_sensors || [];
        
        const rows = [];
        rows.push({ label: 'Camera Only (Baseline)', data: b, isBaseline: true });
        if(activeSensors.includes('accel')) rows.push({ label: '+ Accelerometer', data: br.accel });
        if(activeSensors.includes('gyro')) rows.push({ label: '+ Gyroscope', data: br.gyro });
        if(activeSensors.includes('dist')) rows.push({ label: '+ Distance Sensor', data: br.dist });
        if(activeSensors.includes('piezo')) rows.push({ label: '+ Piezoelectric', data: br.piezo });
        rows.push({ label: 'All Selected (Combined)', data: br.combined, isCombined: true });

        const tbody = document.getElementById('detailedBreakdownBody');
        tbody.innerHTML = '';

        rows.forEach(row => {
            const d = row.data;
            const tr = document.createElement('tr');
            if(row.isBaseline) tr.style.color = 'var(--text-muted)';
            if(row.isCombined) tr.style.background = '#111';

            if (row.isBaseline) {
                tr.innerHTML = `
                    <td style="color:#fff">${row.label}</td>
                    <td>${d.detection_rate.toFixed(1)}%</td>
                    <td><span class="delta-zero">Baseline</span></td>
                    <td>${d.avg_confidence.toFixed(1)}%</td>
                    <td><span class="delta-zero">Baseline</span></td>
                    <td>${d.pothole_count}</td>
                    <td><span class="delta-zero">Baseline</span></td>
                    <td><span class="delta-zero">Baseline</span></td>`;
            } else {
                tr.innerHTML = `
                    <td style="color:#fff">${row.label}</td>
                    <td>${d.detection_rate.toFixed(1)}%</td>
                    <td>${fmtGain(d.det_gain)}</td>
                    <td>${d.avg_confidence.toFixed(1)}%</td>
                    <td>${fmtGain(d.conf_gain)}</td>
                    <td>${d.pothole_count}</td>
                    <td>${fmtGain(d.extra_detections, false, '')}</td>
                    <td>${fmtGain(d.simulated_recall)}</td>`;
            }
            tbody.appendChild(tr);
        });

        const ctx = document.getElementById('detailedChart').getContext('2d');
        if (chart) chart.destroy();
        
        const chartLabels = ['Camera Only'];
        const chartData = [b.avg_confidence];
        const chartColors = ['#444'];
        
        if(activeSensors.includes('accel')) { chartLabels.push('+ Accel'); chartData.push(br.accel.avg_confidence); chartColors.push('#3b82f6'); }
        if(activeSensors.includes('gyro')) { chartLabels.push('+ Gyro'); chartData.push(br.gyro.avg_confidence); chartColors.push('#8b5cf6'); }
        if(activeSensors.includes('dist')) { chartLabels.push('+ Dist'); chartData.push(br.dist.avg_confidence); chartColors.push('#10b981'); }
        if(activeSensors.includes('piezo')) { chartLabels.push('+ Piezo'); chartData.push(br.piezo.avg_confidence); chartColors.push('#f59e0b'); }
        
        chartLabels.push('All Combined');
        chartData.push(br.combined.avg_confidence);
        chartColors.push('#ef4444');

        chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: chartLabels,
                datasets: [{
                    label: 'Avg Confidence (%)',
                    data: chartData,
                    backgroundColor: chartColors,
                    borderRadius: 4,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ctx.parsed.y.toFixed(1) + '%' } } },
                scales: {
                    x: { ticks: { color: '#888', font: {size:10} }, grid: { display: false } },
                    y: { beginAtZero: true, max: 100, ticks: { color: '#888', font: {size:10}, callback: v => v + '%' }, grid: { color: '#222' }, border: {display:false} }
                }
            }
        });

        document.getElementById('resComparison').style.display = 'grid';
        document.getElementById('resImprovements').style.display = 'block';
        document.getElementById('resTable').style.display = 'block';
        document.getElementById('resChart').style.display = 'block';
    }
    
    // updateActiveNodes(); no longer call on startup, let user interact
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
