# Pothole Detection System — Sensor Simulator

A Flask-based research tool that demonstrates how **sensor fusion** can significantly improve the accuracy of a YOLOv8-based pothole detection system. Upload any road video and see exactly how each physical sensor contributes to detection performance — with a real-time progress bar and a detailed per-sensor metrics breakdown table.

---

## How It Works

The system runs **6 detection passes** per uploaded video:

| Pass | Mode | Description |
|------|------|-------------|
| 1 | Vision Only | YOLOv8 alone — the baseline |
| 2 | + Accelerometer | Boosts confidence on high pixel-flow magnitude |
| 3 | + Gyroscope | Boosts confidence on high rotational variance |
| 4 | + Distance Sensor | Boosts confidence on high edge-density changes |
| 5 | + Piezoelectric | Boosts confidence on sharp acceleration jerk spikes |
| 6 | All Selected | Combined effect of all checked sensors |

Each sensor is **simulated from the video itself** using optical flow — no physical hardware required.

---

## Simulated Sensors

### Accelerometer
Calculates the **mean magnitude** of pixel movement (dense optical flow) between frames. High movement = high G-force reading.

### Gyroscope
Measures the **standard deviation of flow angles**, simulating rotational velocity. Camera tilt/swerve = high gyroscope reading.

### Distance Sensor
Uses **Canny edge density** as an inverse-distance proxy. More edges in frame = closer / more complex surface = lower "distance" reading.

### Piezoelectric Sensor
Calculates **Jerk** — the frame-to-frame rate of change of acceleration. A sudden spike (like hitting a pothole) mimics the voltage burst a real piezo sensor would generate.

---

## Metrics Explained

| Metric | Meaning |
|--------|---------|
| **Detection Rate** | % of frames where at least one pothole was found |
| **Avg Confidence** | Mean certainty score of all detections (0–100%) |
| **Simulated Recall** | % increase in potholes found vs camera-only baseline |
| **Frame Coverage** | Same as detection rate — fraction of video with active detections |
| **Extra Detections** | Raw count of additional potholes found thanks to sensors |

> **Note:** When any sensor is active, the YOLO confidence threshold drops from `0.30` to `0.25`. This allows borderline detections (which sensors can *validate*) to be captured — producing the "extra found" potholes.

---

## Included Files

| File | Purpose |
|------|---------|
| `pothole_sensor_simulator_flask.py` | Main Flask application (backend + frontend in one file) |
| `best.pt` | Pre-trained YOLOv8 model weights for pothole detection |
| `temp_video.mp4` | Sample road video for testing |

---

## Requirements

- Python 3.8+
- pip

---

## Setup & Run

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install flask ultralytics opencv-python numpy

# 3. Run the app
python pothole_sensor_simulator_flask.py
```

Then open your browser to: **http://localhost:5000**

---

## Usage

1. **Upload** a road video (MP4, AVI, MOV — 30–60 seconds recommended)
2. **Toggle** sensor checkboxes (Accelerometer, Gyroscope, Distance, Piezoelectric)
3. **Click "Run Analysis"** — watch the real-time progress bar as all 6 passes complete
4. **View Results:**
   - Overview metrics (total frames, potholes found, duration)
   - Vision-only vs. sensor-augmented comparison
   - **Individual sensor breakdown table** showing each sensor's isolated contribution
   - Confidence distribution chart
   - Metrics explanation guide

---

## Architecture

```
pothole_sensor_simulator_flask.py
├── generate_sensor_data_from_video()   # Optical flow → 4 sensor streams
├── detect_potholes()                   # YOLOv8 + sensor confidence boost + progress tracking
├── compute_delta()                     # Per-sensor metric deltas vs baseline
├── GET  /                              # Serves the single-page web UI
├── GET  /progress?task_id=<id>         # Real-time progress polling endpoint
└── POST /analyze                       # Main analysis endpoint (6 passes, returns JSON)
```

---

## Legacy YOLO CLI Usage

You can also run the raw YOLOv8 model directly without the Flask interface:

```bash
yolo detect predict model=best.pt source="path\to\your\image_or_video"
```

---

## Notes

- The app runs with `threaded=True` so the `/progress` polling endpoint can respond concurrently while the main `/analyze` pass is executing.
- All sensor data is generated **once** per upload and reused across all 6 detection passes for efficiency.
- Progress tracker data is stored in memory — suitable for local single-user use.
