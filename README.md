# Local License Plate Recognition (ALPR)

Detect license plates with a pre-trained YOLOv8 model and read characters with EasyOCR. No training required—works offline after the first model download.

## Quick setup

```bash
# From project root
conda env create -f environment.yml   # or: conda create -n local_alpr python=3.10 -y
conda activate local_alpr

pip install opencv-python ultralytics easyocr

# GPU (recommended on your RTX 4070)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

```bash
conda activate local_alpr
cd /home/haytham/projects/car_plate_detector

# Webcam / USB camera (0 = default, try 1 or 2 for external cameras)
python alpr_system.py --source 0

# Image
python alpr_system.py --source path/to/car.jpg

# Video
python alpr_system.py --source path/to/traffic.mp4

# Save annotated still
python alpr_system.py --source car.jpg --output result.jpg
```

Press **q** in the OpenCV window to exit camera/video mode.

### Plate log (CSV + terminal)

By default, each unique plate is logged to `plates_log.csv` with:


| Column         | Meaning                                 |
| -------------- | --------------------------------------- |
| `plate_number` | OCR text (normalized)                   |
| `first_seen`   | ISO timestamp when first read           |
| `frame_count`  | Number of frames that plate appeared in |


The terminal prints `[plate] first seen: …` for new plates and a summary when you quit.

```bash
# Custom log file
python alpr_system.py --source 0 --csv /path/to/my_plates.csv

# Disable logging
python alpr_system.py --source 0 --no-csv
```

### Camera tuning (recommended)


| Flag                   | Default  | Purpose                                             |
| ---------------------- | -------- | --------------------------------------------------- |
| `--conf`               | 0.35     | Raise (e.g. 0.5) to reduce false plate boxes        |
| `--ocr-conf`           | 0.4      | Raise if OCR picks up noise                         |
| `--width` / `--height` | 1280×720 | Lower (e.g. 640×480) for higher FPS                 |
| `--skip-frames`        | 1        | Run YOLO every other frame; increase for slower PCs |
| `--no-track`           | off      | Disable multi-frame text voting                     |


Example for a live USB camera:

```bash
python alpr_system.py --source 0 --width 1280 --height 720 --conf 0.4 --skip-frames 2
```

## First run

YOLO and EasyOCR download weights on first launch (about 1–2 minutes). Later runs are fully local.

## How it works

1. **YOLO** (`yasirfaizahmed/license-plate-object-detection`, trained on keremberke data) finds plate bounding boxes.
2. Each crop is passed to **EasyOCR** for text.
3. For video/camera, a short **vote window** stabilizes the same plate text across frames.
4. Results are drawn in green with an FPS overlay.

## Suggested improvements (camera)

Already included in this repo:

- Frame skipping (`--skip-frames`) for real-time FPS
- OCR voting across frames to reduce flicker
- Configurable resolution and confidence thresholds
- FPS overlay

Further ideas you can add later:

- **ROI**: Only scan the lower half of the frame where plates usually appear (parking/gate cameras).
- **Motion trigger**: Run YOLO only when motion is detected to save GPU.
- **Country-specific regex**: Filter OCR to valid plate patterns (e.g. US `ABC1234`, EU formats).
- **Record on read**: Log plate + timestamp to CSV when confidence stays high for N frames.
- **RTSP/IP cameras**: Use `--source rtsp://user:pass@ip/stream` (OpenCV supports many RTSP URLs).
- **Stronger detector**: Swap to `yolov8s` or a custom fine-tuned `.pt` if the nano model misses plates at distance.

## Troubleshooting


| Issue             | Fix                                                                          |
| ----------------- | ---------------------------------------------------------------------------- |
| Camera won't open | Try `--source 1` or `2`; check `v4l2` / permissions on Linux                 |
| Low FPS           | Lower resolution, increase `--skip-frames`, ensure CUDA PyTorch is installed |
| Wrong OCR         | Move camera closer, improve lighting, raise `--conf` and `--ocr-conf`        |
| No plates found   | Lower `--conf` slightly; ensure plate is visible and in focus                |


## License

Uses third-party models (Ultralytics YOLO, EasyOCR, Hugging Face weights). Check their respective licenses for commercial use.

**Note:** The blueprint’s `keremberke/yolov8n-license-plate-localization` repo is no longer publicly available on Hugging Face. This project uses `yasirfaizahmed/license-plate-object-detection` instead (same dataset family, `best.pt` weights).