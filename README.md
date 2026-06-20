# Local License Plate Recognition (ALPR)

Detect license plates with a pre-trained YOLOv8 model and read characters with a pluggable OCR engine (EasyOCR by default, PaddleOCR optional). No training required — works offline after the first model download.

## Quick setup

```bash
# From project root
conda env create -f environment.yml   # or: conda create -n local_alpr python=3.10 -y
conda activate local_alpr

pip install opencv-python ultralytics easyocr

# GPU (recommended on your RTX 4070)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Optional: PaddleOCR backend (better accuracy on small/angled plates)
pip install paddleocr paddlepaddle

# Optional: fast-plate-ocr backend (plate-specialized OCR, recommended for accuracy)
pip install "fast-plate-ocr[onnx-gpu]"   # CPU only: "fast-plate-ocr[onnx]"
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

### Evidence capture (images + per-event CSV)

Pass `--save-evidence` to save a plate crop and the full annotated frame for every detection event, plus a `detections_log.csv` with confidence scores and image paths.

```bash
python alpr_system.py --source 0 --save-evidence
```

Images are saved to `captures/` by default:

```
captures/
  ABC123_20260619_220112_0001_crop.jpg    ← plate region only (95% JPEG)
  ABC123_20260619_220112_0001_frame.jpg   ← full frame at that moment
```

`detections_log.csv` columns:

| Column         | Meaning                                        |
| -------------- | ---------------------------------------------- |
| `detection_id` | Sequential event number                        |
| `plate_number` | OCR text (normalized)                          |
| `timestamp`    | ISO timestamp of the detection                 |
| `yolo_conf`    | YOLO bounding-box confidence (0–1)             |
| `ocr_conf`     | Mean OCR character confidence (0–1)            |
| `crop_path`    | Absolute path to the plate crop image          |
| `frame_path`   | Absolute path to the full frame image          |

Additional flags:

```bash
# Custom evidence folder and CSV
python alpr_system.py --source 0 --save-evidence \
  --evidence-dir /data/alpr/caps --detections-csv /data/alpr/events.csv

# Stop saving after 500 events (avoids disk fill on long sessions)
python alpr_system.py --source 0 --save-evidence --max-captures 500
```

### Plate-format validation (`--country`)

Plate-format rules are country-aware and selected with `--country` (default `il`):

- **`il` (Israeli, default):** civilian plates are **numeric only**. This does two things:
  1. Restricts EasyOCR to digits (`allowlist="0123456789"`), eliminating the most common OCR errors (letter/digit confusion like `O`/`0`, `I`/`1`, `B`/`8`).
  2. Validates and formats the reading by length: 8 digits -> `XXX-XX-XXX` (since 2017), 7 -> `XX-XXX-XX` (1980-2017), 5-6 -> legacy. Readings with any other length are rejected, so the `--track` stabilizer keeps waiting for a plausible plate instead of locking onto garbage.
- **`none`:** no character restriction and no validation (accepts any alphanumeric reading). Use this for non-Israeli or mixed plates.

```bash
# Israeli numeric plates (default)
python alpr_system.py --source car.jpg --country il

# Plate-specialized OCR + Israeli validation (best accuracy)
python alpr_system.py --source traffic.mp4 --country il --ocr-engine fastplate --track
```

Adding another country later is a single registry entry in `PLATE_FORMATS` (in [alpr_system.py](alpr_system.py)): provide an `allowlist` (widen to include letters for alphanumeric plates) and a `validate` function for that country's pattern.

### Evaluation / batch-run harness (`evaluate.py`)

Test the pipeline over one or more dataset directories registered in `datasets/datasets.yaml`:

```bash
python evaluate.py                                   # all registered datasets
python evaluate.py --dataset mock --ocr-engine fastplate
python evaluate.py --country none --ocr-conf 0.5
```

Each dataset entry has a `name`, a `path` (image folder), and an optional `ground_truth` CSV:

- **With `ground_truth`** (columns `filename,plate_number`): the harness **scores** predictions and writes `eval_output/eval_results_<name>.csv` with per-image exact-match, character accuracy, plus a console summary (exact-match accuracy, character accuracy, detection rate).
- **Without `ground_truth`**: the harness **batch-runs**, saving annotated images to `eval_output/<name>/` and a `eval_output/predictions_<name>.csv` (no scoring).

Layout:

```
datasets/
  datasets.yaml              ← registry (add more dataset dirs here)
  mock/
    images/                  ← drop test images here (mock starts empty)
    ground_truth.csv         ← filename,plate_number (optional, for scoring)
```

To add a real dataset (e.g. a Roboflow export), drop its images in a new folder and add an entry to `datasets.yaml`; include a `ground_truth.csv` if you want accuracy scores.

### Camera tuning (recommended)


| Flag                   | Default  | Purpose                                             |
| ---------------------- | -------- | --------------------------------------------------- |
| `--conf`               | 0.35     | Raise (e.g. 0.5) to reduce false plate boxes        |
| `--ocr-conf`           | 0.4      | Raise if OCR picks up noise                         |
| `--ocr-engine`         | easyocr  | OCR backend: `easyocr`, `paddle` (requires paddleocr), or `fastplate` (requires fast-plate-ocr) |
| `--country`            | il       | Plate-format rules: `il` (Israeli digits-only) or `none` (no validation) |
| `--width` / `--height` | 1280×720 | Lower (e.g. 640×480) for higher FPS                 |
| `--skip-frames`        | 1        | Run YOLO every other frame; increase for slower PCs |
| `--track`              | off      | Enable IoU-based multi-frame OCR stabilizer (recommended for moving cars) |
| `--preprocess`         | off      | Upscale + denoise + CLAHE contrast on plate crop before OCR; runs OCR twice and picks higher confidence (adds ~2x OCR time per plate) |


Example for a live USB camera:

```bash
python alpr_system.py --source 0 --width 1280 --height 720 --conf 0.4 --skip-frames 2
```

## First run

YOLO and the OCR engine download weights on first launch (about 1–2 minutes). Later runs are fully local.

## How it works

1. **YOLO** (`yasirfaizahmed/license-plate-object-detection`, trained on keremberke data) finds plate bounding boxes.
2. Each crop is passed to the selected **OCR engine** (`--ocr-engine easyocr` by default, `paddle` for PaddleOCR, or `fastplate` for the plate-specialized fast-plate-ocr). All engines return `(text, confidence)` pairs through the same interface, so swapping is a single flag.
3. The reading is run through the **country plate-format validator** (`--country`, default `il`): it normalizes the text and reports whether it matches a plausible plate (for Israel, a 7/8-digit numeric plate). Invalid-format readings are filtered out.
4. Optionally (via `--track`), an **IoU-based stabilizer** matches plate boxes across frames by overlap rather than exact position, then picks the winning text by **confidence-weighted voting** — summing OCR confidence scores per candidate over a short window, so a single high-confidence reading can outweigh multiple shaky ones. Only valid-format readings get a vote. Tracks that disappear for 30+ frames are automatically expired.
5. Results are drawn in green with an FPS overlay.

## Suggested improvements (camera)

Already included in this repo:

- Frame skipping (`--skip-frames`) for real-time FPS
- IoU-based OCR stabilizer across frames (`--track`) to reduce flicker on moving cars
- Configurable resolution and confidence thresholds
- FPS overlay
- Evidence capture (`--save-evidence`): plate crop + full frame images with YOLO and OCR confidence scores per event
- Pluggable OCR engine (`--ocr-engine`): swap between EasyOCR, PaddleOCR, and fast-plate-ocr with a single flag
- Plate-specialized OCR (`--ocr-engine fastplate`): fast-plate-ocr, a model trained specifically on license plates
- Country-aware plate validation (`--country`): digit allowlist + length/format rules (Israeli `il` built in)
- Plate crop preprocessing (`--preprocess`): upscale + bilateral filter + CLAHE before OCR
- Confidence-weighted voting in the stabilizer: high-confidence readings outweigh shaky ones (valid-format only)
- Evaluation/batch-run harness (`evaluate.py`): score accuracy against ground truth, or batch-run over dataset folders

Further ideas you can add later:

- **ROI**: Only scan the lower half of the frame where plates usually appear (parking/gate cameras).
- **Motion trigger**: Run YOLO only when motion is detected to save GPU.
- **More countries**: Add entries to `PLATE_FORMATS` (allowlist + validator) for non-Israeli/alphanumeric plates.
- **Record on read**: Log plate + timestamp to CSV when confidence stays high for N frames.
- **RTSP/IP cameras**: Use `--source rtsp://user:pass@ip/stream` (OpenCV supports many RTSP URLs).
- **Stronger detector**: Swap to `yolov8s` or a custom fine-tuned `.pt` if the nano model misses plates at distance.
- **Fine-tune OCR**: Fine-tune fast-plate-ocr on Israeli plates for near-production accuracy.

## Troubleshooting


| Issue             | Fix                                                                          |
| ----------------- | ---------------------------------------------------------------------------- |
| Camera won't open | Try `--source 1` or `2`; check `v4l2` / permissions on Linux                 |
| Low FPS           | Lower resolution, increase `--skip-frames`, ensure CUDA PyTorch is installed |
| Wrong OCR         | Move camera closer, improve lighting, raise `--conf` and `--ocr-conf`        |
| No plates found   | Lower `--conf` slightly; ensure plate is visible and in focus                |


## License

Uses third-party models and libraries (Ultralytics YOLO, EasyOCR, PaddleOCR, Hugging Face weights). Check their respective licenses for commercial use.

**Note:** The blueprint’s `keremberke/yolov8n-license-plate-localization` repo is no longer publicly available on Hugging Face. This project uses `yasirfaizahmed/license-plate-object-detection` instead (same dataset family, `best.pt` weights).