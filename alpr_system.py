#!/usr/bin/env python3
"""
Local Automatic License Plate Recognition (ALPR) using YOLOv8 + pluggable OCR.

Supports webcam/camera (integer source), static images, and video files.
OCR engine is selectable via --ocr-engine: easyocr (default) or paddle.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Pre-trained plate detector on Hugging Face (downloads best.pt on first run)
DEFAULT_YOLO_MODEL = "yasirfaizahmed/license-plate-object-detection"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def parse_source(source: str) -> tuple[str, int | str]:
    """Return ('camera', index) or ('image'/'video', path)."""
    if source.isdigit():
        return "camera", int(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {source}")

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image", str(path.resolve())
    if suffix in VIDEO_EXTENSIONS:
        return "video", str(path.resolve())

    raise ValueError(
        f"Unsupported source '{source}'. Use a camera index (0, 1, ...), "
        f"or a path ending in {sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)}"
    )


def gpu_available() -> bool:
    return torch.cuda.is_available()


def clean_plate_text(raw: str) -> str:
    """Normalize OCR output for display (alphanumeric + common plate chars)."""
    text = raw.upper().strip()
    text = re.sub(r"[^A-Z0-9\- ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    text: str
    yolo_conf: float
    ocr_conf: float
    crop: object  # numpy ndarray — not hashed/compared


@dataclass
class PlateRecord:
    plate_number: str
    first_seen: datetime
    frame_count: int = 0


class PlateLog:
    """Track plates seen in a session; print events and write CSV on exit."""

    CSV_FIELDS = ("plate_number", "first_seen", "frame_count")

    def __init__(self, csv_path: Path | None):
        self._records: dict[str, PlateRecord] = {}
        self.csv_path = csv_path
        self._seen_this_frame: set[str] = set()

    def begin_frame(self) -> None:
        self._seen_this_frame.clear()

    def record(self, plate: str) -> None:
        if not plate:
            return

        now = datetime.now()
        if plate not in self._records:
            self._records[plate] = PlateRecord(plate_number=plate, first_seen=now)
            print(f"[plate] first seen: {plate} at {now.isoformat(timespec='seconds')}")

        if plate in self._seen_this_frame:
            return

        self._seen_this_frame.add(plate)
        self._records[plate].frame_count += 1
        self._write_csv()

    def _write_csv(self) -> None:
        if self.csv_path is None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            for rec in sorted(self._records.values(), key=lambda r: r.first_seen):
                writer.writerow(
                    {
                        "plate_number": rec.plate_number,
                        "first_seen": rec.first_seen.isoformat(timespec="seconds"),
                        "frame_count": rec.frame_count,
                    }
                )

    def print_summary(self) -> None:
        if not self._records:
            print("No plates logged.")
            return
        print("\n--- Plate log ---")
        for rec in sorted(self._records.values(), key=lambda r: r.first_seen):
            print(
                f"  {rec.plate_number}: first_seen={rec.first_seen.isoformat(timespec='seconds')}, "
                f"frames={rec.frame_count}"
            )
        if self.csv_path is not None:
            print(f"CSV saved to: {self.csv_path.resolve()}")


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over Union for two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class PlateTracker:
    """Stabilize OCR across frames using IoU-based track matching and confidence-weighted voting.

    Boxes are matched to existing tracks by IoU rather than exact coordinates,
    so the vote history accumulates even as a car moves across the frame.
    Each reading is stored with its OCR confidence; the winner is chosen by
    summing confidence scores per candidate text rather than counting occurrences.
    Tracks that have not been updated for `max_age` frames are expired.
    """

    def __init__(self, window: int = 5, iou_threshold: float = 0.4, max_age: int = 30):
        self._history: dict[tuple[int, int, int, int], deque[tuple[str, float]]] = {}
        self._last_seen: dict[tuple[int, int, int, int], int] = {}
        self._window = window
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._frame: int = 0

    def _find_matching_key(
        self, box: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int] | None:
        best_key, best_iou = None, self._iou_threshold
        for key in self._history:
            score = _iou(key, box)
            if score > best_iou:
                best_iou, best_key = score, key
        return best_key

    def _expire_old_tracks(self) -> None:
        stale = [k for k, f in self._last_seen.items() if self._frame - f > self._max_age]
        for k in stale:
            del self._history[k]
            del self._last_seen[k]

    def update(self, box: tuple[int, int, int, int], text: str, conf: float = 0.0) -> str:
        self._frame += 1
        self._expire_old_tracks()

        key = self._find_matching_key(box)
        if key is None:
            key = box
            self._history[key] = deque(maxlen=self._window)

        if text and conf > 0.0:
            self._history[key].append((text, conf))
        self._last_seen[key] = self._frame

        entries = self._history[key]
        if not entries:
            return ""

        scores: dict[str, float] = {}
        for t, c in entries:
            scores[t] = scores.get(t, 0.0) + c
        return max(scores, key=lambda t: scores[t])


class EvidenceStore:
    """Save plate crop + full frame images and append rows to a detections CSV.

    One row is written per unique plate per frame. The summary CSV (PlateLog)
    keeps an aggregate view; this table is the per-event evidence trail.
    """

    DETECTION_CSV_FIELDS = (
        "detection_id",
        "plate_number",
        "timestamp",
        "yolo_conf",
        "ocr_conf",
        "crop_path",
        "frame_path",
    )

    def __init__(self, evidence_dir: Path, detections_csv: Path, max_captures: int = 0):
        self._dir = evidence_dir
        self._csv_path = detections_csv
        self._max_captures = max_captures
        self._detection_id = 0
        self._seen_this_frame: set[str] = set()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        self._init_csv()

    def _init_csv(self) -> None:
        if self._csv_path.exists():
            return
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self._csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.DETECTION_CSV_FIELDS).writeheader()

    def begin_frame(self) -> None:
        self._seen_this_frame.clear()

    def save(
        self,
        det: Detection,
        frame,
    ) -> None:
        if not det.text or det.text in self._seen_this_frame:
            return
        if self._max_captures > 0 and self._detection_id >= self._max_captures:
            return

        self._seen_this_frame.add(det.text)
        self._detection_id += 1
        now = datetime.now()
        ts = now.strftime("%Y%m%d_%H%M%S")
        safe_plate = re.sub(r"[^A-Z0-9]", "_", det.text)
        det_id = self._detection_id

        crop_name = f"{safe_plate}_{ts}_{det_id:04d}_crop.jpg"
        frame_name = f"{safe_plate}_{ts}_{det_id:04d}_frame.jpg"
        crop_path = self._dir / crop_name
        frame_path = self._dir / frame_name

        cv2.imwrite(str(crop_path), det.crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        with self._csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.DETECTION_CSV_FIELDS).writerow(
                {
                    "detection_id": det_id,
                    "plate_number": det.text,
                    "timestamp": now.isoformat(timespec="seconds"),
                    "yolo_conf": f"{det.yolo_conf:.3f}",
                    "ocr_conf": f"{det.ocr_conf:.3f}",
                    "crop_path": str(crop_path.resolve()),
                    "frame_path": str(frame_path.resolve()),
                }
            )
        print(f"[evidence] #{det_id} {det.text} → {crop_name}")


class OcrEngine(Protocol):
    """Common interface for OCR backends.

    Each adapter wraps a specific OCR library and normalises its output to a
    flat list of (text, confidence) pairs. All filtering, cleaning, and
    preprocessing logic lives in `recognize_plate` and is engine-agnostic.
    """

    def read(self, img) -> list[tuple[str, float]]:
        """Return (text, confidence) pairs found in img (any channel count)."""
        ...


class EasyOcrEngine:
    """Adapter for EasyOCR. Imported lazily so PaddleOCR installs are optional."""

    def __init__(self, languages: list[str], gpu: bool) -> None:
        import easyocr  # noqa: PLC0415
        print("Loading EasyOCR reader (first run may download models)...")
        self._reader = easyocr.Reader(languages, gpu=gpu)

    def read(self, img) -> list[tuple[str, float]]:
        return [(text, float(conf)) for _bbox, text, conf in self._reader.readtext(img)]


class PaddleOcrEngine:
    """Adapter for PaddleOCR. Install with: pip install paddleocr paddlepaddle"""

    def __init__(self, gpu: bool) -> None:
        import os  # noqa: PLC0415
        import logging  # noqa: PLC0415
        # Force-disable OneDNN before Paddle initializes — it has a compatibility
        # bug on some Windows PaddlePaddle builds.
        os.environ["FLAGS_use_mkldnn"] = "0"
        from paddleocr import PaddleOCR  # noqa: PLC0415
        import paddle  # noqa: PLC0415
        try:
            paddle.set_flags({"FLAGS_use_mkldnn": False})
        except Exception:
            pass
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        print("Loading PaddleOCR (first run may download models)...")
        device = "gpu" if gpu else "cpu"
        self._ocr = PaddleOCR(lang="en", use_textline_orientation=True, device=device)

    def read(self, img) -> list[tuple[str, float]]:
        try:
            results = self._ocr.predict([img])  # predict() expects a batch (list of images)
        except Exception as e:
            print(f"[PaddleOCR] predict failed: {e}", file=__import__("sys").stderr)
            return []
        if not results:
            return []
        out: list[tuple[str, float]] = []
        for batch in results:
            if not batch:
                continue
            items = batch if isinstance(batch, list) else [batch]
            for item in items:
                try:
                    if isinstance(item, dict):
                        text = item.get("rec_text", "")
                        conf = float(item.get("rec_score", 0.0))
                    elif hasattr(item, "rec_text"):
                        text = item.rec_text
                        conf = float(item.rec_score)
                    else:
                        continue
                    if text:
                        out.append((text, conf))
                except Exception:
                    continue
        return out


def resolve_yolo_weights(model_arg: str) -> str:
    """Local .pt path, Hugging Face repo id, or direct URL -> filesystem path."""
    path = Path(model_arg)
    if path.is_file():
        return str(path.resolve())

    if "/" in model_arg and not model_arg.startswith(("http://", "https://")):
        print(f"Downloading YOLO weights from Hugging Face: {model_arg}")
        return hf_hub_download(repo_id=model_arg, filename="best.pt")

    return model_arg


def init_models(
    yolo_weights: str,
    ocr_engine_name: str,
    languages: list[str],
    use_gpu: bool,
) -> tuple[YOLO, OcrEngine]:
    device = "cuda" if use_gpu and gpu_available() else "cpu"
    print(f"Using device: {device}")

    weights_path = resolve_yolo_weights(yolo_weights)
    print("Loading YOLO plate detector (first run may download weights)...")
    yolo = YOLO(weights_path)

    gpu = use_gpu and gpu_available()
    if ocr_engine_name == "paddle":
        engine: OcrEngine = PaddleOcrEngine(gpu=gpu)
    else:
        engine = EasyOcrEngine(languages=languages, gpu=gpu)

    return yolo, engine


def preprocess_plate(crop_bgr, target_h: int = 100):
    """Upscale (if small), grayscale, denoise, and enhance contrast of a plate crop.

    Returns a single-channel (grayscale) image ready for OCR.
    Upscales only when the crop is shorter than target_h to avoid blowing up
    large crops and slowing EasyOCR unnecessarily.
    Uses CLAHE instead of global equalizeHist to avoid over-brightening
    plates that already have decent contrast.
    """
    h = crop_bgr.shape[0]
    if h < target_h:
        scale = target_h / h
        crop_bgr = cv2.resize(
            crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def recognize_plate(
    engine: OcrEngine,
    crop_bgr,
    min_conf: float,
    preprocess: bool = False,
) -> tuple[str, float]:
    """Run the OCR engine on a plate crop; return (cleaned text, mean confidence).

    When preprocess=True, runs OCR on both the raw crop and a preprocessed
    (upscaled + denoised + contrast-enhanced) version, returning whichever
    yields higher confidence. When False, runs OCR on the raw crop only.
    Engine-agnostic: works with any OcrEngine adapter.
    """
    def _ocr(img) -> tuple[str, float]:
        parts, confs = [], []
        for text, conf in engine.read(img):
            if conf >= min_conf and text.strip():
                parts.append(text.strip())
                confs.append(conf)
        avg = sum(confs) / len(confs) if confs else 0.0
        return clean_plate_text(" ".join(parts)), avg

    text_raw, conf_raw = _ocr(crop_bgr)

    if not preprocess:
        return text_raw, conf_raw

    text_pre, conf_pre = _ocr(preprocess_plate(crop_bgr))
    return (text_pre, conf_pre) if conf_pre >= conf_raw else (text_raw, conf_raw)


def draw_detection(frame, box: tuple[int, int, int, int], label: str) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    display = label or "plate"
    (tw, th), baseline = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ty = max(y1 - 8, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 4, ty + baseline), (0, 255, 0), -1)
    cv2.putText(
        frame,
        display,
        (x1 + 2, ty - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def process_frame(
    frame,
    yolo: YOLO,
    engine: OcrEngine,
    conf: float,
    ocr_conf: float,
    tracker: PlateTracker | None,
    preprocess: bool = False,
) -> list[Detection]:
    """Detect plates, OCR crops, return list of Detection objects."""
    h, w = frame.shape[:2]
    results = yolo.predict(frame, conf=conf, verbose=False)
    detections: list[Detection] = []

    if not results or results[0].boxes is None:
        return detections

    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        yolo_conf = float(box.conf[0].cpu().numpy())
        text, ocr_conf_val = recognize_plate(engine, crop, ocr_conf, preprocess)
        plate_box = (x1, y1, x2, y2)
        if tracker is not None:
            text = tracker.update(plate_box, text, ocr_conf_val)
        detections.append(Detection(
            box=plate_box,
            text=text,
            yolo_conf=yolo_conf,
            ocr_conf=ocr_conf_val,
            crop=crop.copy(),
        ))

    return detections


def open_capture(source_type: str, source_value: int | str, width: int, height: int):
    if source_type == "camera":
        cap = cv2.VideoCapture(source_value)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {source_value}. "
                "Try another index (1, 2) or check permissions."
            )
    else:
        cap = cv2.VideoCapture(source_value)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {source_value}")

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def log_frame_detections(
    plate_log: PlateLog | None,
    evidence: EvidenceStore | None,
    detections: list[Detection],
    frame,
) -> None:
    if plate_log is not None:
        plate_log.begin_frame()
    if evidence is not None:
        evidence.begin_frame()

    seen_text: set[str] = set()
    for det in detections:
        if not det.text or det.text in seen_text:
            continue
        seen_text.add(det.text)
        if plate_log is not None:
            plate_log.record(det.text)
        if evidence is not None:
            evidence.save(det, frame)


def run_stream(
    cap: cv2.VideoCapture,
    yolo: YOLO,
    engine: OcrEngine,
    window_name: str,
    conf: float,
    ocr_conf: float,
    skip_frames: int,
    use_tracker: bool,
    plate_log: PlateLog | None,
    evidence: EvidenceStore | None,
    preprocess: bool = False,
) -> None:
    tracker = PlateTracker() if use_tracker else None
    frame_idx = 0
    prev_time = time.perf_counter()
    last_detections: list[Detection] = []

    print("Press 'q' in the video window to quit.")
    if plate_log is not None and plate_log.csv_path is not None:
        print(f"Logging plates to: {plate_log.csv_path.resolve()}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("End of stream or failed to read frame.")
                break

            frame_idx += 1
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 1:
                display = frame.copy()
                for det in last_detections:
                    draw_detection(display, det.box, det.text)
            else:
                last_detections = process_frame(
                    frame, yolo, engine, conf, ocr_conf, tracker, preprocess
                )
                display = frame.copy()
                for det in last_detections:
                    draw_detection(display, det.box, det.text)

            log_frame_detections(plate_log, evidence, last_detections, frame)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(
                display,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.imshow(window_name, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if plate_log is not None:
            plate_log.print_summary()


def run_image(
    path: str,
    yolo: YOLO,
    engine: OcrEngine,
    conf: float,
    ocr_conf: float,
    output: str | None,
    plate_log: PlateLog | None,
    evidence: EvidenceStore | None,
    preprocess: bool = False,
) -> None:
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"Failed to read image: {path}")

    detections = process_frame(frame, yolo, engine, conf, ocr_conf, tracker=None, preprocess=preprocess)
    log_frame_detections(plate_log, evidence, detections, frame)
    for det in detections:
        draw_detection(frame, det.box, det.text)
        if not plate_log:
            print(f"Plate: {det.text or '(no text)'}")

    if not detections:
        print("No license plates detected.")

    if plate_log is not None:
        plate_log.print_summary()

    cv2.imshow("ALPR Result", frame)
    print("Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if output:
        cv2.imwrite(output, frame)
        print(f"Saved annotated image to {output}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Local ALPR with YOLOv8 plate detection and EasyOCR."
    )
    p.add_argument(
        "--source",
        required=True,
        help="Camera index (0, 1, ...) or path to image/video file",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_YOLO_MODEL,
        help="YOLO weights (default: Hugging Face plate model)",
    )
    p.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    p.add_argument(
        "--ocr-conf",
        type=float,
        default=0.4,
        help="Minimum EasyOCR character confidence",
    )
    p.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU for OCR (YOLO still uses Ultralytics default device)",
    )
    p.add_argument(
        "--ocr-engine",
        choices=["easyocr", "paddle"],
        default="easyocr",
        help="OCR backend to use: easyocr (default) or paddle (requires: pip install paddleocr paddlepaddle)",
    )
    p.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Capture width for camera/video (0 = native)",
    )
    p.add_argument(
        "--height",
        type=int,
        default=720,
        help="Capture height for camera/video (0 = native)",
    )
    p.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Run detection every N+1 frames (1=every other frame) for speed",
    )
    p.add_argument(
        "--track",
        action="store_true",
        help="Enable multi-frame IoU-based OCR stabilizer (useful for moving cars)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Save annotated image to this path (image mode only)",
    )
    p.add_argument(
        "--csv",
        default="plates_log.csv",
        metavar="PATH",
        help="CSV log path (plate_number, first_seen, frame_count). Default: plates_log.csv",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable CSV logging and plate summary",
    )
    p.add_argument(
        "--preprocess",
        action="store_true",
        help=(
            "Preprocess plate crops before OCR: upscale small crops, "
            "denoise with bilateral filter, enhance contrast with CLAHE. "
            "Runs OCR twice (raw + preprocessed) and picks the higher-confidence result. "
            "Improves accuracy on small/blurry plates at the cost of ~2x OCR time per plate."
        ),
    )
    p.add_argument(
        "--save-evidence",
        action="store_true",
        help="Save plate crop + full frame images and a per-event detections CSV",
    )
    p.add_argument(
        "--evidence-dir",
        default="captures",
        metavar="DIR",
        help="Directory for evidence images (default: captures/)",
    )
    p.add_argument(
        "--detections-csv",
        default="detections_log.csv",
        metavar="PATH",
        help="Per-event CSV with confidence and image paths (default: detections_log.csv)",
    )
    p.add_argument(
        "--max-captures",
        type=int,
        default=0,
        metavar="N",
        help="Stop saving evidence after N events (0 = unlimited)",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    use_gpu = not args.no_gpu

    try:
        source_type, source_value = parse_source(args.source)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        yolo, engine = init_models(args.model, args.ocr_engine, ["en"], use_gpu)
    except Exception as exc:
        print(f"Failed to load models: {exc}", file=sys.stderr)
        return 1

    plate_log: PlateLog | None = None
    if not args.no_csv:
        plate_log = PlateLog(Path(args.csv))

    evidence: EvidenceStore | None = None
    if args.save_evidence:
        evidence = EvidenceStore(
            evidence_dir=Path(args.evidence_dir),
            detections_csv=Path(args.detections_csv),
            max_captures=args.max_captures,
        )
        print(f"Evidence capture enabled → {Path(args.evidence_dir).resolve()}")

    try:
        if source_type == "image":
            run_image(
                source_value,
                yolo,
                engine,
                args.conf,
                args.ocr_conf,
                args.output,
                plate_log,
                evidence,
                args.preprocess,
            )
        else:
            cap = open_capture(source_type, source_value, args.width, args.height)
            run_stream(
                cap,
                yolo,
                engine,
                "ALPR",
                args.conf,
                args.ocr_conf,
                args.skip_frames,
                use_tracker=args.track,
                plate_log=plate_log,
                evidence=evidence,
                preprocess=args.preprocess,
            )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
