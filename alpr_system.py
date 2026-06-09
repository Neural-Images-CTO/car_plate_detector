#!/usr/bin/env python3
"""
Local Automatic License Plate Recognition (ALPR) using YOLOv8 + EasyOCR.

Supports webcam/camera (integer source), static images, and video files.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import easyocr
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


class PlateTracker:
    """Stabilize OCR across frames using a short vote window (reduces camera flicker)."""

    def __init__(self, window: int = 5):
        self._history: dict[tuple[int, int, int, int], deque[str]] = {}
        self._window = window

    def update(self, box: tuple[int, int, int, int], text: str) -> str:
        key = box
        if key not in self._history:
            self._history[key] = deque(maxlen=self._window)
        if text:
            self._history[key].append(text)
        votes = self._history[key]
        if not votes:
            return ""
        return Counter(votes).most_common(1)[0][0]


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
    languages: list[str],
    use_gpu: bool,
) -> tuple[YOLO, easyocr.Reader]:
    device = "cuda" if use_gpu and gpu_available() else "cpu"
    print(f"Using device: {device}")

    weights_path = resolve_yolo_weights(yolo_weights)
    print("Loading YOLO plate detector (first run may download weights)...")
    yolo = YOLO(weights_path)

    print("Loading EasyOCR reader (first run may download models)...")
    reader = easyocr.Reader(languages, gpu=use_gpu and gpu_available())
    return yolo, reader


def recognize_plate(
    reader: easyocr.Reader,
    crop_bgr,
    min_conf: float,
) -> str:
    """Run EasyOCR on a plate crop and return the best combined string."""
    results = reader.readtext(crop_bgr)
    parts: list[str] = []
    for _bbox, text, conf in results:
        if conf >= min_conf and text.strip():
            parts.append(text.strip())
    return clean_plate_text(" ".join(parts))


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
    reader: easyocr.Reader,
    conf: float,
    ocr_conf: float,
    tracker: PlateTracker | None,
) -> list[tuple[tuple[int, int, int, int], str]]:
    """Detect plates, OCR crops, return list of (box, text)."""
    h, w = frame.shape[:2]
    results = yolo.predict(frame, conf=conf, verbose=False)
    detections: list[tuple[tuple[int, int, int, int], str]] = []

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

        text = recognize_plate(reader, crop, ocr_conf)
        plate_box = (x1, y1, x2, y2)
        if tracker is not None:
            text = tracker.update(plate_box, text)
        detections.append((plate_box, text))

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
    detections: list[tuple[tuple[int, int, int, int], str]],
) -> None:
    if plate_log is None:
        return
    plate_log.begin_frame()
    seen_text: set[str] = set()
    for _box, text in detections:
        if text and text not in seen_text:
            plate_log.record(text)
            seen_text.add(text)


def run_stream(
    cap: cv2.VideoCapture,
    yolo: YOLO,
    reader: easyocr.Reader,
    window_name: str,
    conf: float,
    ocr_conf: float,
    skip_frames: int,
    use_tracker: bool,
    plate_log: PlateLog | None,
) -> None:
    tracker = PlateTracker() if use_tracker else None
    frame_idx = 0
    prev_time = time.perf_counter()
    last_detections: list[tuple[tuple[int, int, int, int], str]] = []

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
                for box, text in last_detections:
                    draw_detection(display, box, text)
            else:
                last_detections = process_frame(
                    frame, yolo, reader, conf, ocr_conf, tracker
                )
                display = frame.copy()
                for box, text in last_detections:
                    draw_detection(display, box, text)

            log_frame_detections(plate_log, last_detections)

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
    reader: easyocr.Reader,
    conf: float,
    ocr_conf: float,
    output: str | None,
    plate_log: PlateLog | None,
) -> None:
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"Failed to read image: {path}")

    detections = process_frame(frame, yolo, reader, conf, ocr_conf, tracker=None)
    log_frame_detections(plate_log, detections)
    for box, text in detections:
        draw_detection(frame, box, text)
        if not plate_log:
            print(f"Plate: {text or '(no text)'}")

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
        help="Force CPU for EasyOCR (YOLO still uses Ultralytics default device)",
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
        "--no-track",
        action="store_true",
        help="Disable multi-frame OCR voting (images only benefit little)",
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
        yolo, reader = init_models(args.model, ["en"], use_gpu)
    except Exception as exc:
        print(f"Failed to load models: {exc}", file=sys.stderr)
        return 1

    plate_log: PlateLog | None = None
    if not args.no_csv:
        plate_log = PlateLog(Path(args.csv))

    try:
        if source_type == "image":
            run_image(
                source_value,
                yolo,
                reader,
                args.conf,
                args.ocr_conf,
                args.output,
                plate_log,
            )
        else:
            cap = open_capture(source_type, source_value, args.width, args.height)
            run_stream(
                cap,
                yolo,
                reader,
                "ALPR",
                args.conf,
                args.ocr_conf,
                args.skip_frames,
                use_tracker=not args.no_track,
                plate_log=plate_log,
            )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
