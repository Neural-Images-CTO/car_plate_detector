#!/usr/bin/env python3
"""Evaluate / batch-run the ALPR pipeline over registered dataset directories.

Reads a dataset registry (datasets/datasets.yaml by default) and, for each
dataset:
  - if a ground_truth CSV is present -> SCORES predictions per plate and writes
    eval_results_<name>.csv (one row per ground-truth plate: exact-match,
    character accuracy), plus a console summary.
  - otherwise -> BATCH-RUNS, writing annotated images to <output-dir>/<name>/
    and a predictions_<name>.csv with ONE ROW PER DETECTION (so multi-car
    images are fully represented), no scoring.

Multi-plate aware: images may contain several plates. The ground-truth CSV may
list multiple rows per filename (one per true plate); predictions are matched to
true plates greedily by character similarity.

Use --emit-truth-template to generate a ready-to-edit ground_truth CSV
pre-filled with predictions (one row per detection) that you then correct.

Examples:
  python evaluate.py --dataset roboflow_il --ocr-engine fastplate
  python evaluate.py --dataset roboflow_il --ocr-engine fastplate --emit-truth-template
  python evaluate.py --country none --ocr-engine easyocr
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import yaml

from alpr_system import (
    DEFAULT_YOLO_MODEL,
    IMAGE_EXTENSIONS,
    PLATE_FORMATS,
    Detection,
    draw_detection,
    init_models,
    process_frame,
)


def load_config(config_path: Path) -> list[dict]:
    """Return the list of dataset entries from the YAML registry."""
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    datasets = cfg.get("datasets") or []
    if not isinstance(datasets, list):
        raise ValueError("'datasets' must be a list in the config file")
    return datasets


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_ground_truth(csv_path: Path) -> dict[str, list[str]]:
    """Map filename -> list of true plates (multiple rows per filename allowed).

    Columns required: `filename`, `plate_number`. Blank/comment (#) rows and
    rows with an empty plate_number are skipped.
    """
    gt: dict[str, list[str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("filename") or "").strip()
            if not name or name.startswith("#"):
                continue
            plate = (row.get("plate_number") or "").strip()
            if not plate:
                continue
            gt.setdefault(name, []).append(plate)
    return gt


def normalize(text: str) -> str:
    """Strip to uppercase alphanumeric so '123-45-678' == '12345678'."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def char_accuracy(pred: str, gt: str) -> float:
    p, g = normalize(pred), normalize(gt)
    if not g:
        return 1.0 if not p else 0.0
    dist = levenshtein(p, g)
    return max(0.0, 1.0 - dist / max(len(g), len(p)))


def match_plates(
    dets: list[Detection], trues: list[str]
) -> tuple[list[dict], int]:
    """Greedily match predicted plates to true plates by character similarity.

    Returns (rows, false_positives) where `rows` has one entry per true plate
    (with its best-matching prediction, exact flag, and char accuracy) and
    `false_positives` is the number of non-empty predictions left unmatched.
    """
    preds = [d for d in dets if d.text]
    used = [False] * len(preds)
    rows: list[dict] = []

    for truth in trues:
        best_i, best_ca = -1, -1.0
        for i, d in enumerate(preds):
            if used[i]:
                continue
            ca = char_accuracy(d.text, truth)
            if ca > best_ca:
                best_ca, best_i = ca, i
        if best_i >= 0:
            used[best_i] = True
            d = preds[best_i]
            exact = normalize(d.text) == normalize(truth)
            rows.append({
                "predicted": d.text,
                "exact_match": int(exact),
                "char_accuracy": best_ca,
                "ocr_conf": d.ocr_conf,
                "yolo_conf": d.yolo_conf,
            })
        else:
            rows.append({
                "predicted": "",
                "exact_match": 0,
                "char_accuracy": 0.0,
                "ocr_conf": 0.0,
                "yolo_conf": 0.0,
            })

    false_positives = sum(1 for u in used if not u)
    return rows, false_positives


def _resolve(path_str: str) -> Path:
    return Path(path_str).expanduser()


def _rel_name(img_path: Path, root: Path) -> str:
    return (
        img_path.relative_to(root).as_posix()
        if root in img_path.parents
        else img_path.name
    )


def _box_str(box: tuple[int, int, int, int]) -> str:
    return "x".join(str(v) for v in box)


def emit_truth_template(
    entry: dict, yolo, engine, plate_format,
    conf: float, ocr_conf: float, preprocess: bool, output_dir: Path,
) -> None:
    """Write a ground-truth CSV pre-filled with predictions (one row per plate).

    The loader only reads `filename` and `plate_number`; the extra columns are
    hints to help you correct the predictions. Images with no detection get one
    blank-plate row so you can fill them in manually.
    """
    name = entry.get("name", "unnamed")
    path = _resolve(str(entry.get("path", "")))
    images = list_images(path)
    print(f"\n=== Template: {name} ({path}) ===")
    if not images:
        print(f"  No images found in {path} (skipping).")
        return

    fields = ["filename", "plate_number", "detection_index", "box",
              "ocr_conf", "yolo_conf"]
    rows: list[dict] = []
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [skip] unreadable image: {img_path}", file=sys.stderr)
            continue
        dets = process_frame(
            frame, yolo, engine, conf, ocr_conf, tracker=None,
            preprocess=preprocess, plate_format=plate_format,
        )
        rel = _rel_name(img_path, path)
        if not dets:
            rows.append({"filename": rel, "plate_number": "", "detection_index": 0,
                         "box": "", "ocr_conf": "0.000", "yolo_conf": "0.000"})
            continue
        for idx, d in enumerate(dets):
            rows.append({
                "filename": rel,
                "plate_number": d.text,  # prediction -> correct this by hand
                "detection_index": idx,
                "box": _box_str(d.box),
                "ocr_conf": f"{d.ocr_conf:.3f}",
                "yolo_conf": f"{d.yolo_conf:.3f}",
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"ground_truth_template_{name}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {len(rows)} rows -> {out_csv}")
    print("  Edit the 'plate_number' column (fix wrong reads, fill blanks),")
    print("  then point the dataset's ground_truth at this file to score.")


def evaluate_dataset(
    entry: dict,
    yolo,
    engine,
    plate_format,
    conf: float,
    ocr_conf: float,
    preprocess: bool,
    output_dir: Path,
) -> dict:
    name = entry.get("name", "unnamed")
    path = _resolve(str(entry.get("path", "")))
    print(f"\n=== Dataset: {name} ({path}) ===")

    images = list_images(path)
    if not images:
        print(f"  No images found in {path} (skipping). Add images to evaluate.")
        return {"name": name, "images": 0}

    gt: dict[str, list[str]] = {}
    gt_path_str = entry.get("ground_truth")
    if gt_path_str:
        gt_path = _resolve(gt_path_str)
        if gt_path.exists():
            gt = load_ground_truth(gt_path)
        else:
            print(f"  ground_truth not found at {gt_path}; running in batch mode.")
    has_gt = bool(gt)
    mode = "scoring (per plate)" if has_gt else "batch-run"
    print(f"  {len(images)} images | mode: {mode}")

    rows: list[dict] = []
    images_with_detection = 0
    total_true = 0
    exact_hits = 0
    char_acc_sum = 0.0
    false_positives = 0

    out_subdir = output_dir / name
    if not has_gt:
        out_subdir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [skip] unreadable image: {img_path}", file=sys.stderr)
            continue

        dets = process_frame(
            frame, yolo, engine, conf, ocr_conf, tracker=None,
            preprocess=preprocess, plate_format=plate_format,
        )
        if any(d.text for d in dets):
            images_with_detection += 1

        rel = _rel_name(img_path, path)

        if has_gt:
            trues = gt.get(img_path.name, gt.get(rel, []))
            if not trues:
                continue  # no ground truth for this image -> not scored
            matched, fps = match_plates(dets, trues)
            false_positives += fps
            for truth, m in zip(trues, matched):
                total_true += 1
                exact_hits += m["exact_match"]
                char_acc_sum += m["char_accuracy"]
                rows.append({
                    "filename": rel,
                    "ground_truth": truth,
                    "predicted": m["predicted"],
                    "exact_match": m["exact_match"],
                    "char_accuracy": f"{m['char_accuracy']:.3f}",
                    "ocr_conf": f"{m['ocr_conf']:.3f}",
                    "yolo_conf": f"{m['yolo_conf']:.3f}",
                })
        else:
            annotated = frame.copy()
            for d in dets:
                draw_detection(annotated, d.box, d.text)
            cv2.imwrite(str(out_subdir / img_path.name), annotated)
            if not dets:
                rows.append({
                    "filename": rel, "detection_index": 0, "predicted": "",
                    "box": "", "ocr_conf": "0.000", "yolo_conf": "0.000",
                })
            for idx, d in enumerate(dets):
                rows.append({
                    "filename": rel,
                    "detection_index": idx,
                    "predicted": d.text,
                    "box": _box_str(d.box),
                    "ocr_conf": f"{d.ocr_conf:.3f}",
                    "yolo_conf": f"{d.yolo_conf:.3f}",
                })

    if has_gt:
        out_csv = output_dir / f"eval_results_{name}.csv"
        fields = ["filename", "ground_truth", "predicted", "exact_match",
                  "char_accuracy", "ocr_conf", "yolo_conf"]
    else:
        out_csv = output_dir / f"predictions_{name}.csv"
        fields = ["filename", "detection_index", "predicted", "box",
                  "ocr_conf", "yolo_conf"]

    output_dir.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "name": name,
        "images": len(images),
        "detection_rate": images_with_detection / len(images) if images else 0.0,
    }
    if has_gt and total_true:
        summary["exact_match_acc"] = exact_hits / total_true
        summary["char_acc"] = char_acc_sum / total_true
        summary["total_true"] = total_true
        summary["false_positives"] = false_positives

    print(f"  image detection rate: {summary['detection_rate']:.1%}")
    if "exact_match_acc" in summary:
        print(f"  per-plate exact-match: {summary['exact_match_acc']:.1%} "
              f"({exact_hits}/{total_true})")
        print(f"  per-plate char accuracy: {summary['char_acc']:.1%}")
        print(f"  false positives (extra reads): {false_positives}")
    print(f"  results -> {out_csv}")
    if not has_gt:
        print(f"  annotated images -> {out_subdir}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="datasets/datasets.yaml",
                   help="Path to the dataset registry YAML (default: datasets/datasets.yaml)")
    p.add_argument("--dataset", default=None,
                   help="Only evaluate the dataset with this name (default: all)")
    p.add_argument("--model", default=DEFAULT_YOLO_MODEL,
                   help="YOLO weights (default: Hugging Face plate model)")
    p.add_argument("--ocr-engine", choices=["easyocr", "paddle", "fastplate"],
                   default="easyocr", help="OCR backend (default: easyocr)")
    p.add_argument("--country", choices=sorted(PLATE_FORMATS.keys()), default="il",
                   help="Plate-format rules (default: il)")
    p.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    p.add_argument("--ocr-conf", type=float, default=0.4,
                   help="Minimum OCR character confidence")
    p.add_argument("--preprocess", action="store_true",
                   help="Upscale + denoise + CLAHE plate crops before OCR")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU")
    p.add_argument("--output-dir", default="C:/src/DATASETS/eval_output_fastplate",
                   help="Directory for result CSVs and annotated images (default: C:/src/DATASETS/eval_output_fastplate)")
    p.add_argument("--emit-truth-template", action="store_true",
                   help="Write a ground_truth CSV pre-filled with predictions "
                        "(one row per detection) for you to correct, then exit")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 1

    try:
        entries = load_config(config_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.dataset:
        entries = [e for e in entries if e.get("name") == args.dataset]
        if not entries:
            print(f"Error: no dataset named '{args.dataset}' in {config_path}",
                  file=sys.stderr)
            return 1

    if not entries:
        print("No datasets registered. Add entries to the config file.")
        return 0

    plate_format = PLATE_FORMATS[args.country]
    use_gpu = not args.no_gpu

    try:
        yolo, engine = init_models(
            args.model, args.ocr_engine, ["en"], use_gpu,
            allowlist=plate_format.allowlist,
        )
    except Exception as exc:
        print(f"Failed to load models: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)

    if args.emit_truth_template:
        for entry in entries:
            emit_truth_template(
                entry, yolo, engine, plate_format,
                args.conf, args.ocr_conf, args.preprocess, output_dir,
            )
        return 0

    summaries = []
    for entry in entries:
        summaries.append(evaluate_dataset(
            entry, yolo, engine, plate_format,
            args.conf, args.ocr_conf, args.preprocess, output_dir,
        ))

    print("\n=== Overall summary ===")
    for s in summaries:
        if s.get("images", 0) == 0:
            print(f"  {s['name']}: no images")
            continue
        line = f"  {s['name']}: {s['images']} imgs, det {s['detection_rate']:.1%}"
        if "exact_match_acc" in s:
            line += (f", exact {s['exact_match_acc']:.1%}, "
                     f"char {s['char_acc']:.1%} "
                     f"(n={s['total_true']}, fp={s['false_positives']})")
        else:
            line += " (batch-run, no ground truth)"
        print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
