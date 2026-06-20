# CSRT Motion Tracking + Conditional OCR

## The Problem

On skipped frames, the current pipeline freezes the last known bounding box
and redraws it on the new frame unchanged:

```python
# run_stream() — skipped frame path
for det in last_detections:
    draw_detection(display, det.box, det.text)   # old coordinates, new frame
```

For slow or parked cars this is fine. For fast-moving cars the box drifts
visibly away from the plate until the next detection frame:

```
Frame 1 (detection):   box at x=200  ← correct
Frame 2 (skipped):     box at x=200  ← car moved to x=260, box is wrong
Frame 3 (skipped):     box at x=200  ← car moved to x=320, box is way off
Frame 4 (detection):   box at x=320  ← snaps back
```

---

## The Fix — CSRT Motion Tracker

Replace "freeze old box" with an OpenCV CSRT tracker that **predicts where
the plate moved** on each skipped frame.

```
Detection frame:       YOLO finds plate → initialize CSRT with that box
Skipped frames:        tracker.update(frame) → predicted new box → draw that
Next detection frame:  YOLO refreshes real position → re-initialize tracker
```

### New `run_stream` logic (pseudo-code)

```python
active_trackers: dict[int, tuple[cv2.Tracker, Detection]] = {}

frame_idx += 1

if detection_frame:
    detections = process_frame(frame)          # YOLO + OCR as usual
    active_trackers.clear()
    for i, det in enumerate(detections):
        tr = cv2.TrackerCSRT_create()
        x1, y1, x2, y2 = det.box
        tr.init(frame, (x1, y1, x2 - x1, y2 - y1))   # OpenCV wants (x,y,w,h)
        active_trackers[i] = (tr, det)

else:
    display = frame.copy()
    for i, (tr, det) in list(active_trackers.items()):
        success, (x, y, w, h) = tr.update(frame)
        if success:
            new_box = (int(x), int(y), int(x + w), int(y + h))
            draw_detection(display, new_box, det.text)
        else:
            del active_trackers[i]             # tracker lost, wait for YOLO refresh
```

---

## Conditional OCR

Once motion tracking is in place, there is no need to run OCR on every
detection frame for a plate that is already reading well.

**Current behaviour:** OCR runs every detection frame, regardless of how
confident the last reading was.

**Improved behaviour:** Only re-run OCR when:
- The plate is seen for the first time (no existing reading).
- The tracker was lost and re-acquired by YOLO.
- The previous OCR confidence is below a threshold (e.g. 0.8).

```python
existing = track_ocr_conf.get(track_id)

if existing is None or existing < 0.8 or track_was_lost:
    text, conf = recognize_plate(reader, crop, ocr_conf, preprocess)
    track_ocr_conf[track_id] = conf
else:
    text = track_text[track_id]   # reuse high-confidence reading
```

This reduces the most expensive call in the pipeline (EasyOCR) on frames
where the plate text is already known with high confidence.

---

## CSRT Limitations

| Limitation | Detail |
|---|---|
| Single-object tracker | One `TrackerCSRT_create()` instance per plate; lifecycle must be managed manually (create on detection, delete on loss) |
| Identity swap | If two cars pass close to each other, trackers can swap plates |
| No re-identification | Once a tracker is lost it cannot recover — must wait for YOLO to re-detect |

These limitations are all resolved by **ByteTrack** (see `VEHICLE_TRACKING_ENHANCEMENT.md` Stage 2), which supersedes CSRT.

---

## Where It Fits in the Pipeline

```
current:
  detection frame  → YOLO + OCR → store detections
  skipped frame    → freeze old boxes → redraw

with CSRT:
  detection frame  → YOLO + conditional OCR → init/refresh CSRT trackers
  skipped frame    → tracker.update(frame)  → draw predicted boxes
```

No changes to `process_frame`, `PlateTracker`, `EvidenceStore`, or `PlateLog`.
Only `run_stream` needs updating.

---

## Relationship to Other Enhancements

- **Stage 1** (`VEHICLE_TRACKING_ENHANCEMENT.md`): fix IoU tracker key → do first.
- **This (Stage 1.5)**: CSRT replaces frozen boxes → do second, no new dependency.
- **Stage 2** (`VEHICLE_TRACKING_ENHANCEMENT.md`): ByteTrack via `yolo.track()` → supersedes CSRT entirely when ready.

---

## Status

- [ ] Update `run_stream()` to initialize CSRT trackers on detection frames
- [ ] Update skipped-frame path to call `tracker.update()` instead of freezing boxes
- [ ] Add conditional OCR logic keyed on `ocr_conf` threshold
- [ ] Add `--ocr-refresh-conf` CLI flag (default: 0.8) to control the threshold
