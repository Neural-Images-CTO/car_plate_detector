# Vehicle Detection & Tracking Enhancement

## Current Limitation

The current `PlateTracker` uses IoU on **plate bounding boxes** to link detections across frames:

```python
PlateTracker.update(plate_box, text)
```

This assumes: *"If two plate boxes overlap, they are the same car."*

That breaks for moving traffic. Example:

```
Frame 1:   plate box = (300, 200, 380, 230)
Frame 30:  plate box = (700, 260, 780, 290)   ← same car, moved right
```

IoU ≈ 0 → tracker thinks the old car disappeared and a new plate appeared.

---

## Proposed Architecture (Full Production)

Suggested by a reviewer. Pipeline changes from:

```
Frame
 └── plate YOLO detector
      └── plate bbox
           └── OCR
                └── PlateTracker (IoU on plate box)
                     └── plate text
```

To:

```
Frame
 └── vehicle YOLO detector  (YOLOv8 COCO: car, truck, bus)
      └── ByteTrack
           └── vehicle_id + vehicle bbox
                └── crop vehicle region
                     └── plate YOLO detector
                          └── plate bbox
                               └── OCR
                                    └── history attached to vehicle_id
```

### Step-by-step

**Step 1 — Detect vehicles first**
Use a vehicle detector (e.g. YOLOv8 COCO) to find car/truck/bus bounding boxes
before looking for plates.

**Step 2 — Track vehicles with ByteTrack**
Assign a stable `vehicle_id` to each car across frames, even as coordinates shift.
ByteTrack is built into Ultralytics — available via `yolo.track()`.

**Step 3 — Detect plates inside the vehicle crop**
Run the plate detector only on the vehicle region, not the whole frame.
Smaller search area = fewer false positives, faster inference.

**Step 4 — Attach OCR history to vehicle ID**

Before:
```python
plate_history = { "ABC123": 5 }
```

After:
```python
vehicles = {
    1: { "plate_history": ["ABC123", "ABC123", "ABC12B"] }
}
```

The key relationship becomes `vehicle_id → plate text`, not `plate box → text`.

### New `process_frame` shape

```python
def process_frame():
    detect vehicles
    update vehicle tracker (ByteTrack)

    for vehicle in tracked_vehicles:
        crop = frame[vehicle.y1:vehicle.y2, vehicle.x1:vehicle.x2]
        plate_box = plate_detector(crop)
        text, conf = OCR(plate_box)
        vehicle_tracker[vehicle.id].add(text)
```

### Two-model vs one-model

| Option | Description | Trade-off |
|---|---|---|
| **A — One model** | Train/use a single YOLO that detects cars AND plates | Simpler pipeline, harder to find a good pre-trained model |
| **B — Two models** (recommended) | YOLOv8 COCO for vehicles + existing plate detector | More inference calls, but each model is best-in-class |

---

## Recommended Incremental Path

The full two-model pipeline is the right end goal but is a large refactor.
A staged approach gets 80% of the benefit with less risk:

### Stage 1 — Fix tracker key (small change, high impact)

Track on the **vehicle body box** instead of the plate box. The body box is much
larger, so IoU stays high even as the car moves several hundred pixels.
No new model needed.

### Stage 1.5 — Replace "reuse old box" with CSRT motion tracking (small-medium change)

See **`CSRT_MOTION_TRACKING.md`** for full details, pseudo-code, and status checklist.

Summary: replace the frozen-box approach on skipped frames with an OpenCV CSRT
tracker that predicts where each plate moved. Also covers conditional OCR —
skipping the expensive EasyOCR call when the existing reading already has high
confidence. Superseded by Stage 2 (ByteTrack) when ready.

### Stage 2 — Replace CSRT with ByteTrack via `yolo.track()` (medium change)

Replace `yolo.predict()` with `yolo.track()`. Ultralytics ships ByteTrack
built-in — you get a stable `vehicle_id` per car with minimal code change.
This supersedes Stage 1.5: ByteTrack handles multi-object identity, occlusion,
and re-entry better than CSRT, and requires no manual tracker lifecycle.
Still one model.

### Stage 3 — Add vehicle detector as a pre-filter (full refactor)

Add a second YOLO model (YOLOv8n COCO) to find vehicle bounding boxes first,
then run the plate detector only inside each crop. Requires restructuring
`process_frame`, `Detection`, `EvidenceStore`, and `PlateLog`.

---

## Trade-off Summary

| Approach | Effort | Benefit | Best for |
|---|---|---|---|
| Fix tracker key to vehicle box | Small | High | Most use cases |
| CSRT motion tracking on skip frames | Small-medium | High | Fast cars, no new dep |
| ByteTrack via `yolo.track()` | Medium | Higher | Multi-car, moving traffic |
| Full two-model pipeline | Large | Highest | Multi-lane, production |

---

## Known Limitations of ByteTrack

- Vehicle IDs can reset or jump when cars occlude each other or leave and
  re-enter the frame.
- Works well for gate/parking/single-lane cameras; needs more tuning for
  dense highway scenes.

---

## Dependencies to Add (when ready)

ByteTrack is already bundled with Ultralytics — no extra install needed.

If the two-model path is chosen, the vehicle model can be loaded with:
```python
vehicle_yolo = YOLO("yolov8n.pt")   # downloads from Ultralytics on first run
```

Classes to track from COCO: `car` (2), `truck` (7), `bus` (5), `motorcycle` (3).

---

## Status

- [ ] Stage 1:   Fix tracker key to vehicle body box
- [ ] Stage 1.5: Replace "reuse old box" with CSRT motion tracking on skipped frames
- [ ] Stage 1.5: Add conditional OCR (skip re-OCR when confidence is already high)
- [ ] Stage 2:   Replace CSRT with ByteTrack via `yolo.track()`
- [ ] Stage 3:   Add second vehicle detector model + two-stage crop pipeline
