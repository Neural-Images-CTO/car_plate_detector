# Perspective Correction Enhancement

## The Problem

Real traffic cameras see plates at angles. OCR accuracy drops when characters
are skewed or keystoned:

```
Bad (angled):        Good (straight):

   ________          +----------+
  /  ABC123/         |  ABC123  |
 /________/          +----------+
```

The fix is to detect the plate's actual corners and apply a homography transform
to produce a flat, head-on view before passing the crop to OCR.

---

## Why It Can't Be Done with the Current YOLO Output

`cv2.getPerspectiveTransform()` requires **4 exact corner points of the plate**.

YOLO currently returns an **axis-aligned bounding box** — its 4 corners belong
to the enclosing rectangle, not the plate itself:

```
YOLO gives you this:        You need this:
+------------------+           /----------/
|   /ABC123/       |          /  ABC123  /
|                  |         /----------/
+------------------+
```

Using the box corners directly would warp the image incorrectly.

---

## Implementation Options

### Option A — Keypoint/Segmentation YOLO model (recommended)

Use a plate detector model trained to return the **4 corner keypoints** of the
plate in addition to (or instead of) a bounding box. Several models on
Hugging Face support this.

Pipeline:
```
Frame
 └── YOLO (keypoint model)
      └── 4 plate corners (x,y each)
           └── getPerspectiveTransform()
                └── warpPerspective()
                     └── OCR
```

Pros: most reliable, corners are accurate by design.
Cons: requires swapping or adding a second model.

### Option B — Contour detection on the crop

After YOLO crops the bounding box, run edge detection and find the largest
quadrilateral contour inside the crop to extract real plate corners.

```python
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# find largest 4-corner polygon
for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    if len(approx) == 4:
        src_pts = approx.reshape(4, 2).astype(np.float32)
        break

dst_pts = np.float32([[0,0],[w,0],[w,h],[0,h]])
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(crop, M, (w, h))
```

Pros: no model change needed.
Cons: sensitive to lighting, plate color, and cluttered backgrounds. A wrong
contour makes OCR worse than no correction at all. Needs careful tuning.

---

## Where to Insert in the Pipeline

```
current:
  crop → preprocess_plate() → OCR

with perspective correction:
  crop → perspective_correct() → preprocess_plate() → OCR
```

Correction should happen **before** preprocessing so that upscaling and
contrast enhancement work on an already-straightened image.

---

## When to Prioritise This

This enhancement matters most when:
- The camera is mounted at a steep angle (top-down parking cameras, side-angle
  gate cameras).
- Plates appear heavily skewed in captured frames.
- Other improvements (tracking, preprocessing) are already in place and OCR
  errors are still dominated by perspective distortion.

For front-facing cameras where plates appear roughly head-on, the angle is
mild and CLAHE + upscaling already compensates enough. Implement this after
vehicle tracking (ByteTrack) is in place and real-world results show angled
plates as the remaining bottleneck.

---

## Risks

- A failed contour detection (wrong polygon) applies a bad warp and actively
  hurts OCR. Needs a fallback: if no valid 4-corner contour is found, skip
  correction and pass the raw crop.
- Should be gated behind a `--perspective` CLI flag (off by default), same
  pattern as `--preprocess` and `--track`.

---

## Status

- [ ] Decide on Option A (keypoint model) vs Option B (contour detection)
- [ ] Implement `perspective_correct(crop)` with fallback on detection failure
- [ ] Add `--perspective` CLI flag (off by default)
- [ ] Insert correction step before `preprocess_plate()` in `process_frame()`
- [ ] Test on angled plate samples and compare OCR confidence vs without correction
