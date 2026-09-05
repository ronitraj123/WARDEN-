# WARDEN
### Railway Crossing Violation Detection System

WARDEN is a computer-vision pipeline that monitors a railway level crossing, detects when pedestrians or vehicles enter the crossing zone while the boom barrier is closing or closed, and produces a structured, evidence-backed log of each violation. It combines two purpose-built YOLOv8 models, geometry-based barrier-state inference, ROI filtering, object tracking, and an optional plate-reading enrichment step, surfaced through a live monitoring dashboard.

This README documents the system as actually built and evaluated — including what worked, what didn't, and why specific design decisions were made.

---

## Table of contents

- [Motivation](#motivation)
- [System architecture](#system-architecture)
- [Repository structure](#repository-structure)
- [Datasets and annotation](#datasets-and-annotation)
- [Models and evaluation](#models-and-evaluation)
- [Pipeline components](#pipeline-components)
- [Known limitations](#known-limitations)
- [Dashboard](#dashboard)
- [License plate OCR](#license-plate-ocr)
- [Setup and usage](#setup-and-usage)
- [Jetson Nano deployment](#jetson-nano-deployment)
- [Design decisions — the "why"](#design-decisions--the-why)
- [Possible extensions](#possible-extensions)

---

## Motivation

Level crossings without automated enforcement rely entirely on drivers and pedestrians respecting a closing/closed barrier. WARDEN is a low-cost, camera-only system intended to run on edge hardware (Jetson Nano) beside a crossing, automatically identify and log violations with visual evidence, and surface them through a dashboard — without requiring specialized traffic-enforcement hardware.

---

## System architecture

```
                     ┌────────────────────┐
 Camera / Video ───▶ │  Barrier detector   │  (every frame, cheap, single-class)
                     └─────────┬──────────┘
                               │ bounding box
                               ▼
                     ┌────────────────────┐
                     │ Geometric state     │  aspect ratio → open / closing / closed
                     │ inference + temporal│  (empirically calibrated thresholds,
                     │ smoothing           │   5-frame majority-vote window)
                     └─────────┬──────────┘
                               │
                    barrier == closing/closed?
                               │ yes
                               ▼
                     ┌────────────────────┐
                     │  Object detector +  │  car / cycle / person / rickshaw /
                     │  ByteTrack tracking │  two wheeler, with persistent track IDs
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │ Rider/vehicle       │  collapses person+cycle/two-wheeler
                     │ disambiguation      │  overlaps into one entity (IoU-based)
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │  ROI filtering      │  polygon calibrated on native camera
                     │                     │  resolution; overlap-ratio threshold
                     └─────────┬──────────┘
                               │ objects inside crossing zone
                               ▼
                     ┌────────────────────┐
                     │ Track-based event   │  deduplicates per-frame detections into
                     │ deduplication       │  discrete violation events by track ID
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │ Evidentiary logging │  CSV: violation_id, track_id, class,
                     │ + snapshot capture  │  barrier_state, start/end time, duration,
                     └─────────┬──────────┘  confidence, snapshot_path
                               │
                     ┌─────────┴──────────┐
                     ▼                    ▼
           ┌──────────────────┐  ┌──────────────────┐
           │ Streamlit         │  │ Gemini-based      │
           │ dashboard          │  │ plate OCR          │
           │ (analytics + live) │  │ (offline, on-demand,│
           │                    │  │  decoupled from the │
           │                    │  │  edge pipeline)      │
           └──────────────────┘  └──────────────────┘
```

Two separate YOLOv8n models are used rather than one combined model — see [Design decisions](#design-decisions--the-why) for why.

---

## Repository structure

```
warden/
├── config/
│   └── roi_config.py            # calibrated ROI polygon (native camera resolution)
├── datasets/
│   ├── objects/
│   │   ├── object_v1/           # original 302-image dataset (labels only; images gitignored)
│   │   └── objects_v2/          # expanded 683-image dataset after targeted re-annotation
│   └── barrier/                 # single-class barrier localization dataset
├── models/
│   ├── warden_objects_v2/weights/best.pt
│   └── warden_barrier/weights/best.pt
├── scripts/
│   ├── extract_frames.py        # video → frame extraction
│   ├── roi_calibrate.py         # click-to-define ROI polygon on a reference frame
│   ├── roi_filter.py            # polygon-overlap ROI filtering
│   ├── barrier_state.py         # barrier detection + geometric state classification
│   ├── violation_pipeline.py    # combines detection + tracking + ROI + barrier gate
│   ├── violation_logger.py      # track-ID based event deduplication + CSV export
│   ├── test_violation_pipeline.py
│   ├── dashboard.py             # Streamlit analytics + live monitor
│   ├── gemini_ocr.py            # on-demand plate OCR via Gemini vision
│   ├── train_objects.py
│   └── train_barrier.py
├── requirements.txt
└── README.md
```

---

## Datasets and annotation

### Object detection dataset

- **v1**: 302 images, manually annotated in Roboflow. Original class scheme conflated object type, location (`on road` / `on track`), and barrier state into single labels (e.g. `two wheeler on track`) — this was deliberately restructured, since location and barrier-state belong to dedicated pipeline stages (ROI filtering, barrier detector), not the object detector's class space. Final classes: `car`, `cycle`, `person`, `rickshaw`, `two wheeler`.
- **v2**: expanded to 683 images by targeting the two weakest classes identified during evaluation — `person` and `two wheeler` in clustered/distant scenes — using SAM 3 (Meta's segmentation foundation model) for model-assisted annotation on `person`/`car`/`cycle`/`two wheeler`, with `rickshaw` left for a future batch given its already-strong performance.
- Preprocessing: auto-orient, "fit within" resize (not stretch, to preserve aspect ratio), 70/20/10 train/valid/test split, class-balancing augmentation for underrepresented classes.

### Barrier state dataset

- ~122 images across `open` / `closing` / `closed` states, merged into a single `boom barrier` class for detection (state is inferred geometrically post-detection — see below).

---

## Models and evaluation

### Object detector (YOLOv8n, v2)

Final held-out **test set** results (`model.val(split='test')`):

| Class | Precision | Recall | mAP50 | mAP50-95 | Test instances |
|---|---|---|---|---|---|
| car | 100% | 100% | 0.995 | 0.708 | 4 |
| rickshaw | 100% | 96.8% | 0.965 | 0.661 | 31 |
| person | 96.9% | 72.1% | 0.723 | 0.561 | 86 |
| cycle | 88.5% | 58.1% | 0.566 | 0.367 | 93 |
| two wheeler | 74% | 67.3% | 0.65 | 0.404 | 55 |

**Overall: mAP50 = 0.780, mAP50-95 = 0.540** (test set, n=269 instances across 68 images)

Small-sample classes (`car`, n=4) are reported with explicit sample sizes rather than treated as high-confidence claims.

### Barrier detector (YOLOv8n, single-class)

- Precision ≈ 97.5%, recall ≈ 87.5% on validation confusion matrix.
- Geometric state thresholds calibrated empirically from a 54-image labeled sample: `open` ratio ≈ 0.21–0.23, `closing` ratio ≈ 0.25–1.53, `closed` ratio ≈ 5.46–10.93 — a clean, well-separated gap between `closing` and `closed`, the transition that matters most for the safety gate.

### The `person` class — a documented, investigated limitation

Fine-tuning on the original imbalanced dataset caused **catastrophic forgetting** of the pretrained model's general person-detection ability (confirmed via a controlled comparison: fine-tuned model detected people in 2/6 validation frames vs. stock YOLOv8n's 6/6 on the identical frames). Re-annotating and retraining (v1 → v2) improved this substantially, moving recall to a statistically meaningful 72.1% on 86 real test instances (vs. an earlier, near-meaningless 66.7% on only 3 instances).

---

## Pipeline components

### ROI filtering
Polygon calibrated by clicking corners directly on a **native-resolution video frame** (not a resized training image — an earlier version calibrated against a 512×512 training frame and silently mismatched the video's actual resolution). Overlap is computed via mask intersection with a configurable threshold (default 0.3).

### Barrier-state gating
Geometric aspect-ratio classification with a 5-frame majority-vote smoothing window, so a single missed detection or noisy frame doesn't flip the gate. Violation logic is deliberately triggered during **both** `closing` and `closed` states — entering during an active closing sequence is treated as comparable real-world risk to entering when fully closed.

### Rider/vehicle disambiguation
IoU-based matching collapses a `person` detection overlapping a `cycle`/`two wheeler` detection into a single vehicle entity, avoiding double-counting a rider as two separate violating objects.

### Event deduplication (ByteTrack)
Per-frame detections are tracked via YOLO's built-in ByteTrack integration (`model.track(..., persist=True)`), and violation events are aggregated by `track_id` rather than logged per-frame — an earlier per-frame-logging version produced ~915 log lines for what was actually 21 real violation events in the same 40-second clip.

---

## Known limitations

- **Two-wheeler detection in clustered/distant scenes** is measurably weaker (test recall 67.3%) — confirmed both via evaluation metrics and visual inspection of small, closely-grouped riders at a distance.
- **ByteTrack ID reuse**: after brief occlusion, a recycled track ID can occasionally split one real violation into two logged events, or cause snapshot overwrites for reused IDs. Not corrected in this version; accepted given low observed frequency.
- **Snapshot legibility is bounded by source object size** in the original frame — near-field vehicles produce usable evidence crops; far-field/small objects do not, regardless of padding/upscaling applied at save time.
- **Live visual monitoring is local to the installation** (a monitor physically connected to the Jetson via `cv2.imshow`); remote operators access processed violation records through the dashboard, not a live video feed. A networked stream (MJPEG/RTSP) was designed and scoped but not implemented, given it requires hardware not available for testing.
- **Running two full YOLO models simultaneously is not viable on Jetson Nano's compute budget** — this is why the pipeline gates object detection behind the (cheap, single-class) barrier detector, rather than running both on every frame regardless of barrier state.

---

## Dashboard

Streamlit app (`scripts/dashboard.py`) with two views:

- **Dashboard tab**: filterable summary stats, a recent-violations thumbnail gallery, class/timeline charts, the full event log, and an evidence viewer showing each violation's representative snapshot alongside its full metadata.
- **Live Monitor tab**: auto-refreshing (via `st.fragment`, isolated from the rest of the app to avoid unnecessary full-page reruns) view of current barrier state and the most recently logged violations, intended to run on the same network as the edge device.

Runs directly on the Jetson (`streamlit run scripts/dashboard.py --server.headless true --server.address 0.0.0.0`), accessed remotely over the local network by any browser.

---

## License plate OCR

Plate reading is a **deliberately decoupled, offline enrichment step** — it does not run on the Jetson, and is not part of the real-time detection pipeline.

**Investigation summary**: four approaches were evaluated in increasing specialization before settling on the final one:

1. General OCR (EasyOCR) on the full vehicle crop — unreliable, no plate localization.
2. Generic two-stage ALPR (`fast-alpr`) — correct architecture, but confidently produced a wrong plate reading.
3. India-trained plate detector + general OCR — correctly localized the plate, but recognition still failed, with the wrong plate rejected by strict Indian-format validation.
4. India-trained plate detector + a plate-specialized (but not India-fine-tuned) recognizer (`fast-plate-ocr`) — still produced incorrect reads on this camera's oblique, distant viewing angle.

**Final approach**: a general-purpose vision-language model (Gemini), prompted specifically for plate reading with an explicit "return null rather than guess" instruction, proved more robust to this camera's viewing conditions than any of the specialized local models tried. Output is validated against a strict Indian plate-format regex before being accepted — an unreadable or malformed result is stored as such, not silently discarded or guessed.

Run on-demand from the dashboard: select car violations → "Run OCR on selected images" → results are written back into `violation_events.csv` (`plate_number`, `plate_confidence`, `plate_reasoning` columns).

Requires a `GEMINI_API_KEY` environment variable.

---

## Setup and usage

```bash
git clone https://github.com/yourusername/warden.git
cd warden
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**Run the full pipeline against a video file, with live cv2 display:**
```bash
python -m scripts.test_violation_pipeline
```
Outputs an annotated video, `violation_events.csv`, and cropped snapshots in `violation_snapshots/`.

**Launch the dashboard:**
```bash
streamlit run scripts/dashboard.py
```

**Run plate OCR** (requires `GEMINI_API_KEY` set):
```bash
python -m scripts.gemini_ocr
```

**Retrain a model:**
```bash
python -m scripts.train_objects
python -m scripts.train_barrier
```
(Training was performed on Google Colab's free-tier GPU; see script comments for the Colab cell breakdown.)

---

## Jetson Nano deployment

Target hardware: NVIDIA Jetson Nano (JetPack 4.6.4 — the maximum officially supported version for this device). Deployment uses the official `ultralytics/ultralytics:latest-jetson-jetpack4` Docker image to avoid the fragile native PyTorch/Python-version compatibility issues inherent to this platform's Python 3.6 default.

High-level flow: flash JetPack → install Docker + verify `nvidia` runtime → pull the Jetson image → export both models to TensorRT (`format='engine', half=True`) for inference speed → run the pipeline against a live USB/CSI camera with `cv2.imshow` displayed on a directly-connected monitor (via X11 passthrough into the container) → run the dashboard natively (outside Docker, no GPU needed) for network-accessible analytics.

Given the barrier-gated architecture, expected real-world throughput favors the (cheap) barrier check running on every frame with the (heavier) object detector only activating during closing/closed states — this was a deliberate design choice specifically to keep the pipeline viable on Nano's limited compute budget.

---

## Design decisions — the "why"

- **Two separate YOLOv8n models (object + barrier) instead of one combined model**: different object types, different frequencies of state change, and no benefit to forcing barrier-state and object-class into a shared label space. Also directly avoids running an unnecessary second model's compute cost on every frame — the barrier model gates whether the object model runs at all.
- **Geometric (aspect-ratio) barrier-state inference instead of a 3-class classifier**: reuses a single-class localizer's output, avoids training a second full classifier, and produces an explainable, empirically-calibrated decision rule rather than an opaque model boundary.
- **Track-ID-based event deduplication instead of per-frame logging or IoU-continuity matching**: ByteTrack IDs are already available from the object detector's tracking mode; anchoring the reported class to each track's highest-confidence frame resolves intra-track classification instability (a single physical object occasionally being misclassified between visually similar classes, e.g. rickshaw vs. two-wheeler, on individual frames).
- **Ensemble diagnostic (stock YOLOv8n for `person`, fine-tuned model for the rest) as a temporary unblocking step, not the shipped architecture**: used to confirm and quantify catastrophic forgetting during development; the production path is a single retrained, rebalanced model, since running two full detection models per frame is not viable on Jetson Nano's compute budget.
- **OCR fully decoupled from the edge pipeline**: plate reading is not time-critical for an evidentiary record, and keeping it off the Jetson preserves inference headroom for real-time detection, which is safety-critical.

---

## Possible extensions

- Networked live-feed streaming (MJPEG/RTSP) for remote-operator video access, beyond the current locally-connected-monitor design.
- Fine-tuned, India-specific plate recognizer to remove the dependency on a cloud OCR API.
- Direction-of-travel and speed estimation from track position history.
- Multi-crossing centralized dashboard.
