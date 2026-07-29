# scripts/barrier_state.py

from ultralytics import YOLO
import numpy as np
from collections import deque

BARRIER_MODEL_PATH = 'models/warden_barrier/weights/best.pt'
barrier_model = YOLO(BARRIER_MODEL_PATH)

# thresholds - to be tuned empirically against your labeled data (see below)
ASPECT_RATIO_CLOSED = 3.5   # ratio >= this => closed (horizontal)
ASPECT_RATIO_OPEN = 0.24     # ratio <= this => open (vertical)


def detect_barrier_box(frame, conf=0.4):
    """Returns the barrier bounding box [x1,y1,x2,y2] or None if not detected."""
    results = barrier_model.predict(frame, conf=conf, verbose=False)[0]
    if len(results.boxes) == 0:
        return None
    # if multiple detected (shouldn't happen with single-class + NMS), take highest confidence
    best_box = results.boxes[np.argmax(results.boxes.conf.cpu().numpy())]
    return best_box.xyxy[0].tolist()


def compute_aspect_ratio(box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if height == 0:
        return float('inf')
    return width / height


def classify_barrier_state(box):
    if box is None:
        return 'unknown'
    ratio = compute_aspect_ratio(box)
    if ratio >= ASPECT_RATIO_CLOSED:
        return 'closed'
    elif ratio <= ASPECT_RATIO_OPEN:
        return 'open'
    else:
        return 'closing'



class BarrierStateTracker:
    def __init__(self, window_size=5, min_agreement=3):
        self.history = deque(maxlen=window_size)
        self.min_agreement = min_agreement
        self.current_state = 'unknown'

    def update(self, box):
        state = classify_barrier_state(box)
        self.history.append(state)

        if len(self.history) >= self.min_agreement:
            counts = {s: list(self.history).count(s) for s in set(self.history)}
            most_common_state, count = max(counts.items(), key=lambda x: x[1])
            if count >= self.min_agreement:
                self.current_state = most_common_state

        return self.current_state