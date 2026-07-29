import os
from scripts.barrier_state import detect_barrier_box, compute_aspect_ratio
import cv2

# point this at your ORIGINAL (pre-merge) barrier annotations if you kept them,
# or manually sort a handful of sample images into open/closing/closed folders for this check
LABELED_DIRS = {
    'open': 'datasets/barrier_calibration/open/',
    'closing': 'datasets/barrier_calibration/closing/',
    'closed': 'datasets/barrier_calibration/closed/',
}

for state, dir_path in LABELED_DIRS.items():
    ratios = []
    for img_file in os.listdir(dir_path):
        frame = cv2.imread(os.path.join(dir_path, img_file))
        box = detect_barrier_box(frame)
        if box:
            ratios.append(compute_aspect_ratio(box))
    if ratios:
        print(f"{state}: min={min(ratios):.2f}, max={max(ratios):.2f}, mean={sum(ratios)/len(ratios):.2f}, n={len(ratios)}")